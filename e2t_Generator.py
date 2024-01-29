#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: Generator.py
# Create time: 2023/12/23 00:17:32
# Author: Ji Hongchen

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
import cfg
import copy
import time
import math
import random
import difflib
from Transformer import *
from Dataprocess import *
from StructrueTransformer import *
from SupTrain import *
from NewCodingMethod import GenerateNewCodeSeq

# use_model = 'TCNN_Encoder_T_Decoder'
use_model = 'TCNN_Encoder_STRUC_Decoder'


RETURN_DLOSS = False

P_DICT = cfg.par().proteindict
NODE_DICT = pd.read_csv('node_index.csv', names=['seq', 'count', 'ind'])
NODE_DICT.set_index('ind', inplace=True, drop=True)
node_index = pd.read_csv('node_index.csv', names=['seq', 'count', 'ind'])
node_index['seqlen'] = node_index['seq'].str.len()
node_index.sort_values(by=['seqlen', 'count'], inplace=True, ascending=False)
print(node_index)
node_index['liststr'] = 'a'
node_index.set_index('seq', inplace=True, drop=True)
for i in node_index.index:
    node_index['liststr'].loc[i] = list(i)

t = time.localtime()
modelsavename = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) + \
    '-' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + use_model

torch.backends.cudnn.enabled = False
c = copy.deepcopy

# load model from supervised training model.
# ==============================================
ORIGIN_DIS_DICT = torch.load('./model/CATCR_D.pt').state_dict()
ORIGIN_SRC_DICT = torch.load('./model/CATCR_D.pt').src_encoder.state_dict()
ORIGIN_SRC_ATTN = torch.load('./model/src_attentions_1.pt').sum(1)
STRUC_MODEL = torch.load('./model/RCMT.pt')
for param in STRUC_MODEL.parameters():
    param.requires_grad = False
# ==============================================

# load Data from the following folder:
# ==============================================
TRAIN_FOLDER = CFG.projectdata + 'b' + \
    str(cfg.DataPara().BATCHSIZE) + '_' + 'UnsupTrainBatch/'
# ==============================================

NUM_TRAIN_VAL = int(len(os.listdir(TRAIN_FOLDER)) / 10)
NUM_TRAIN = int(len(os.listdir(TRAIN_FOLDER)) / 10 * 0.95)
VAL_LIST = range(
    int(len(os.listdir(TRAIN_FOLDER)) / 10 *
        0.95), int(len(os.listdir(TRAIN_FOLDER)) / 10)
)


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    # gold = F.softmax(gold, dim=1)

    if smoothing:
        eps = 0.05
        # n_class = pred.size(1)
        n_class = cfg.DataPara().PEPDIM - 1

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        # loss = -(one_hot * log_prb).sum()
        loss = -(one_hot * log_prb).mean()
    else:
        loss = F.cross_entropy(pred, gold)
    return loss


def get_match_ratio(ori_seq, decoded_seq, ignore):
    match_ratio = difflib.SequenceMatcher(
        lambda x: x == ignore, ori_seq, decoded_seq
    ).ratio()
    return match_ratio


def read_sequence(seq):
    return ''.join([P_DICT[k.item()] for k in seq])


def extract_sequence(seq):  # mode: max, derict, node
    decoded_seq = ''.join([NODE_DICT['seq'][k.item()] for k in seq])
    return decoded_seq.split('*')[0]


def extract_batch_seq(seq_list):
    neo_batch_seq = []
    for seq in seq_list:
        neo_batch_seq.append(extract_sequence(seq))
    return neo_batch_seq


class ScheduleOptim():

    def __init__(self, optimizer, Para=cfg.TransformerPara()):
        self._optimizer = optimizer
        self.init_lr = Para.init_lr
        self.d_model = Para.d_model
        self.n_warmup_step = Para.n_warmup_step
        self.n_step = 0

    def step_and_update_lr(self):
        # Step with the inner optimizer
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_step, n_warmup_steps = self.n_step, self.n_warmup_step
        return (d_model ** -0.5) * min(n_step ** (-0.5), n_step * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        self.n_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class DLoss(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(DLoss, self).__init__()
        self.dmodel = TransformerSup()

        # load state_dict, notice src or tgt
        # ===========================================
        dmodel_dict = self.dmodel.state_dict()
        ori_dict = {
            k: v for k, v in ORIGIN_DIS_DICT.items(
            ) if k in dmodel_dict
        }
        dmodel_dict.update(ori_dict)
        self.dmodel.load_state_dict(dmodel_dict)
        for param in self.dmodel.parameters():
            param.requires_grad = False
        # ===========================================

    def forward(self, src_d, tgt_d, src_seq, tgt_seq):
        dmodel_output, *_ = self.dmodel(src_d, tgt_d, src_seq, tgt_seq)

        return torch.mean(dmodel_output[:, 1] - dmodel_output[:, 0])


class Generator(nn.Module):
    def __init__(self, Para=cfg.TransformerPara(), weight_sharing=True):
        super(Generator, self).__init__()
        self.Para = Para

        # Chose the encoder
        self.encoder = TransformerSup().src_encoder
        if 'T_Decoder' in use_model:
            self.decoder = T_Decoder()
        elif 'STRUC_Decoder' in use_model:
            self.decoder = STRUC_Decoder()

        self.projection = nn.Sequential(
            nn.Linear(Para.d_model, Para.linear_2, bias=False),
            nn.BatchNorm1d(Para.linear_2),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_2, Para.linear_1, bias=False),
            nn.BatchNorm1d(Para.linear_1),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_1, Para.linear_1 * 2, bias=False),
            nn.BatchNorm1d(Para.linear_1 * 2),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_1 * 2, Para.linear_1 * 2, bias=False),
            nn.BatchNorm1d(Para.linear_1 * 2),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_1 * 2, Para.d_word_vocab, bias=False),
            nn.BatchNorm1d(Para.d_word_vocab),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                # nn.init.kaiming_normal_(p)

        # load state_dict, notice src or tgt
        # ===========================================
        encoder_dict = self.encoder.state_dict()
        ori_dict = {
            k: v for k, v in ORIGIN_SRC_DICT.items(
            ) if k in encoder_dict
        }
        encoder_dict.update(ori_dict)
        self.encoder.load_state_dict(encoder_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
        # ===========================================

        # if weight_sharing == True:
        #     self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight
        #     self.x_logit_scale = (Para.d_model ** -0.5)

    def forward(self, src_seq, tgt_seq, src_d=None, tgt_d=None):
        src_mask = None
        tgt_mask = get_subsequence_mask(tgt_seq)
        enc_output, enc_attn = self.encoder(src_seq, src_d, src_mask)
        enc_output = enc_output.view(enc_output.size(0),
                                     -1,
                                     cfg.TransformerPara().d_model)

        dec_output, dec_attn, dec_enc_attn = self.decoder(
            tgt_seq, tgt_mask, enc_output, src_mask, src_d, tgt_d
        )
        dec_output = dec_output.contiguous().view(-1, self.Para.d_model)
        dec_output = self.projection(dec_output)
        return dec_output, enc_attn, dec_attn, dec_enc_attn


def train(model, s_model, optimizer, criterion, dloss=DLoss().cuda()):
    best_v_loss = 1e9
    train_loss = []
    validation_loss = []
    model = model.cuda()
    s_model = s_model.cuda()
    model.train()
    s_model.train()
    for epoch in range(cfg.TrainPara().EPOCH):
        running_loss = 0
        for batch in random.sample(list(range(NUM_TRAIN)),
                                   int(NUM_TRAIN * cfg.TrainPara().TRAIN_SAMPLE_FRAC)):

            _, epi_d, tcr_s, epi_s, tcr_s_label, epi_s_label, _ = DataLoder(
                batch_folder=TRAIN_FOLDER,
                batch_num=batch,
            )
            epi_d = epi_d.cuda()
            tcr_s = tcr_s.cuda().long()
            epi_s = epi_s[:, 1:].cuda().long()
            tcr_d = s_model(epi_s, epi_d)
            tcr_s_input = tcr_s[:, :-1]
            tcr_s_label = tcr_s[:, 1:]
            tcr_s_label = tcr_s_label.contiguous().view(-1)

            optimizer.zero_grad()

            decode_output, e_atn, d_atn, d_e_atn = model(
                epi_s, tcr_s_input, epi_d, tcr_d)
            extracted_decode_output = decode_output.max(
                1).indices.view(cfg.DataPara().BATCHSIZE, -1)

            D_loss = dloss(tcr_d, epi_d, extracted_decode_output, epi_s)
            loss = criterion(decode_output, tcr_s_label) + (1 - D_loss) * 0.5
            loss.backward()

            optimizer.step_and_update_lr()

            running_loss += loss.item()
            # if running_loss < 1e-9:
            #     break
            if batch % 100 == 0:
                print('. Epoch: %d, Step: %d, Loss: %.4f, Current LR: %.8f' %
                      (epoch + 1, batch, loss.item(),
                       optimizer._optimizer.state_dict()['param_groups'][0]['lr']))

        losses = running_loss / NUM_TRAIN
        train_loss.append(losses)

        print('-----> Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, cfg.TrainPara().EPOCH, losses / cfg.TrainPara().TRAIN_SAMPLE_FRAC))
        if epoch % 1 == 0:
            batch_v_loss = validation(model, s_model, criterion, dloss)
            validation_loss.append(batch_v_loss)
            torch.save(model, CFG.resultspace +
                       modelsavename + '/latest_model.pt')
            torch.save(s_model, CFG.resultspace +
                       modelsavename + '/latest_struc_model.pt')
            torch.save(train_loss, CFG.resultspace +
                       modelsavename + '/train_loss.pt')
            torch.save(validation_loss, CFG.resultspace +
                       modelsavename + '/val_loss.pt')

            if batch_v_loss < best_v_loss:
                best_v_loss = batch_v_loss
                torch.save(model, CFG.resultspace +
                           modelsavename + '/best_model.pt')
                print('****************************************************************')
                print('!!!!!!!!!!!!!!!!!!!!!!Model update!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('****************************************************************')
            print('Epoch ', epoch, ', Validation loss: ', batch_v_loss)
            print('----------------------------------------------------------------')
            print('Best Validation loss: ', best_v_loss)

        print('****************************************************************')
    return model, losses, validation_loss


def validation(model, s_model, criterion, dloss):
    model.eval()
    s_model.eval()
    total_validation_loss = 0
    with torch.no_grad():
        for v_batch in VAL_LIST:
            _, epi_v_d, tcr_v_s, epi_v_s, tcr_v_s_label, epi_v_s_label, _ = DataLoder(
                batch_folder=TRAIN_FOLDER,
                batch_num=v_batch
            )
            epi_v_d = epi_v_d.cuda()
            tcr_v_s = tcr_v_s.cuda().long()
            epi_v_s = epi_v_s[:, 1:].cuda().long()
            tcr_v_d = s_model(epi_v_s, epi_v_d)
            tcr_v_s_input = tcr_v_s[:, :-1]
            tcr_v_s_label = tcr_v_s[:, 1:].cuda(
            ).long().contiguous().view(-1)

            decode_v_output, *_ = model(
                epi_v_s, tcr_v_s_input, epi_v_d, tcr_v_d)

            extracted_decode_output = decode_v_output.max(
                1).indices.view(cfg.DataPara().BATCHSIZE, -1)
            D_v_loss = dloss(tcr_v_d, epi_v_d,
                             extracted_decode_output, epi_v_s)

            loss_v = criterion(
                decode_v_output, tcr_v_s_label) + (1 - D_v_loss) * 0.5
            total_validation_loss += loss_v.item()

            decode_v_output = decode_v_output.view(
                cfg.DataPara().BATCHSIZE, cfg.DataPara().MAXSEQLENGTH, -1
            )
            tcr_v_s_label = tcr_v_s_label.view(cfg.DataPara().BATCHSIZE, -1)
            for k in range(cfg.DataPara().BATCHSIZE):
                if np.random.randint(0, cfg.TransformerPara().vital_frac) == 1:
                    ori_seq = extract_sequence(tcr_v_s_label[k])
                    val_seq = extract_sequence(
                        decode_v_output[k].max(1).indices)
                    print('Origin Sequence: ', ori_seq)
                    print('Valida Sequence: ', val_seq)
                    print('Validation match: ', get_match_ratio(
                        ori_seq, val_seq, '*'))
        batch_v_loss = total_validation_loss / len(VAL_LIST)
        print('----------------------------------------------------------------')
        print('Total Validation Loss: ', batch_v_loss)
        print('----------------------------------------------------------------')
    return batch_v_loss


if __name__ == '__main__':
    if os.path.exists(CFG.resultspace + modelsavename) == False:
        os.makedirs(CFG.resultspace + modelsavename)
    shutil.copy('cfg.py', CFG.resultspace + modelsavename + '/cfg.py')

    print('Use model: ', use_model)
    train_model = Generator()

    model, train_loss, validation_loss = train(
        train_model,
        STRUC_MODEL,
        optimizer=ScheduleOptim(optim.Adam(
            train_model.parameters(), betas=[0.9, 0.98])),
        # optimizer=ScheduleOptim(optim.SGD(
        #     train_model.parameters(), lr=cfg.TrainPara().LR)),
        # criterion=cal_loss
        criterion=nn.CrossEntropyLoss(reduction='mean')
    )

    torch.save(model, CFG.resultspace + modelsavename + '/model.pt')
    torch.save(train_loss, CFG.resultspace + modelsavename + '/train_loss.pt')
    torch.save(validation_loss, CFG.resultspace +
               modelsavename + '/validation_loss.pt')
