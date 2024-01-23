#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: StructrueTransformer.py
# Create time: 2023/12/13 15:23:14
# Author: Ji Hongchen

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cfg
import os
import math
import random
import shutil
from Dataprocess import *
from Transformer import *
from SupTrain import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# use_model = 'Attention_encoder_2D_Generator'
use_model = 'structure_encoder'

t = time.localtime()
modelsavename = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) + \
    '-' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + use_model

torch.backends.cudnn.enabled = False


P_DICT = cfg.par().proteindict
NODE_DICT = pd.read_csv('node_index.csv', names=['seq', 'count', 'ind'])
NODE_DICT.set_index('ind', inplace=True, drop=True)
LOAD_MODEL = 'model'

ORIGIN_SRC_ATTN = torch.load(
    CFG.resultspace + LOAD_MODEL + '/src_attentions_1.pt'
)
ORIGIN_TGT_ATTN = torch.load(
    CFG.resultspace + LOAD_MODEL + '/tgt_attentions_1.pt'
)


TRAIN_FOLDER = CFG.projectdata + 'b' + \
    str(cfg.DataPara().BATCHSIZE) + '_' + 'UnsupTrainBatch/'

NUM_TRAIN_VAL = int(len(os.listdir(TRAIN_FOLDER)) / 10)
NUM_TRAIN = int(len(os.listdir(TRAIN_FOLDER)) / 10 * 0.95)
VAL_LIST = range(
    int(len(os.listdir(TRAIN_FOLDER)) / 10 *
        0.95), int(len(os.listdir(TRAIN_FOLDER)) / 10)
)


def LassoLoss(y_pred, y_true, weights, lambda_=2):
    mse_loss = torch.nn.functional.smooth_l1_loss(
        y_pred, y_true, reduction='mean')
    l2_loss = lambda_ * torch.norm(weights, p=3)
    return mse_loss + l2_loss


class StructureTransformer(nn.Module):
    def __init__(self, Para=cfg.CNNPara()):
        super(StructureTransformer, self).__init__()
        self.Para = Para
        self.src_encoder = TCNN_Encoder()
        # ===========================================
        emodel_dict = self.src_encoder.state_dict()
        ori_dict = {
            k: v for k, v in ORIGIN_SRC_DICT.items(
            ) if k in emodel_dict
        }
        emodel_dict.update(ori_dict)
        self.src_encoder.load_state_dict(emodel_dict)
        for param in self.src_encoder.parameters():
            param.requires_grad = False
        # ===========================================
        self.fc_dim = cfg.TransformerPara().d_model * cfg.DataPara().MAXSEQLENGTH
        self.lr = nn.Linear(self.fc_dim, self.fc_dim)

        self.cnn_fc = nn.Sequential(
            nn.Linear(self.fc_dim, Para.linear_channel_2),
            nn.BatchNorm1d(Para.linear_channel_2),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_channel_2, Para.linear_channel_3),
            nn.BatchNorm1d(Para.linear_channel_3),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_channel_3, Para.linear_channel_2),
            nn.BatchNorm1d(Para.linear_channel_2),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_channel_2, self.fc_dim),
            nn.BatchNorm1d(self.fc_dim),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_dim * 2, self.fc_dim),
            nn.BatchNorm1d(self.fc_dim),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, src_seq, src_d, tgt_attn=ORIGIN_SRC_ATTN):
        enc_output, enc_attn = self.src_encoder(src_seq, src_d)
        enc_output = self.lr(enc_output)
        enc_output = self.cnn_fc(enc_output)
        enc_output = torch.cat(
            (enc_output, tgt_attn.sum(1).contiguous().view(
                cfg.DataPara().BATCHSIZE, -1)),
            dim=-1)
        enc_output = self.fc(enc_output)
        enc_output = enc_output.view(
            cfg.DataPara().BATCHSIZE,
            -1, cfg.TransformerPara().d_model).unsqueeze(1)

        return enc_output


def train(model, optimizer, criterion):

    best_v_loss = 1e9
    train_loss = []
    validation_losses = []
    model = model.cuda()
    model.train()

    for epoch in range(cfg.TrainPara().EPOCH):
        running_loss = 0
        for batch in random.sample(list(range(NUM_TRAIN)),
                                   int(NUM_TRAIN * cfg.TrainPara().TRAIN_SAMPLE_FRAC)):
            if cfg.TransformerPara().coding_method == 'embedding':
                tcr_d, epi_d, tcr_s, epi_s, _, _, label = DataLoder(
                    batch_folder=TRAIN_FOLDER,
                    batch_num=batch,
                )
                tcr_d, epi_d, tcr_s, epi_s, label = \
                    tcr_d.cuda(), epi_d.cuda(), tcr_s.cuda().long()[:, 1:], \
                    epi_s.cuda().long()[:, 1:], label.cuda()

            else:
                tcr_d, epi_d, tcr_s, epi_s, _, _, label = DataLoder(
                    batch_folder=TRAIN_FOLDER,
                    batch_num=batch,
                )
                tcr_d, epi_d, tcr_s, epi_s, label = \
                    tcr_d.cuda(), epi_d.cuda(), tcr_s.cuda().long()[:, 1:, :], \
                    epi_s.cuda().long()[:, 1:, :], label.cuda()

            # enc_output = model(tcr_s, tcr_d)
            enc_output = model(epi_s, epi_d)
            optimizer.zero_grad()
            loss = criterion(enc_output, tcr_d, model.lr.weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch % 200 == 0:
                print('. Epoch: %d, Step: %d, Loss: %.4f, Current LR: %.8f' %
                      (epoch + 1, batch + 1, loss.item(),
                       # optimizer._optimizer.state_dict()['param_groups'][0]['lr']))
                       optimizer.state_dict()['param_groups'][0]['lr']))
        losses = running_loss / NUM_TRAIN
        torch.save(losses, CFG.resultspace + modelsavename + '/train_loss.pt')

        train_loss.append(losses)

        print('-----> Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, cfg.TrainPara().EPOCH, losses / cfg.TrainPara().TRAIN_SAMPLE_FRAC))

        if epoch % 1 == 0:
            v_loss = validation(model, criterion)
            print('Validation loss: ', v_loss)
            print('Best validation accuracy: ', best_v_loss)
            print('================================================================')
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                torch.save(model, CFG.resultspace +
                           modelsavename + '/best_model.pt')

                print('****************************************************************')
                print('!!!!!!!!!!!!!!!!!!!!Model update!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('****************************************************************')
            validation_losses.append(v_loss)
    return model, train_loss, validation_losses


def validation(model, criterion):
    model.eval()
    total_validation_loss = 0
    with torch.no_grad():
        for v_batch in VAL_LIST:
            if cfg.TransformerPara().coding_method == 'embedding':
                tcr_v_d, epi_v_d, tcr_v_s, epi_v_s, _, _, label_v = DataLoder(
                    batch_folder=TRAIN_FOLDER,
                    batch_num=v_batch
                )
                tcr_v_d, epi_v_d, tcr_v_s, epi_v_s, label_v = \
                    tcr_v_d.cuda(), epi_v_d.cuda(), tcr_v_s.cuda(
                    ).long()[:, 1:], epi_v_s.cuda()[:, 1:].long(), label_v.cuda()

            else:
                tcr_v_d, epi_v_d, tcr_v_s, epi_v_s, _, _, label_v = DataLoder(
                    batch_folder=TRAIN_FOLDER,
                    batch_num=v_batch
                )
                tcr_v_d, epi_v_d, tcr_v_s, epi_v_s, label_v = \
                    tcr_v_d.cuda(), epi_v_d.cuda(), tcr_v_s.cuda(
                    ).long()[:, 1:, :], epi_v_s.cuda()[:, 1:, :].long(), label_v.cuda()

            # label_v = label_v.argmax(1)
            pred_v = model(epi_v_s, epi_v_d)
            loss = criterion(pred_v, tcr_v_d, model.lr.weight)
            total_validation_loss += loss.item()
            # total_acc += get_sup_accuracy(pred_v, label_v)

        v_loss = total_validation_loss / len(VAL_LIST)
        print('Total Validation Loss: ', v_loss)
    return v_loss


if __name__ == '__main__':
    # ==============================================
    ORIGIN_SRC_DICT = torch.load(
        CFG.resultspace + LOAD_MODEL + '/CATCR_D_1.pt'
    ).src_encoder.state_dict()
    ORIGIN_TGT_DICT = torch.load(
        CFG.resultspace + LOAD_MODEL + '/CATCR_D_1.pt'
    ).tgt_encoder.state_dict()
    ORIGIN_FC_DICT = torch.load(
        CFG.resultspace + LOAD_MODEL + '/CATCR_D_1.pt'
    ).fc.state_dict()
    # ==============================================
    if os.path.exists(CFG.resultspace + modelsavename) == False:
        os.makedirs(CFG.resultspace + modelsavename)
    shutil.copy('cfg.py', CFG.resultspace + modelsavename + '/cfg.py')
    print('Use model: ', use_model)
    st = StructureTransformer()
    train(st,
          optimizer=optim.SGD(params=st.parameters(), lr=cfg.TrainPara().LR),
          criterion=LassoLoss,
          # criterion=nn.L1Loss(reduction='mean'),
          )
