#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Ji Hongchen
# @Email : jhca.xyt@163.com
# @Last version : 2023-08-20 01:00
# @Filename : train.py

import time
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from random import sample
import torch.nn.functional as F
import cfg
import os
import math
import random
import shutil
from Dataprocess import *
from Transformer import *

# use_model = 'GRU'
# use_model = 'CNN'
# use_model = 'TRANSFORMER'
# use_model = 'T_encoder'
# use_model = 'CNN_encoder'
use_model = 'TCNN_encoder'
# use_model = 'AttentionCNN_encoder'
# use_model = 'TCNN_cross_encoder'


t = time.localtime()
modelsavename = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) + \
    '-' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + use_model

torch.backends.cudnn.enabled = False

CFG = cfg.Cfg()
TRAIN_FOLDER = CFG.projectdata + 'b' + \
    str(cfg.DataPara().BATCHSIZE) + '_' + 'SupTrainBatch/'

NUM_TRAIN_VAL = int(len(os.listdir(TRAIN_FOLDER)) / 10)
NUM_TRAIN = int(len(os.listdir(TRAIN_FOLDER)) / 10 * 0.95)
VAL_LIST = range(
    int(len(os.listdir(TRAIN_FOLDER)) / 10 *
        0.95), int(len(os.listdir(TRAIN_FOLDER)) / 10)
)


def get_sup_accuracy(pred, label, mode='max'):  # mode: 'round' or 'max
    if mode == 'max':
        # return ((pred.argmax(1) == label.argmax(1)).sum() / cfg.DataPara().BATCHSIZE).item()
        return ((pred.argmax(1) == label.argmax(1)).sum() / cfg.DataPara().BATCHSIZE).item()
    elif mode == 'round':
        return ((pred.round().int().squeeze(1) == label).sum() / cfg.DataPara().BATCHSIZE).item()


def LassoLoss(y_pred, y_true, weights, lambda_=2):
    mse_loss = torch.nn.functional.cross_entropy(
        y_pred, y_true, reduction='mean')
    l2_loss = lambda_ * torch.norm(weights, p=3)
    return mse_loss + l2_loss


class CNN_only(nn.Module):
    def __init__(self, CNN_Para=cfg.CNNPara(), GRU_Para=cfg.GRUEncoderPara()):
        super(CNN_only, self).__init__()
        self.cnn_tcr = CNN_part(CNN_Para=cfg.CNNPara())
        self.cnn_epi = CNN_part(CNN_Para=cfg.CNNPara())
        self.batch_size = cfg.DataPara().BATCHSIZE
        self.target_len = cfg.DataPara().MAXSEQLENGTH
        self.output_dim = cfg.DataPara().PEPDIM
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(
                CNN_Para.linear_channel_2 * 2,
                CNN_Para.linear_channel_3
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(CNN_Para.linear_channel_3),
            nn.Dropout(CNN_Para.dropout_value),

            nn.Linear(
                CNN_Para.linear_channel_3,
                CNN_Para.output_dim
            ),
            nn.BatchNorm1d(CNN_Para.output_dim),
            # nn.Dropout(CNN_Para.dropout_value),
            nn.Sigmoid(),
        )

    def forward(self, tcr_dist, epi_dist, tcr_seq, epi_seq):
        tcr_output = self.cnn_tcr(tcr_dist)
        epi_output = self.cnn_epi(epi_dist)
        output = torch.cat((tcr_output,
                            epi_output), dim=1)
        output = self.fc(output)
        # output = self.sigmoid(output)
        return output


class CNN_combination(nn.Module):
    def __init__(self, CNN_Para=cfg.CNNPara(), GRU_Para=cfg.GRUEncoderPara()):
        super(CNN_combination, self).__init__()
        self.cnn_tcr_1 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=1))
        self.cnn_epi_1 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=1))
        self.cnn_tcr_3 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=3))
        self.cnn_epi_3 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=3))
        self.cnn_tcr_7 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=7))
        self.cnn_epi_7 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=7))
        self.batch_size = cfg.DataPara().BATCHSIZE
        self.target_len = cfg.DataPara().MAXSEQLENGTH
        self.output_dim = cfg.DataPara().PEPDIM
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(
                CNN_Para.linear_channel_2,
                CNN_Para.linear_channel_3
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(CNN_Para.linear_channel_3),
            nn.Dropout(CNN_Para.dropout_value),

            nn.Linear(
                CNN_Para.linear_channel_3,
                CNN_Para.output_dim
            ),
            nn.BatchNorm1d(CNN_Para.output_dim),
            nn.Dropout(CNN_Para.dropout_value),
            nn.Sigmoid(),
        )

    def forward(self, tcr_dist, epi_dist, tcr_seq, epi_seq):
        tcr_output_1 = self.cnn_tcr_1(tcr_dist)
        epi_output_1 = self.cnn_epi_1(epi_dist)
        tcr_output_3 = self.cnn_tcr_3(tcr_dist)
        epi_output_3 = self.cnn_epi_3(epi_dist)
        tcr_output_7 = self.cnn_tcr_7(tcr_dist)
        epi_output_7 = self.cnn_epi_7(epi_dist)
        output = torch.add(
            tcr_output_1,
            epi_output_1,
        )
        output = torch.add(output, tcr_output_3)
        output = torch.add(output, epi_output_3)
        output = torch.add(output, tcr_output_7)
        output = torch.add(output, epi_output_7)

        output = self.fc(output)
        return output


class TransformerSup(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(TransformerSup, self).__init__()
        self.Para = Para

        # Encoder part:
        if use_model == 'T_encoder':
            self.src_encoder = T_Encoder()
            self.tgt_encoder = T_Encoder()
        elif use_model == 'CNN_encoder':
            self.src_encoder = CNN_Encoder()
            self.tgt_encoder = CNN_Encoder()
        elif use_model == 'TCNN_encoder':
            self.src_encoder = TCNN_Encoder()
            self.tgt_encoder = TCNN_Encoder()
        elif use_model == 'AttentionCNN_encoder':
            self.src_encoder = AttentionCNN_Encoder()
            self.tgt_encoder = AttentionCNN_Encoder()
        elif use_model == 'TCNN_cross_encoder':
            self.src_encoder = TCNN_cross_Encoder()
            self.tgt_encoder = TCNN_cross_Encoder()

        self.fc = nn.Sequential(
            nn.Linear(Para.d_model * cfg.DataPara().MAXSEQLENGTH *
                      2, Para.linear_1),
            nn.BatchNorm1d(Para.linear_1),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_1, Para.linear_2),
            nn.BatchNorm1d(Para.linear_2),
            nn.Dropout(cfg.DataPara().dropout),
            nn.Sigmoid(),

            nn.Linear(Para.linear_2, Para.linear_3),
            nn.BatchNorm1d(Para.linear_3),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.linear_3, Para.output_dim),
            nn.BatchNorm1d(Para.output_dim),
            nn.Dropout(cfg.DataPara().dropout),
            nn.Sigmoid(),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_d, tgt_d, src_seq, tgt_seq):
        src_output, src_attns = self.src_encoder(src_seq, src_d)
        src_output = src_output.view(cfg.DataPara().BATCHSIZE, -1)
        tgt_output, tgt_attns = self.tgt_encoder(tgt_seq, tgt_d)
        tgt_output = tgt_output.view(cfg.DataPara().BATCHSIZE, -1)

        output = torch.cat((src_output, tgt_output), dim=1)

        output = self.fc(output)

        return output, src_attns[-1], tgt_attns[-1]


def train(model, optimizer, criterion):

    best_v_acc = 0
    train_loss = []
    validation_losses = []
    model = model.cuda()
    model.train()

    for epoch in range(cfg.TrainPara().EPOCH):
        running_loss = 0
        # for batch in range(NUM_TRAIN):
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

            # tcr_s = random_mask(tcr_s)
            # epi_s = random_mask(epi_s)
            optimizer.zero_grad()
            pred, src_attn, tgt_attn = model(tcr_d, epi_d, tcr_s, epi_s)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch % 200 == 0:
                print('. Epoch: %d, Step: %d, Loss: %.4f, Current LR: %.8f' %
                      (epoch + 1, batch + 1, loss.item(),
                       # optimizer._optimizer.state_dict()['param_groups'][0]['lr']))
                       optimizer.state_dict()['param_groups'][0]['lr']))
        losses = running_loss / NUM_TRAIN

        train_loss.append(losses)

        print('-----> Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, cfg.TrainPara().EPOCH, losses / cfg.TrainPara().TRAIN_SAMPLE_FRAC))

        if epoch % 1 == 0:
            v_loss, v_acc = validation(model, criterion)
            validation_losses.append(v_loss)
            torch.save(train_loss, CFG.resultspace +
                       modelsavename + '/train_loss.pt')
            torch.save(validation_losses, CFG.resultspace +
                       modelsavename + '/val_loss.pt')
            torch.save(model, CFG.resultspace +
                       modelsavename + '/final_model.pt')
            print('Validation accuracy: ', v_acc)
            print('Best validation accuracy: ', best_v_acc)
            print('================================================================')
            if v_acc > best_v_acc:
                best_v_acc = v_acc
                torch.save(model, CFG.resultspace +
                           modelsavename + '/best_model.pt')
                torch.save(src_attn, CFG.resultspace +
                           modelsavename + '/src_attentions.pt')
                torch.save(tgt_attn, CFG.resultspace +
                           modelsavename + '/tgt_attentions.pt')
                print('****************************************************************')
                print('!!!!!!!!!!!!!!!!!!!!Model update!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('****************************************************************')
            validation_losses.append(v_loss)
    return model, train_loss, validation_losses


def validation(model, criterion):
    model.eval()
    total_validation_loss = 0
    total_acc = 0
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

            pred_v, *_ = model(tcr_v_d, epi_v_d, tcr_v_s, epi_v_s)
            loss = criterion(pred_v, label_v)
            total_validation_loss += loss.item()
            total_acc += get_sup_accuracy(pred_v, label_v)

        v_loss = total_validation_loss / len(VAL_LIST)
        avr_v_acc = total_acc / len(VAL_LIST)
        print('Total Validation Loss: ', v_loss)
    return v_loss, avr_v_acc


if __name__ == '__main__':
    if os.path.exists(CFG.resultspace + modelsavename) == False:
        os.makedirs(CFG.resultspace + modelsavename)
    shutil.copy('cfg.py', CFG.resultspace + modelsavename + '/cfg.py')
    print('Use model: ', use_model)

    train_model = TransformerSup()

    train(
        train_model,
        optimizer=optim.SGD(train_model.parameters(),
                            lr=cfg.TrainPara().LR,
                            weight_decay=cfg.TrainPara().WEIGHTDECAY),
        criterion=nn.CrossEntropyLoss(reduction='mean')
    )
