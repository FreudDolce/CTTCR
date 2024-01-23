#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: D_test.py
# Create time: 2024/01/23 19:58:46
# Author: Ji Hongchen

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
from SupTrain import *

# use_model = 'GRU'
# use_model = 'CNN'
# use_model = 'TRANSFORMER'
# use_model = 'T_encoder'
use_model = 'CNN_encoder'
# use_model = 'TCNN_encoder'
# use_model = 'AttentionCNN_encoder'
# use_model = 'TCNN_cross_encoder'

TEST_FOLDER = './demo_data/b128_SupTestBatch/'
NUM_TEST = int(len(os.listdir(TEST_FOLDER)) / 10)

t = time.localtime()
modelsavename = str(t.tm_year) + '-' + str(t.tm_mon) + '-' + str(t.tm_mday) + \
    '-' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + use_model

torch.backends.cudnn.enabled = False

test_model = torch.load(
    CFG.resultspace + 'model/CATCR_D_1.pt'
)


def get_sup_accuracy(pred, label, mode='max'):  # mode: 'round' or 'max
    if mode == 'max':
        # return ((pred.argmax(1) == label.argmax(1)).sum() / cfg.DataPara().BATCHSIZE).item()
        return ((pred.argmax(1) == label.argmax(1)).sum() / cfg.DataPara().BATCHSIZE).item()
    elif mode == 'round':
        return ((pred.round().int().squeeze(1) == label).sum() / cfg.DataPara().BATCHSIZE).item()


def test(model):
    model.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for t_batch in range(NUM_TEST):
            if cfg.TransformerPara().coding_method == 'embedding':
                tcr_t_d, epi_t_d, tcr_t_s, epi_t_s, _, _, label_t = DataLoder(
                    batch_folder=TEST_FOLDER,
                    batch_num=t_batch
                )
                tcr_t_d, epi_t_d, tcr_t_s, epi_t_s, label_t = \
                    tcr_t_d.cuda(), epi_t_d.cuda(), tcr_t_s.cuda(
                    ).long()[:, 1:], epi_t_s.cuda()[:, 1:].long(), label_t.cuda()

            else:
                tcr_t_d, epi_t_d, tcr_t_s, epi_t_s, _, _, label_t = DataLoder(
                    batch_folder=TEST_FOLDER,
                    batch_num=t_batch
                )
                tcr_t_d, epi_t_d, tcr_t_s, epi_t_s, label_t = \
                    tcr_t_d.cuda(), epi_t_d.cuda(), tcr_t_s.cuda(
                    ).long()[:, 1:, :], epi_t_s.cuda()[:, 1:, :].long(), label_t.cuda()

            pred_t, *_ = model(tcr_t_d, epi_t_d, tcr_t_s, epi_t_s)
            total_acc += get_sup_accuracy(pred_t, label_t)
            print('Batch: ', t_batch, ' Loss: ',
                  get_sup_accuracy(pred_t, label_t))

        avr_t_acc = total_acc / NUM_TEST
        print('Final accuracy: ', avr_t_acc)
        print('----------------------------------------------------------------')
    return avr_t_acc


if __name__ == '__main__':
    test_acc = test(test_model)
