#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Ji Hongchen
# @Email : jhca.xyt@163.com
# @Last version : 2023-08-19 12:03
# @Filename : cfg.py

import os
import pandas as pd

# sturc: "structure"
# ENCD: "encode"
# DECD: "decode"

PROJECT = 'TcrPredict'


class Cfg():
    def __init__(self):
        self.workspace = '/home/ji/Documents/'
        self.publicdata = '/home/ji/Documents/Data/PublicData/'
        self.extradata = '/media/ji/BI/TcrPredict/'
        self.projectdata = self.workspace + '/Data/ProjectData/' + PROJECT + '/'
        self.codespace = self.workspace + 'Code/' + PROJECT + '/'
        self.resultspace = self.workspace + 'Result/' + PROJECT + '/'
        self.figuerspace = self.workspace + 'Figure/' + PROJECT + '/'
        # self.batchspace = self.projectdata + 'TrainBatch/'
        self.allstructrure = '/media/ji/BI/OpenFold/PredictedStructure/predictions/'
        self.strucspace = self.projectdata + 'PdbStructure/'
        self.model_batches = [
            self.projectdata + 'UnsupTrainBatch/',
            self.projectdata + 'UnsupTestBatch/',
            self.projectdata + 'SupTrainBatch/',
            self.projectdata + 'SupTestBatch/'
        ]


class DataPara():
    def __init__(self):
        CFG = Cfg()
        self.MAXSEQLENGTH = 25
        self.PEPDIM = 26
        self.BATCHSIZE = 128
        # self.BATCHSIZE = 128
        self.dropout = 0.5
        self.datatype_seq = 'onehot'
        self.datatype_dist = 'normal'
        # self.train_method = 'Super'
        # self.train_method = 'Unsuper'
        self.BEAM_SIZE = 5
        # self.CORRFILE = pd.read_csv(CFG.projectdata + 'TCR-PEP-MHC_labeled.csv')
        # self.CORRFILE = pd.read_csv(CFG.projectdata + 'TCR-PEP-MHC.csv')


class TrainPara():
    def __init__(self):
        CFG = Cfg()
        self.EPOCH = 10000
        self.LR = 0.08
        # self.LOSSFUNC = 'mse'
        # self.OPTIMFUNC = 'Adam'
        self.LOSSWEIGHT = 1.0
        self.WEIGHTDECAY = 0  # 1e-7
        self.TRAIN_SAMPLE_FRAC = 0.2


class CNNPara():
    def __init__(self, N_kernal=3):
        # N_kernal: number of kernal in CNN
        CFG = Cfg()
        self.CNN_layer_1_channel = 16
        self.CNN_layer_1_kernel_size = N_kernal
        self.CNN_layer_2_channel = self.CNN_layer_1_channel * 2
        self.CNN_layer_2_kernel_size = N_kernal
        self.MP_layer_1_kernel_size = 2
        self.MP_layer_1_stride = 1
        self.MP_layer_2_kernel_size = 2
        self.MP_layer_2_stride = 1

        if DataPara().datatype_dist == 'transformer':
            self.linear_mut_mun = 25 - ((N_kernal - 1) * 2)
        else:
            self.linear_mut_mun = 24 - ((N_kernal - 1) * 2)
        self.linear_channel_1 = 1024
        self.linear_channel_2 = 256
        self.linear_channel_3 = 32
        self.output_dim = 2
        self.transformer_output_dim = 64
        self.dropout_value = DataPara().dropout
        # self.linear_channel_1 = int(1024 / pow(2, (N_kernal - 2)))
        # if N_kernal - 2 < 0:
        #    self.linear_channel_1 = 1024
        # self.linear_channel_2 = 16


class GRUEncoderPara():
    def __init__(self):
        CFG = Cfg()
        self.input_dim = DataPara().PEPDIM
        self.hidden_dim = 512
        self.hidden_layers = 2
        self.middle_dim = 16
        # self.output_dim = DataPara().PEPDIM
        self.output_dim = 2
        self.bidirectional = True
        self.dropout_value = DataPara().dropout


class GRUDecoderPara():
    def __init__(self):
        encoder_Para = GRUEncoderPara()
        self.input_dim = encoder_Para.input_dim
        self.hidden_dim = encoder_Para.hidden_dim
        self.hidden_layers = encoder_Para.hidden_layers
        self.middle_dim = 128
        self.output_dim = encoder_Para.output_dim
        self.bidirectional = encoder_Para.bidirectional
        self.dropout = encoder_Para.dropout
        self.cnn_output_dim_more = CNNPara().linear_channel_2
        self.cnn_output_dim_less = CNNPara().linear_channel_3


class TransformerPara():
    def __init__(self):
        CFG = Cfg()
        self.coding_method = 'embedding'  # 'blosum'
        self.seq_len = DataPara().MAXSEQLENGTH + 1
        # self.d_word_vocab = DataPara().PEPDIM - 1
        self.d_word_vocab = 908
        self.random_mask_frac = 0.4
        if self.coding_method == 'embedding':
            self.d_model = 25
        else:
            self.d_model = 25
        self.beam_size = 7
        self.d_ff = 1024
        self.d_k = 64
        self.d_v = 64
        self.n_layers = 6
        self.n_heads = 8
        self.src_len = 26
        self.tgt_len = 26
        self.attn_dropout = 0.4
        self.init_lr = TrainPara().LR
        self.n_warmup_step = 2000
        self.vital_frac = 1000
        self.cross_weight = False
        self.gru_layer_num = 2

        self.linear_1 = 256
        self.linear_2 = 32
        self.linear_3 = 8
        self.linear_4 = 4
        self.output_dim = 2


class GeneratorPara():
    def __init__(self):
        CFG = Cfg()
        self.linear_1 = 256
        self.linear_2 = 64
        self.linear_3 = 16
        self.linear_4 = 4


class par():
    def __init__(self):
        self.proteindict = {
            0: 'A',
            1: 'R',
            2: 'N',
            3: 'D',
            4: 'C',
            5: 'Q',
            6: 'E',
            7: 'G',
            8: 'H',
            9: 'I',
            10: 'L',
            11: 'K',
            12: 'M',
            13: 'F',
            14: 'P',
            15: 'S',
            16: 'T',
            17: 'W',
            18: 'Y',
            19: 'V',
            20: 'B',
            21: 'J',
            22: 'Z',
            23: 'X',
            24: '*',
            25: '!'
        }
        self.basedict = {
            'A': 0,
            'T': 1,
            'C': 2,
            'G': 3
        }
        self.AAdict = {
            'ALA': 'A',
            'PHE': 'F',
            'CYS': 'C',
            'SEC': 'U',
            'ASP': 'D',
            'ASN': 'N',
            'GLU': 'E',
            'GLN': 'Q',
            'GLY': 'G',
            'HIS': 'H',
            'LEU': 'L',
            'ILE': 'I',
            'LYS': 'K',
            'PYL': 'O',
            'MET': 'M',
            'PRO': 'P',
            'ARG': 'R',
            'SER': 'S',
            'THR': 'T',
            'VAL': 'V',
            'TRP': 'W',
            'TYR': 'Y'
        }
