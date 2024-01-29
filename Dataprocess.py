#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: Dataprocess.py
# Create time: 2023/07/16 20:14:03
# Author: Ji Hongchen

import pandas as pd
import numpy as np
import cfg
import os
import torch

CFG = cfg.Cfg()
Blosum62 = pd.read_csv(
    CFG.publicdata + 'Blosum62.csv', index_col=[0]
)
Blosum62_c = list(Blosum62.columns)


def DataLoder(batch_folder, batch_num,
              coding_method=cfg.DataPara().datatype_seq):
    tcr_dist = np.load(
        batch_folder + 'dist_batch_' + str(batch_num) + '.npy'
    )
    epi_dist = np.load(
        batch_folder + 'dist_y_batch_' + str(batch_num) + '.npy'
    )

    sup_label = np.load(
        batch_folder + 'ol_batch_' + str(batch_num) + '.npy'
    )
    tcr_label = np.load(
        batch_folder + 'ori_seq_batch_' + str(batch_num) + '.npy'
    )
    epi_label = np.load(
        batch_folder + 'label_seq_batch_' + str(batch_num) + '.npy'
    )

    if coding_method == 'blosum':
        tcr_seq = np.load(
            batch_folder + 'seq_batch_b_' + str(batch_num) + '.npy'
        )
        epi_seq = np.load(
            batch_folder + 'seq_y_batch_b_' + str(batch_num) + '.npy'
        )
    else:
        tcr_seq = np.load(
            batch_folder + 'seq_batch_o_' + str(batch_num) + '.npy'
        )
        epi_seq = np.load(
            batch_folder + 'seq_y_batch_o_' + str(batch_num) + '.npy'
        )

    tcr_dist = torch.Tensor(tcr_dist)  # , dtype=torch.float32)
    epi_dist = torch.Tensor(epi_dist)  # , dtype=torch.float32)
    sup_label = torch.Tensor(sup_label)  # , dtype=torch.float32)
    epi_label = torch.Tensor(epi_label)  # , dtype=
    tcr_label = torch.Tensor(tcr_label)
    try:
        tcr_seq = torch.Tensor(tcr_seq)  # , dtype=torch.float32)
        epi_seq = torch.Tensor(epi_seq)  # , dtype=torch.float32)
    except TypeError:
        tcr_seq = torch.Tensor(tcr_seq.astype('int'))  # , dtype=torch.float32)
        epi_seq = torch.Tensor(epi_seq.astype('int'))  # , dtype=torch.float32)

    return tcr_dist, epi_dist, tcr_seq, epi_seq, tcr_label, epi_label, sup_label


def ReconstructData(batch_data, method='max'):  # method: 'max', 'dict'
    detach_batch_data = batch_data.clone().detach().cpu().numpy()
    if method == 'max':
        detach_batch_data = detach_batch_data.reshape(
            detach_batch_data.shape[0],
            cfg.DataPara().MAXSEQLENGTH + 1,
            cfg.DataPara().PEPDIM
        )
    elif method == 'dict':
        detach_batch_data = detach_batch_data.reshape(
            detach_batch_data.shape[0],
            -1
        )
    return detach_batch_data


def RestorePepSequence(detached_batch_data, method='max'):   # method: 'max', 'dict'
    restored_sequences = []
    if method == 'max':
        for i in detached_batch_data:
            i = pd.DataFrame(i, columns=Blosum62_c)
            restored_sequences.append(''.join(list(i.idxmax(axis=1))))
    elif method == 'dict':
        _dict = cfg.par().proteindict
        pep_dict = {v: k for k, v in _dict.items()}
        for i in detached_batch_data:
            pep_seq = []
            for j in i:
                pep_seq.append(pep_dict[j])
            pep_seq = ''.join(list(pep_seq))
            restored_sequences.append(pep_seq)

    return restored_sequences


def TransferData2Onehot(numpy_array):
    numpy_array = np.load(numpy_array)
    onehot_label = np.eye(2)[numpy_array]
    return onehot_label


if __name__ == '__main__':
    folder = '/home/ji/Documents/Data/ProjectData/TcrPredict/b128_SupTrainBatch/'
    for f in os.listdir(folder):
        if (('label' in f) and ('seq' not in f)):
            ol = TransferData2Onehot(folder + f)
            # ol: One_hot lbael
            np.save(folder + 'ol_' + '_'.join(f.split('_')[1:]), ol)
