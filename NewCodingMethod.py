#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: NewCodingMethod.py
# Create time: 2023/12/07 19:26:34
# Author: Ji Hongchen

import pandas as pd
import numpy as np
import cfg
import re

CFG = cfg.Cfg()
MAX_NODE = 5

merge_cols = ['CDR3_B', 'EPITOPE']
corr_file = cfg.Cfg().projectdata + 'TrainData/ori_TCR-PEP-MHC.csv'
corr_data = pd.read_csv(corr_file, index_col=[0])[merge_cols]

"""
seq_list = list(set(corr_data['CDR3_B']))
seq_list.extend(list(set(corr_data['EPITOPE'])))

countframe = pd.DataFrame(columns=['count'])

for seq in range(len(seq_list)):
    for i in range(2, MAX_NODE + 1):
        for k in range(0, len(seq_list[seq]) - i):
            node_seq = seq_list[seq][k: k + i]
            try:
                countframe['count'].loc[node_seq] += 1
            except KeyError:
                countframe.loc[node_seq] = 1
    if seq % 1000 == 0:
        print(seq)
        print(countframe.sort_values(by=['count'], ascending=False))
    
countframe.to_csv(cfg.Cfg().projectdata + 'TrainData/node_TCR-PEP-MHC.csv')
"""

node_index = pd.read_csv(cfg.Cfg().projectdata + 'TrainData/node_index.csv',
                         names=['seq', 'count', 'ind'])
node_index['seqlen'] = node_index['seq'].str.len()
node_index.sort_values(by=['seqlen', 'count'], inplace=True, ascending=False)
print(node_index)
node_index['liststr'] = 'a'
node_index.set_index('seq', inplace=True, drop=True)
for i in node_index.index:
    node_index['liststr'].loc[i] = list(i)


def GenerateNewCodeSeq(seq_list, node_index):
    neo_seq_list = []
    for seq in seq_list:
        if seq != '':
            for k in node_index.index:
                if k in seq:
                    seq = seq.replace(k, str(node_index['ind'].loc[k]) + ',')
            seq = seq.split(',')[: -1]
            seq.extend([24] * (cfg.DataPara().MAXSEQLENGTH - len(seq)))
            neo_seq_list.append(seq)
        elif seq == '':
            neo_seq_list.append([24] * 25)
    return np.array(neo_seq_list).astype('int')


if __name__ == '__main__':
    neo_seq_list = GenerateNewCodeSeq(['AAASKKKFF', ''], node_index)
    print(len(neo_seq_list[0]))
