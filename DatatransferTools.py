#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Ji Hongchen
# @Email : jhca.xyt@163.com
# @Last version : 2023-06-19 22:48
# @Filename : vdjdatatransfer.py

import pandas as pd
import numpy as np
import os
import cfg
import shutil
from pyfaidx import Fasta

CFG = cfg.Cfg()
PAR = cfg.par()
REF_SEQ = Fasta(CFG.publicdata + 'HG38Ref/hg38_sequence.fa')
GENE_INFO = pd.read_csv(CFG.publicdata + 'GenePosition.csv',
                        index_col=[0])
GENE_INFO = GENE_INFO[['chromosome', 'from', 'to']]
# GENE_INFO.columns = ['chromosome', 'from', 'to']

pardict_blosum = pd.read_csv(CFG.publicdata + 'Blosum62.csv',
                             index_col=[0])
pardict_onehot = pd.read_csv(CFG.publicdata + 'Onehot.csv',
                             index_col=[0])


def ProteinSequence2Code(proteinseq, pardict_b=pardict_blosum, pardict_o=pardict_onehot):
    proteinseq.extend(
        list('*' * (cfg.DataPara().MAXSEQLENGTH - len(proteinseq)))
    )
    # proteinarray = np.zeros((len(proteinseq), 20))
    proteinarray_b = np.zeros((cfg.DataPara().MAXSEQLENGTH + 1, 25))
    proteinarray_o = np.zeros((cfg.DataPara().MAXSEQLENGTH + 1, 25))
    for i in range(len(list(proteinseq))):
        proteinarray_b[i + 1] = list(
            pardict_b.loc[proteinseq[i]]
        )
        proteinarray_o[i + 1] = list(
            pardict_o.loc[proteinseq[i]]
        )
    return (proteinarray_b, proteinarray_o)


def GetGenePosition(genename, onehot=False):
    geneseq = REF_SEQ[GENE_INFO['chromosome'][genename]
                      ][GENE_INFO['from'][genename]: GENE_INFO['to'][genename]].seq
    if onehot == True:
        basedict = PAR.basedict
        genearray = np.zeros((len(geneseq), 4))
        for i in range(len(list(geneseq))):
            genearray[i][basedict[geneseq[i]]] = 1
        geneseq = genearray
    return geneseq


def GenerateFastaFile(sequence, savename):
    nseq = []
    for i in range(int(len(sequence) / 80) + 1):
        nseq.append(sequence[80 * i: 80 * (i + 1)])
    output_fa = open(savename + '.fa', 'w')
    output_fa.write('>' + savename.split('/')[-1] + '\n')
    for nl in nseq:
        output_fa.write(nl + '\n')
    output_fa.close()


def PDB2PositionFrame(pdbfile, targetsequence):
    if '.pdb' in pdbfile:
        dt = pd.read_csv(pdbfile, names=['temp'], skiprows=2)
        dt = dt[dt['temp'].str.contains('ATOM')]
        for i in range(40):
            dt['temp'] = dt['temp'].str.replace('  ', ' ')
        dt = dt['temp'].str.split(' ', expand=True)
        del dt[12]
        dt.columns = ['cate', 'range', 'perssad', 'AA', 'pos',
                      'pos_rank', 'X', 'Y', 'Z', 'occupancy',
                      'tempFactor', 'atom']
        # print(dt[dt['atom'] == 'S'])
        dt = dt[dt['perssad'] == 'CA']
        for p in PAR.AAdict:
            dt.loc[dt['AA'] == p, 'AA'] = PAR.AAdict[p]
        dt = dt[['AA', 'pos', 'X', 'Y', 'Z']]
    elif '.csv' in pdbfile:
        dt = pd.read_csv(pdbfile, index_col=[0])

    AAseq = ''.join(list(dt['AA']))
    i = AAseq.index(targetsequence)
    dt = dt.iloc[i: (i + len(targetsequence))]

    return dt


def RepDistanceAndSequenceMatrix(extracted_pdb_data,
                                 max_length=cfg.DataPara().MAXSEQLENGTH,
                                 seq_code_method='OneHot'):     # or 'Blosum62'
    """
         AA pos        X        Y       Z
    1651  N   C  -25.148  -24.039  34.706
    1659  L   C  -21.473  -22.943  34.693
    1667  V   C  -20.408  -19.301  34.658
    1674  P   C  -19.486  -18.482  31.031
    """
    extracted_pdb_data.reset_index(inplace=True)
    extracted_pdb_data[['X', 'Y', 'Z']] = \
        extracted_pdb_data[['X', 'Y', 'Z']].astype('float')
    index_list = list(extracted_pdb_data.index)
    dist_array = np.zeros((max_length, max_length))
    for i in index_list:
        for j in index_list:
            dist_array[i][j] = np.linalg.norm(
                np.array(list(extracted_pdb_data[['X', 'Y', 'Z']].loc[j])) -
                np.array(list(extracted_pdb_data[['X', 'Y', 'Z']].loc[i]))
            )

    protein_array_b, protein_array_o = ProteinSequence2Code(
        list(extracted_pdb_data['AA']),
        # seq_code_method
    )
    return (dist_array, protein_array_b, protein_array_o)


if __name__ == '__main__':
    CDR_PEP_data = pd.read_csv(
        cfg.Cfg().projectdata + 'TrainData/node_TCR-PEP-MHC.csv',
        index_col=[0]
    )
    TRUE_PDB_data = CFG.extradata + \
        'TCR-PEP-MHC_data/ATLAS_data/structures/true_pdb/'
    total_list = list(CDR_PEP_data['CDR3_B'])
    total_list.extend(CDR_PEP_data['EPITOPE'])
    total_list = list(set(total_list))

    i = 0
    matchframe = pd.DataFrame(columns=['PRED', 'TRUE'])

    for f in os.listdir(TRUE_PDB_data):
        print(f)
        if f not in list(pd.read_csv(
            cfg.Cfg().resultspace + 'Manu_results/match_pred_struc.csv')['TRUE']
        ):
            for pep in total_list:
                try:
                    dt = PDB2PositionFrame(
                        CFG.extradata + 'TCR-PEP-MHC_data/ATLAS_data/structures/true_pdb/' + f,
                        targetsequence=pep
                    )
                    print(pep)
                    # dt.to_csv(cfg.Cfg().resultspace + 'Manu_results/test.pdb')
                    shutil.copy(
                        CFG.extradata + 'TCR-PEP-MHC_data/ATLAS_data/structures/true_pdb/' + f,
                        CFG.resultspace + 'Manu_results/' + f
                    )
                    shutil.copy(
                        CFG.projectdata + 'pre_PdbStructure/' + pep + '.pdb',
                        CFG.resultspace + 'Manu_results/' + pep + '.pdb'
                    )
                    matchframe.loc[i] = [pep, f]
                    i += 1
                    matchframe.to_csv(CFG.resultspace +
                                      'Manu_results/match_pred_struc.csv')
                except ValueError:
                    pass
                except KeyError:
                    pass
                # d_t, s_t = RepDistanceAndSequenceMatrix(dt,
                #                                         seq_code_method='Blosum62')
                # print(d_t)
                # print(s_t)
