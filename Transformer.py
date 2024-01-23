#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: UnsupTransformer.py
# Create time: 2023/11/10 21:40:43
# Author: Ji Hongchen

from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cfg
import os
import shutil
import random

CFG = cfg.Cfg()

# The sequence dict
if cfg.DataPara().datatype_seq == 'blosum':
    dictfile = CFG.publicdata + 'Blosum62.csv'
elif cfg.DataPara().datatype_seq == 'onehot':
    dictfile = CFG.publicdata + 'Onehot.csv'

pep_num_dict = {}
for i in cfg.par().proteindict:
    pep_num_dict[cfg.par().proteindict[i]] = i

seqdict = {}
seqdf = pd.read_csv(dictfile, index_col=[0])
for i in seqdf.index:
    seqdict[pep_num_dict[i]] = list(seqdf.loc[i])


def get_pad_mask(seq, pad_idx=24):
    return (seq != pad_idx).unsqueeze(-2)


def random_mask(batch_seq):
    pad_idx = -8
    pad_code = [0] * cfg.TransformerPara().d_model
    pad_num = random.sample(
        range(cfg.DataPara().PEPDIM - 1),
        int((cfg.DataPara().PEPDIM - 1) * cfg.TransformerPara().random_mask_frac))

    # batch_seq[:, pad_num, :] = pad_idx
    batch_seq[:, pad_num] = pad_idx
    return batch_seq


def get_subsequence_mask(seq):
    # sz_b, len_s, pepdim = seq.size()
    sz_b, len_s = seq.size()
    subsequence_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device),
        diagonal=1
    )).bool()

    return subsequence_mask


class PositionEmbedding(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(PositionEmbedding, self).__init__()
        self.Para = Para
        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(
                Para.seq_len, Para.d_model
            )
        )

    def _get_sinusoid_encoding_table(self, n_position, d_model):

        def get_position_angel_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        sinusoid_table = np.array(
            [get_position_angel_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, Para=cfg.TransformerPara()):
        super(ScaledDotProductAttention, self).__init__()
        self.Para = Para
        self.temperature = temperature
        self.dropout = nn.Dropout(Para.attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = Para.n_heads
        self.d_k = Para.d_k
        self.d_v = Para.d_v

        self.w_qs = nn.Linear(
            Para.d_model, self.n_heads * self.d_k, bias=False)
        self.w_ks = nn.Linear(
            Para.d_model, self.n_heads * self.d_k, bias=False)
        self.w_vs = nn.Linear(
            Para.d_model, self.n_heads * self.d_v, bias=False)
        self.fc = nn.Linear(Para.n_heads * self.d_v, Para.d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(Para.attn_dropout)
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_heads, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_heads, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_heads, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class CNN_part(nn.Module):
    def __init__(self, CNN_Para, features='more', softmax=False):
        super(CNN_part, self).__init__()
        self.features = features
        self.softmax = softmax
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, CNN_Para.CNN_layer_1_channel,
                      kernel_size=CNN_Para.CNN_layer_1_kernel_size),
            nn.BatchNorm2d(CNN_Para.CNN_layer_1_channel),
            nn.Dropout(CNN_Para.dropout_value),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh()
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(CNN_Para.CNN_layer_1_channel,
                      CNN_Para.CNN_layer_2_channel,
                      kernel_size=CNN_Para.CNN_layer_2_kernel_size),
            nn.BatchNorm2d(CNN_Para.CNN_layer_2_channel),
            nn.MaxPool2d(CNN_Para.MP_layer_2_kernel_size,
                         CNN_Para.MP_layer_2_stride),
            nn.Dropout(CNN_Para.dropout_value),
            # nn.Tanh(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(CNN_Para.CNN_layer_2_channel *
                      CNN_Para.linear_mut_mun * CNN_Para.linear_mut_mun,
                      cfg.TransformerPara().d_model * cfg.DataPara().MAXSEQLENGTH),
            nn.BatchNorm1d(cfg.TransformerPara().d_model *
                           cfg.DataPara().MAXSEQLENGTH),
            nn.Dropout(CNN_Para.dropout_value),
            # nn.Sigmoid(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class GRUpart(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(GRUpart, self).__init__()
        self.Para = Para
        self.hidden_dim = cfg.DataPara().MAXSEQLENGTH * Para.d_model
        self.position_embedding = PositionEmbedding()
        self.rnn = nn.GRU(
            Para.d_model + self.hidden_dim, self.hidden_dim,
            batch_first=True, num_layers=Para.gru_layer_num
        )
        self.generator = nn.Sequential(
            nn.Linear(Para.d_model + self.hidden_dim * 2, Para.d_model),
            # nn.BatchNorm1d(Para.d_model),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dropout = nn.Dropout(cfg.DataPara().dropout)

    def forward(self, trg_i, context, h_n):
        trg_i = trg_i.unsqueeze(1)
        trg_i = self.position_embedding(self.dropout(trg_i))

        input = torch.cat((trg_i, context), dim=2)
        output, h_n = self.rnn(input, h_n)

        input = input.squeeze(1)
        output = output.squeeze(1)
        input = torch.cat((input, output), dim=1)
        output = self.generator(input)

        return output, h_n


class PositionWiseFeedForward(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(Para.d_model, Para.d_ff)
        self.w_2 = nn.Linear(Para.d_ff, Para.d_model)
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)
        self.dropout = nn.Dropout(Para.attn_dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(self.w_1(x))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(EncoderLayer, self).__init__()
        self.Para = Para
        self.self_attn = MultiHeadAttention()
        self.pos_ffn = PositionWiseFeedForward()

    def forward(self, enc_input, self_atten_mask=None):
        enc_output, enc_self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=self_atten_mask
        )
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_self_attn


class DecoderLayer(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(DecoderLayer, self).__init__()
        self.Para = Para
        self.self_attn = MultiHeadAttention()
        self.enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionWiseFeedForward()

    def forward(self, dec_input, enc_output,
                self_attn_mask, dec_enc_attn_mask):
        dec_output, dec_self_attn = self.self_attn(
            dec_input, dec_input, dec_input, mask=self_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_self_attn, dec_enc_attn


class T_Encoder(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(T_Encoder, self).__init__()
        self.Para = Para
        self.src_word_emb = nn.Embedding(
            self.Para.d_word_vocab, self.Para.d_model)
        self.position_emb = PositionEmbedding()
        self.dropout = nn.Dropout(p=Para.attn_dropout)
        self.layers = nn.ModuleList([
            EncoderLayer() for _ in range(Para.n_layers)
        ])

        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)

    def forward(self, src_seq, src_d=None, src_mask=None, return_attns=True):
        enc_self_attn_list = []

        if self.Para.coding_method == 'embedding':
            enc_output = self.dropout(
                self.position_emb(self.src_word_emb(src_seq)))
        else:
            enc_output = self.dropout(
                self.position_emb(src_seq))

        enc_output = self.layer_norm(enc_output)

        for layer in self.layers:
            enc_output, enc_self_attn = layer(
                enc_output, self_atten_mask=src_mask)
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_self_attn_list
        else:
            return enc_output


class CNN_Encoder(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(CNN_Encoder, self).__init__()
        self.Para = Para
        self.src_word_emb = nn.Embedding(
            self.Para.d_word_vocab, self.Para.d_model)
        self.position_emb = PositionEmbedding()
        self.dropout = nn.Dropout(p=Para.attn_dropout)
        self.layers = nn.ModuleList([
            EncoderLayer() for _ in range(Para.n_layers)
        ])
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)

        self.cnn_k_3 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=3))
        self.cnn_k_5 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=5))
        self.cnn_fc = nn.Sequential(
            nn.Linear(Para.d_model * 2 * cfg.DataPara().MAXSEQLENGTH,
                      Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            nn.BatchNorm1d(Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, src_seq, src_d=None, src_mask=None, return_attns=True):
        enc_self_attn_list = []

        if self.Para.coding_method == 'embedding':
            enc_output = self.dropout(
                self.position_emb(self.src_word_emb(src_seq)))
        else:
            enc_output = self.dropout(
                self.position_emb(src_seq))

        enc_output = self.layer_norm(enc_output)

        for layer in self.layers:
            enc_output, enc_self_attn = layer(
                enc_output, self_atten_mask=src_mask)
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        # CNN part
        cnn_output_3 = self.cnn_k_3(src_d)
        cnn_output_5 = self.cnn_k_5(src_d)
        # cnn_output = torch.cat((cnn_output_1, cnn_output_3), dim=1)
        # cnn_output = torch.cat((cnn_output, cnn_output_7), dim=1)
        cnn_output = torch.cat((cnn_output_3, cnn_output_5), dim=1)
        cnn_output = cnn_output.contiguous().view(src_d.size(0), -1)
        cnn_output = self.cnn_fc(cnn_output)
        # CNN part

        if return_attns:
            return cnn_output, enc_self_attn_list
        else:
            return cnn_output


class TCNN_Encoder(nn.Module):
    def __init__(self, Para=cfg.TransformerPara(), embedding=True):
        super(TCNN_Encoder, self).__init__()
        self.Para = Para
        self.embedding = embedding
        self.src_word_emb = nn.Embedding(
            self.Para.d_word_vocab, self.Para.d_model)
        self.position_emb = PositionEmbedding()
        self.dropout = nn.Dropout(p=Para.attn_dropout)
        self.layers = nn.ModuleList([
            EncoderLayer() for _ in range(Para.n_layers)
        ])
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)

        self.cnn_k_3 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=3))
        self.cnn_k_5 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=5))

        self.total_fc = nn.Sequential(
            nn.Linear(Para.d_model * 3 * cfg.DataPara().MAXSEQLENGTH,
                      Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            # Para.output_dim),
            nn.BatchNorm1d(Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, src_seq, src_d=None, src_mask=None, return_attns=True):
        enc_self_attn_list = []

        # if self.embedding == True:
        enc_output = self.dropout(
            self.position_emb(self.src_word_emb(src_seq)))
        # else:
        #    enc_output = self.dropout(
        #        self.position_emb(src_seq))

        for layer in self.layers:
            enc_output, enc_self_attn = layer(
                enc_output, self_atten_mask=src_mask)
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output).view(src_seq.size(0), -1)
        # CNN part
        cnn_output_3 = self.cnn_k_3(src_d)
        cnn_output_5 = self.cnn_k_5(src_d)
        # cnn_output = torch.cat((cnn_output_1, cnn_output_3), dim=1)
        # cnn_output = torch.cat((cnn_output, cnn_output_7), dim=1)
        cnn_output = torch.cat((cnn_output_3, cnn_output_5), dim=1)
        cnn_output = cnn_output.contiguous().view(src_d.size(0), -1)
        output = torch.cat((cnn_output, enc_output), dim=1)
        output = self.total_fc(output)

        # if Unsupervised:
        # cnn_output = cnn_output.view(src_seq.size(0), -1, self.Para.d_model)

        # CNN part

        if return_attns:
            return output, enc_self_attn_list
        else:
            return output


class TCNN_cross_Encoder(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(TCNN_cross_Encoder, self).__init__()
        self.Para = Para
        self.src_word_emb = nn.Embedding(
            self.Para.d_word_vocab, self.Para.d_model)
        self.position_emb = PositionEmbedding()
        self.dropout = nn.Dropout(p=Para.attn_dropout)
        self.layers = nn.ModuleList([
            EncoderLayer() for _ in range(Para.n_layers)
        ])
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)
        self.batch_norm = nn.BatchNorm1d(Para.d_model)

        self.cnn_k_3 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=3))
        self.cnn_k_5 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=5))
        self.inner_linear = nn.Linear(Para.d_model * 2, Para.d_model)
        self.cnn_fc = nn.Sequential(
            nn.Linear(Para.d_model * 2, Para.d_model * 1),
            nn.BatchNorm1d(Para.d_model * 1),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.total_fc = nn.Sequential(
            nn.Linear(Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH,
                      Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            # Para.output_dim),
            nn.BatchNorm1d(Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, src_seq, src_d=None, src_mask=None, return_attns=True):
        enc_self_attn_list = []

        if self.Para.coding_method == 'embedding':
            enc_output = self.dropout(
                self.position_emb(self.src_word_emb(src_seq)))
        else:
            enc_output = self.dropout(
                self.position_emb(src_seq))

        enc_output = self.layer_norm(enc_output)

        cnn_output_3 = self.cnn_k_3(src_d)
        cnn_output_5 = self.cnn_k_5(src_d)
        # cnn_output = torch.cat((cnn_output_1, cnn_output_3), dim=1)
        # cnn_output = torch.cat((cnn_output, cnn_output_7), dim=1)
        cnn_output = torch.cat((cnn_output_3, cnn_output_5), dim=1)
        cnn_output = cnn_output.view(src_d.size(0), -1, self.Para.d_model * 2)
        cnn_output = self.cnn_fc(cnn_output)

        # CNN part
        for layer in self.layers:
            enc_output = torch.cat((enc_output, cnn_output), dim=2)
            enc_output = self.inner_linear(enc_output)
            enc_output, enc_self_attn = layer(
                enc_output, self_atten_mask=src_mask)
            cnn_output = torch.matmul(
                cnn_output, torch.sum(enc_self_attn, dim=1))
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        # enc_output = torch.cat((enc_output, cnn_output), dim=1)
        enc_output = enc_output.view(cfg.DataPara().BATCHSIZE, -1)
        # enc_output = self.total_fc(enc_output)
        # print(enc_output.shape)

        if return_attns:
            return enc_output, enc_self_attn_list
        else:
            return enc_output


class AttentionCNN_Encoder(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(AttentionCNN_Encoder, self).__init__()
        self.Para = Para
        self.src_word_emb = nn.Embedding(
            self.Para.d_word_vocab, self.Para.d_model)
        self.position_emb = PositionEmbedding()
        self.dropout = nn.Dropout(p=Para.attn_dropout)
        self.layers = nn.ModuleList([
            EncoderLayer() for _ in range(Para.n_layers)
        ])
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)

        self.cnn_k_1 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=1))
        self.cnn_k_3 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=3))
        self.cnn_k_5 = CNN_part(CNN_Para=cfg.CNNPara(N_kernal=5))

        self.cnn_fc = nn.Sequential(
            nn.Linear(Para.d_model * 3 * cfg.DataPara().MAXSEQLENGTH,
                      Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            # Para.output_dim),
            nn.BatchNorm1d(Para.d_model * 1 * cfg.DataPara().MAXSEQLENGTH),
            # nn.BatchNorm1d(Para.output_dim),
            nn.Dropout(cfg.DataPara().dropout),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(Para.d_model, Para.d_model),
            nn.BatchNorm1d(Para.d_model),
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Linear(Para.d_model, Para.d_model),
            # nn.BatchNorm1d(Para.d_model),
            # nn.Dropout(),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(Para.d_model, Para.d_model),
            nn.BatchNorm1d(Para.d_model),
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, src_seq, src_d=None, src_mask=None, return_attns=True):
        enc_self_attn_list = []

        if self.Para.coding_method == 'embedding':
            enc_output = self.dropout(
                self.position_emb(self.src_word_emb(src_seq)))
        else:
            enc_output = self.dropout(
                self.position_emb(src_seq))

        enc_output = self.layer_norm(enc_output)

        # CNN part
        for layer in self.layers:
            enc_output, enc_self_attn = layer(
                enc_output, self_atten_mask=src_mask)
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        src_d = src_d.squeeze(1)
        src_d = torch.matmul(src_d, enc_self_attn.sum(1)).unsqueeze(1)

        # cnn_output_1 = self.cnn_k_1(src_d)
        cnn_output_3 = self.cnn_k_3(src_d)
        cnn_output_5 = self.cnn_k_5(src_d)
        cnn_output = torch.cat((cnn_output_3, cnn_output_5), dim=1)
        # cnn_output = torch.cat((cnn_output, cnn_output_5), dim=1)
        cnn_output = cnn_output.view(src_d.size(0), -1)
        cnn_output = torch.cat((cnn_output, enc_output.contiguous().view(
            enc_output.size(0), -1
        )), dim=1)
        cnn_output = self.cnn_fc(cnn_output)
        cnn_output = cnn_output.contiguous().view(
            cnn_output.size(0), -1, cfg.TransformerPara().d_model
        )

        cnn_output = self.fc(cnn_output)

        # enc_output = enc_output.view(src_d.size(0), -1)
        # cnn_output = self.cnn_fc(cnn_output)
        # cnn_output = torch.cat((cnn_output, enc_output), dim=1)
        # cnn_output = cnn_output.view(cfg.DataPara().BATCHSIZE, -1)

        # if Unsupervised:
        # cnn_output = cnn_output.view(src_seq.size(0), -1, self.Para.d_model)

        if return_attns:
            return cnn_output, enc_self_attn_list
        else:
            return cnn_output


class T_Decoder(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(T_Decoder, self).__init__()
        self.Para = Para
        self.tgt_word_emb = nn.Embedding(Para.d_word_vocab, Para.d_model)
        self.position_emb = PositionEmbedding()
        self.dropout = nn.Dropout(Para.attn_dropout)
        self.layers = nn.ModuleList([DecoderLayer()
                                    for _ in range(Para.n_layers)])
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask,
                src_d=None, tgt_d=None, return_attns=True):
        dec_self_attn_list, dec_enc_attn_list = [], []
        if self.Para.coding_method == 'embedding':
            dec_output = self.dropout(
                self.position_emb(self.tgt_word_emb(tgt_seq)))
        else:
            dec_output = self.dropout(self.position_emb(tgt_seq))

        dec_output = self.layer_norm(dec_output)

        for layer in self.layers:
            dec_output, dec_self_attn, dec_enc_attn = layer(
                dec_output, enc_output,
                self_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask
            )
            dec_self_attn_list += [dec_self_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_self_attn_list, dec_enc_attn_list
        else:
            return dec_output


class STRUC_Decoder(nn.Module):
    def __init__(self, Para=cfg.TransformerPara()):
        super(STRUC_Decoder, self).__init__()
        self.Para = Para
        self.tgt_word_emb = nn.Embedding(Para.d_word_vocab, Para.d_model)
        self.position_emb = PositionEmbedding()
        self.dropout = nn.Dropout(Para.attn_dropout)
        self.layers = nn.ModuleList([DecoderLayer()
                                    for _ in range(Para.n_layers)])
        self.layer_norm = nn.LayerNorm(Para.d_model, eps=1e-6)
        self.sigmoid = nn.Sigmoid()
        self.former_fc = nn.Linear(Para.d_model * 2, Para.d_model)

    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask,
                src_d=None, tgt_d=None, return_attns=True):
        src_d = src_d.squeeze(1)
        tgt_d = tgt_d.squeeze(1)
        dec_self_attn_list, dec_enc_attn_list = [], []
        if self.Para.coding_method == 'embedding':
            dec_output = self.dropout(
                self.position_emb(self.tgt_word_emb(tgt_seq)))
        else:
            dec_output = self.dropout(self.position_emb(tgt_seq))

        dec_output = self.layer_norm(dec_output)

        # enc_output = torch.cat((enc_output, tgt_d), dim = 2)
        enc_output = self.former_fc(torch.cat((enc_output, tgt_d), dim=2))

        for layer in self.layers:
            dec_output, dec_self_attn, dec_enc_attn = layer(
                dec_output, enc_output,
                self_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask
            )
            dec_self_attn_list += [dec_self_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_self_attn_list, dec_enc_attn_list
        else:
            return dec_output


if __name__ == "__main__":
    seq = np.array([[25, 1, 3, 4, 5, 6, 7, 24], [25, 1, 3, 5, 6, 7, 5, 24]])
    seq = torch.Tensor(seq).long().cuda()
    src_input = seq[:, 1:]
    tgt_input = seq[:, :-1]
    tgt_output = seq[:, 1:]
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()

    enc_output, enc_attn = encoder(seq)
    dec_output, dec_attn, d_e_attn = decoder(tgt_input, None, enc_output, None)
    print(dec_output.shape)
