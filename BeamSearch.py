#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: BeamSearch.py
# Create time: 2023/12/23 16:58:46
# Author: Ji Hongchen

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import cfg
from Transformer import *
from StructrueTransformer import *
from SupTrain import *
from e2t_Generator import *

TEST_FOLDER = './demo_data/b128_UnsupTestBatch/'
NUM_TEST = int(len(os.listdir(TEST_FOLDER)) / 10)


corr_info = pd.read_csv(cfg.Cfg().projectdata +
                        'TrainData/node_TCR-PEP-MHC.csv', index_col=[0])
use_model = torch.load(cfg.Cfg().resultspace + '/model/CATCR_G.pt')
sturc_model = torch.load(cfg.Cfg().resultspace + 'model/RCMT.pt')
for param in use_model.parameters():
    param.require_grad = False

for param in sturc_model.parameters():
    param.require_grad = False


def build_match_seq_database(match_info, d_from='EPITOPE', d_to='CDR3_B'):
    match_dict = {}
    for i in match_info.index:
        try:
            match_dict[match_info[d_from][i]].append(match_info[d_to][i])
        except KeyError:
            match_dict[match_info[d_from][i]] = [match_info[d_to][i]]
    return match_dict


class Translator(nn.Module):
    def __init__(self, model):
        super(Translator, self).__init__()

        self.beam_size = cfg.TransformerPara().beam_size
        self.max_seq_len = cfg.TransformerPara().seq_len
        self.model = model

        self.init_seq = torch.Tensor(25 * np.ones((1, 1))).cuda().long()

        # Get the blank sequence for translation
        self.blank_seqs = torch.Tensor(
            np.ones((self.beam_size, cfg.DataPara().MAXSEQLENGTH + 1)) * 25
        ).cuda()

    def _decode_sequence(self, gen_seq):
        max_index = gen_seq.max(1).indices
        decoded_seq = ''
        for i in max_index:
            decoded_seq += self.protein_dict[i.item()]
        return decoded_seq

    def _model_decode(self, trg_seq, enc_output, src_d, tgt_d):
        dec_output, *_ = self.model.decoder(trg_seq, None,
                                            enc_output,
                                            None,
                                            src_d,
                                            tgt_d,
                                            return_attns=True
                                            )
        dec_output = dec_output.contiguous().view(-1, cfg.TransformerPara().d_model)
        dec_output = self.model.projection(dec_output)

        return dec_output

    def _get_init_state(self, src_seq, src_d, tgt_d):
        beam_size = self.beam_size

        encode_output, *_ = self.model.encoder(
            src_seq, src_d, src_mask=None
        )
        encode_output = encode_output.contiguous().view(
            src_seq.size(0), -1, cfg.TransformerPara().d_model
        )
        decode_output = self._model_decode(self.init_seq,
                                           encode_output,
                                           src_d, tgt_d)
        best_k_probs, best_k_ids = decode_output[-1, :].topk(beam_size)
        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_ids

        # The encode ouput has the dim 0 with gen_seq
        encode_output = encode_output.repeat(beam_size, 1, 1)

        return encode_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output,
                                    scores, step):
        assert len(scores.size()) == 1
        dec_output = dec_output.view(
            self.beam_size, -1, cfg.TransformerPara().d_word_vocab)
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(self.beam_size)
        scores = torch.log(best_k2_probs).view(
            self.beam_size, -1) + scores.view(self.beam_size, 1)

        scores, best_k_idx_in_k2 = scores.view(-1).topk(self.beam_size)
        best_k_r_idxs = best_k_idx_in_k2 // self.beam_size
        best_k_c_idxs = best_k_idx_in_k2 % self.beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs,
                                 best_k_c_idxs]
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def beam_translate(self, src_seq, src_d, tgt_d):
        with torch.no_grad():
            enc_output, gen_seq, scores = self._get_init_state(
                src_seq, src_d, tgt_d)

            for step in range(2, self.max_seq_len):
                decode_output = self._model_decode(
                    gen_seq[:, :step].long(),
                    enc_output,
                    src_d.repeat(self.beam_size, 1, 1, 1),
                    tgt_d.repeat(self.beam_size, 1, 1, 1)
                )
                gen_seq, scores = self._get_the_best_score_and_idx(
                    gen_seq, decode_output, scores, step
                )

        gen_seq = extract_batch_seq(gen_seq[:, 1:])
        return gen_seq


def beamtranslate(trans_model, s_model,  matchdict):
    test_loss = 0
    trans_model = trans_model.eval()
    trans_model.eval()

    pred_frame = pd.DataFrame(
        columns=['EPITOPE', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7'])
    with torch.no_grad():
        i = 0
        for t_batch in range(NUM_TEST):
            _, epi_t_d, tcr_t_s, epi_t_s, tcr_t_s_label, epi_t_s_label, _ = DataLoder(
                batch_folder=TEST_FOLDER,
                batch_num=t_batch
            )
            epi_t_d = epi_t_d.cuda()
            tcr_t_s = tcr_t_s[:, 1:].cuda().long()
            epi_t_s = epi_t_s[:, 1:].cuda().long()
            tcr_t_d = s_model(epi_t_s, epi_t_d)
            tcr_t_s_label = tcr_t_s_label[:, 1:].cuda(
            ).long().contiguous().view(-1)

            for k in range(tcr_t_d.size(0)):
                ori_seq = extract_sequence(epi_t_s[k])
                if ori_seq not in list(pred_frame['EPITOPE']):
                    preded_seqs = trans_model.beam_translate(
                        epi_t_s[k].unsqueeze(0),
                        epi_t_d[k].unsqueeze(0),
                        tcr_t_d[k].unsqueeze(0)
                    )
                    pred_list = [ori_seq]
                    pred_list.extend(preded_seqs)

                    pred_frame.loc[i] = pred_list
                    i += 1
    print(pred_frame)
    return pred_frame


if __name__ == "__main__":
    match_dict = build_match_seq_database(corr_info)
    translator = Translator(model=use_model)
    pred_frame = beamtranslate(translator, sturc_model, matchdict=match_dict)
