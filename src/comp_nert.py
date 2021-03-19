""" PGT: Accurate News Recommendation Coalescing Personal and Global Temporal Preferences

Code Authors:
    - Bonhun Koo, (darkgs@snu.ac.kr) Data Mining Lab. at Seoul National University.
    - U Kang, (ukang@snu.ac.kr) Associate Professor.

    File: src/comp_nert.py
    - Competitor function for PGT

"""

import time
import os
import random


import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from optparse import OptionParser

from adressa_dataset import AdressaRec
from ad_util import load_json
from ad_util import option2str

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)
parser.add_option('-s', action="store_true", dest='save_model', default=False)
parser.add_option('-z', action="store_true", dest='search_mode', default=False)

parser.add_option('-t', '--trendy_count', dest='trendy_count', type='int', default=5)
parser.add_option('-r', '--recency_count', dest='recency_count', type='int', default=3)

parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', default=3e-3)
parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=1024)
parser.add_option('-b', '--num_layers', dest='num_layers', type='int', default=1)

parser.add_option('-c', '--u2i_path', dest='u2i_path', type='string', default=None)


class NeRTModel(nn.Module):
    """
        competitor medel wapper class
    """
    def __init__(self, embed_size, cate_dim, args):
        """
            initializer function
            :embed_size: embedding dimention of news
            :cate_dim: embedding dimention of category
            :args: additional arguments
            :return: none
        """
        super(NeRTModel, self).__init__()

        hidden_size = args.hidden_size
        num_layers = args.num_layers

        self.rnn = nn.LSTM(embed_size, int(hidden_size/2), num_layers, batch_first=True, bidirectional=True)
        #self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.mlp = nn.Linear(hidden_size*2, embed_size)

        self._dropout = nn.Dropout(0.3)

        self._W_attn = nn.Parameter(torch.zeros([hidden_size*2, 1], dtype=torch.float32), requires_grad=True)
        self._b_attn = nn.Parameter(torch.zeros([1], dtype=torch.float32), requires_grad=True)

        self._mha = nn.MultiheadAttention(embed_size, 20 if (embed_size % 20) == 0 else 10)
        self._mlp_mha = nn.Linear(embed_size, hidden_size)
        self._mlp_act = nn.Tanh()

        nn.init.xavier_normal_(self._W_attn.data)

        self._rnn_hidden_size = hidden_size

    def forward(self, x1, x2, cate, seq_lens, _, attn_mode=False):
        """
            forwarding function of the model called by the pytorch
            :x1: input tensor
            :x2: input tensor for global temporal representation
            :seq_lens: list cataining the length of each seqeunce
            :attn_mode: flag of attention test
            :return: output tensor
        """
        batch_size = x1.size(0)
        max_seq_length = x1.size(1)
        embed_size = x1.size(2)

        outputs = torch.zeros([max_seq_length, batch_size, embed_size],
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        #
        cate = pack(cate, seq_lens, batch_first=True).data

        # x2: [batch_size, seq_len, num_of_temporal, embed_dim]
        x2 = pack(x2, seq_lens, batch_first=True).data

        mha_input = torch.transpose(x2, 0, 1)
        _, x2_score = self._mha(mha_input, mha_input, mha_input)
        x2_score = torch.softmax(torch.mean(x2_score, 2, keepdim=False), dim=1)
        x2_score = torch.unsqueeze(x2_score, dim=1)

        x2 = torch.squeeze(torch.bmm(x2_score, x2), dim=1)

        #x2 = torch.mean(x2, 0, keepdim=False)
        #x2 = torch.mean(x2.data, 1, keepdim=False)
        x2 = self._mlp_mha(x2)

        # sequence embedding
        x1 = pack(x1, seq_lens, batch_first=True)
        x1, _ = self.rnn(x1)

        sequence_lenths = x1.batch_sizes.cpu().numpy()
        cursor = 0
        prev_x1s = []

        if attn_mode:
            attn_score_save = torch.zeros([max_seq_length, batch_size, max_seq_length],
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        for step in range(sequence_lenths.shape[0]):
            sequence_lenth = sequence_lenths[step]

            x1_step = x1.data[cursor:cursor+sequence_lenth]
            x2_step = x2[cursor:cursor+sequence_lenth]

            prev_x1s.append(x1_step)

            prev_x1s = [prev_x1[:sequence_lenth] for prev_x1 in prev_x1s]

            prev_hs = torch.stack(prev_x1s, dim=1)
            attn_score = []
            for prev in range(prev_hs.size(1)):
                attn_input = torch.cat((prev_hs[:,prev,:], x2_step), dim=1)
                attn_score.append(torch.matmul(attn_input, self._W_attn) + self._b_attn)
            attn_score = torch.softmax(torch.stack(attn_score, dim=1), dim=1)

            if attn_mode:
                attn_score_save[step, :sequence_lenth, :attn_score.shape[1]] = torch.squeeze(attn_score, dim=2)

            x1_step = torch.squeeze(torch.bmm(torch.transpose(attn_score, 1, 2), prev_hs), dim=1)
            
            x_step = torch.cat((x1_step, x2_step), dim=1)
            x_step = self.mlp(x_step)

            outputs[step][:sequence_lenth] = x_step

            cursor += sequence_lenth

        if attn_mode:
            prev_cates = []
            attn_good_data = []

            attn_score_save = torch.transpose(attn_score_save, 0, 1)
            batch_total = 0
            cursor = 0
            for step in range(sequence_lenths.shape[0]):
                sequence_lenth = sequence_lenths[step]

                cate_step = cate[cursor:cursor+sequence_lenth]
                cursor += sequence_lenth

                prev_cates.append(cate_step)
                prev_cates = [prev[:sequence_lenth] for prev in prev_cates]

                batch_total += 1

                min_len = 14
                min_vari_cate_prev = 6
                min_vari_cate_pred = 4
                target_len = 10

#                min_len = 5
#                min_vari_cate_prev = 0
#                min_vari_cate_pred = 0
#                target_len = 2

                if step < min_len:
                    continue

                prev_cates_t = torch.stack(prev_cates, dim=0)
                prev_cates_t = torch.argmax(torch.transpose(prev_cates_t, 0, 1), dim=2)

                for batch in range(prev_cates_t.shape[0]):
                    #attn_score_c = attn_score_save[batch, step-1, :step].cpu().numpy().tolist()
                    prev_cates_c = prev_cates_t[batch, :].cpu().numpy().tolist()

                    prev_cate_set = set([])
                    pred_cate_set = set([])

                    for i, c in enumerate(prev_cates_c):
                        if i < target_len:
                            prev_cate_set.update([c])
                        else:
                            pred_cate_set.update([c])

                    if len(prev_cate_set) < min_vari_cate_prev or len(pred_cate_set) < min_vari_cate_pred:
                        continue

                    ok = 0
                    for c in pred_cate_set:
                        if c not in prev_cate_set:
                            continue
                        ok += 1

                    if ok < 4:
                        continue

                    attn_good_data.append(['==================================='])
                    attn_good_data.append(prev_cates_c[:target_len])
                    attn_good_data.append(prev_cates_c[target_len:])
                    for s in range(target_len, step+1):
                        attn_score_c = attn_score_save[batch, s-1, :s].cpu().numpy().tolist()
                        attn_good_data.append(attn_score_c[:target_len])


        outputs = torch.transpose(outputs, 0, 1)
        #outputs = self._dropout(outputs)

        if attn_mode:
            with open('attn_data_cate.txt', 'a') as f_att:
                attn_data_str = ''
                for data_line in attn_good_data:
                    data_line = [str(d) for d in data_line]
                    attn_data_str += ','.join(data_line) + '\n'
                f_att.write(attn_data_str)

        return outputs


def main():
    """
        main function
    """
    options, args = parser.parse_args()

    if (options.input == None) or (options.d2v_embed == None) or \
                       (options.u2v_path == None) or (options.ws_path == None):
        return

    torch_input_path = options.input
    embedding_dimension = int(options.d2v_embed)
    url2vec_path = '{}_{}'.format(options.u2v_path, embedding_dimension)
    url2info_path = options.u2i_path
    ws_path = options.ws_path
    search_mode = options.search_mode
    model_ws_path = '{}/model/{}'.format(ws_path, option2str(options))

    if not os.path.exists(ws_path):
        os.system('mkdir -p {}'.format(ws_path))

#    os.system('rm -rf {}'.format(model_ws_path))
    os.system('mkdir -p {}'.format(model_ws_path))

    # Save best result with param name
    param_search_path = ws_path + '/param_search'
    if not os.path.exists(param_search_path):
        os.system('mkdir -p {}'.format(param_search_path))
    param_search_file_path = '{}/{}'.format(param_search_path, option2str(options))

    if search_mode and os.path.exists(param_search_file_path):
        print('Param search mode already exist : {}'.format(param_search_file_path))
        return

    print('Loading url2vec : start')
    dict_url2vec = load_json(url2vec_path)
    print('Loading url2vec : end')

    print('Loading url2info : start')
    dict_url2info = load_json(url2info_path)
    print('Loading url2info : end')

    attn_analysis = True
    if attn_analysis:
        print('test mode')

    predictor = AdressaRec(NeRTModel, model_ws_path, torch_input_path,
            dict_url2vec, options, dict_url2info=dict_url2info)

    if attn_analysis:
        predictor.load_model()
        time_start = time.time()
        hit_5, _, mrr_20 = predictor.test_mrr_trendy(metric_count=20, candidate_count=20,
                attn_mode=True, length_mode=False)

        print(hit_5, mrr_20)

        return 

        hit_5, _, mrr_20 = predictor.test_mrr_trendy_history_test(metric_count=20, candidate_count=20)
        print('hitory_test :: hit_5 : {}, mrr_20 : {}'.format(hit_5, mrr_20))
        print('time tooks : {}'.format(time.time() - time_start))
        return

        for candi_count in [40, 60, 80, 100]:
            time_start = time.time()
            hit_5, _, mrr_20 = predictor.test_mrr_trendy(metric_count=20, candidate_count=candi_count)
            print('candi {} :: hit_5 : {}, mrr_20 : {}'.format(candi_count, hit_5, mrr_20))
            print('time tooks : {}'.format(time.time() - time_start))
        return

    best_hit_5, best_auc_20, best_mrr_20 = predictor.do_train()

    if search_mode:
        with open(param_search_file_path, 'w') as f_out:
            f_out.write(str(best_hit_5) + '\n')
            f_out.write(str(best_auc_20) + '\n')
            f_out.write(str(best_mrr_20) + '\n')


if __name__ == '__main__':
    main()

