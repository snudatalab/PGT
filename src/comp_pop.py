""" PGT: Accurate News Recommendation Coalescing Personal and Global Temporal Preferences

Code Authors:
    - Bonhun Koo, (darkgs@snu.ac.kr) Data Mining Lab. at Seoul National University.
    - U Kang, (ukang@snu.ac.kr) Associate Professor.

    File: src/comp_pop.py
    - Competitor function for POP

"""

import time
import os

from optparse import OptionParser

import torch
import torch.nn as nn

from adressa_dataset import AdressaRec

from ad_util import load_json

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)

parser.add_option('-t', '--trendy_count', dest='trendy_count', type='int', default=1)
parser.add_option('-r', '--recency_count', dest='recency_count', type='int', default=1)

parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', default=3e-3)

class SingleLSTMModel(nn.Module):
    """
        dummy class never be used
    """
	def __init__(self, embed_size, cate_dim, args):
		super(SingleLSTMModel, self).__init__()

		hidden_size = 768
		num_layers = 1

		self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, x, _, __, seq_lens):
		batch_size = x.size(0)
		embed_size = x.size(2)

		x = pack(x, seq_lens, batch_first=True)
		outputs, _ = self.rnn(x)
		outputs, _ = unpack(outputs, batch_first=True)
		outputs = self.linear(outputs)

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
	url2vec_path = options.u2v_path
	ws_path = options.ws_path

	os.system('rm -rf {}'.format(ws_path))
	os.system('mkdir -p {}'.format(ws_path))

	print('Loading url2vec : start')
	dict_url2vec = load_json(url2vec_path)
	print('Loading url2vec : end')

	predictor = AdressaRec(SingleLSTMModel, ws_path, torch_input_path, dict_url2vec, options)

	time_start = time.time()
	hit_5, mrr_20 = predictor.pop_history_test(metric_count=20, candidate_count=20)
	print('history test :: hit_5 : mrr_20 : {}'.format(hit_5, mrr_20))
	print('time tooks : {}'.format(time.time() - time_start))
	return

	hit_5, mrr_20 = predictor.pop(metric_count=20, candidate_count=20)
	print('candi {} :: hit_5 : {}, mrr_20 : {}'.format(20, hit_5, mrr_20))
	print('time tooks : {}'.format(time.time() - time_start))

	for candi_count in [40, 60, 80, 100]:
		time_start = time.time()
		hit_5, mrr_20 = predictor.pop(metric_count=20, candidate_count=candi_count)
		print('candi {} :: hit_5 : {}, mrr_20 : {}'.format(candi_count, hit_5, mrr_20))
		print('time tooks : {}'.format(time.time() - time_start))


if __name__ == '__main__':
	main()
