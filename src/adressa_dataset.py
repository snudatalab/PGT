""" PGT: Accurate News Recommendation Coalescing Personal and Global Temporal Preferences

Authors:
    - Bonhun Koo, (darkgs@snu.ac.kr) Data Mining Lab. at Seoul National University.
    - U Kang, (ukang@snu.ac.kr) Associate Professor.

    File: src/adressa_dataset.py
    - Generate torch.utils.data.dataset be fed to RNN models

"""

import os, sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from torch.utils.data.dataset import Dataset  # For custom datasets

import numpy as np

from ad_util import load_json
from ad_util import weights_init


class AdressaDataset(Dataset):
    """
        Dataset warpper class for iteration
    """
    def __init__(self, dict_dataset):
        self._dict_dataset = dict_dataset
        self._data_len = len(self._dict_dataset)

    def __getitem__(self, index):
        return self._dict_dataset[index]

    def __len__(self):
        return self._data_len


class RecInputCategoryMixin(object):
    """
        Mixin class to handle category informations
    """
    def load_category(self, dict_url2vec={}, dict_url2info={}):
        """
            load category information
            :dict_url2vec: dictionary to translate from url to vectorizated representation
            :return: none
        """
        if (not dict_url2vec) or (not dict_url2info):
            self._cate_dim = 0
            self._dict_url2cate = {}
            return

        # cate:string => idx:int
        dict_cate2idx = self.get_cate2idx(dict_url2info)

        # _cate_dim - dimension of an one-hot vector representing category
        # _dict_rul2cate - url:string => catevec:one hot vector
        self._cate_dim, self._dict_url2cate = \
                self.get_url2cate(dict_url2vec, dict_url2info, dict_cate2idx)

    def url2cate(self, url):
        """
            translate from url to one-hot vector of category
            :url: url of a news article
            :return: one-hot vector of category
        """
        if (self._cate_dim == 0) or (not self._dict_url2cate):
            return [0.0]

        return self._dict_url2cate[url]
        #return self._dict_url2cate[self._dict_rec_input['idx2url'][str(idx)]]

    def get_cate2idx(self, dict_url2info):
        """
            translate from one-hot vector of category to index
            :dict_url2info: dictionary to translate from url to information cotaining category
            :return: dictionary to translate from category to index
        """
        categories = set([])

        for url, dict_info in dict_url2info.items():
            category = dict_info.get('category0', '')

            if not category:
                continue

            categories.update([category])

        categories = sorted(list(categories))
        print('categories', categories)

        return {cate:idx for idx, cate in enumerate(categories)}

    def get_url2cate(self, dict_url2vec, dict_url2info, dict_cate2idx):
        """
            generate dictionary of translating from url to one-hot vector of its category
            :dict_url2vec: dictionary to translate from url to its relavent vector
            :dict_url2info: dictionary to translate from url to its additional informations
            :dict_cate2idx: dictionary to translate from category to its index
            :return: dictionary of translating from url to one-hot vector of its category
        """
        cate_dim = len(dict_cate2idx)

        # pre-allocated one-hot vectors
        cate_one_hots = []
        for i in range(cate_dim):
            cate_one_hot = [0.0] * cate_dim
            cate_one_hot[i] = 1.0

            cate_one_hots.append(cate_one_hot)

        cate_one_hots.append([0.0] * cate_dim)

        dict_url2cate = {}
        for url, _ in dict_url2vec.items():
            category = dict_url2info.get(url, {}).get("category0", None)

            if category == None:
                one_hot_idx = cate_dim
            else:
                one_hot_idx = dict_cate2idx[category]

            dict_url2cate[url] = cate_one_hots[one_hot_idx]

        return cate_dim, dict_url2cate


class RecInputWordEmbed(object):
    """
        Dataset warpper class for word embeddings
    """
    def __init__(self, *args, **kwargs):
        """
            init function of class
        """
        super().__init__(*args, **kwargs)

        dict_glove = kwargs.get('glove', {})

        if not dict_glove:
            return

        ##
        self._dict_glove = dict_glove

        word_dim = 0
        if self._dict_glove:
            word_dim = len(next(iter(self._dict_glove['word_idx2vec'].values())))
        self._word_pad_vec = [ np.array([0.0] * word_dim) ]

    def idx2words_vec(self, idx):
        """
            get words embeddings of news from index of the news
            :idx: index of a news article
            :return: vectorized representations of words contained in the news
        """
        url = self._dict_rnn_input['idx2url'][str(idx)]
        if url == 'url_pad':
            return [self._word_pad_vec]
        word_indices = self._dict_glove['url2word_idx'][url]
        return [ self._dict_glove['word_idx2vec'][word_idx] for word_idx in word_indices ]


class RecInputMixin(object):
    """
        Mixin class for torch rnn input functions
    """
    def get_article_size(self):
        """
            get total number of news
            :return: total number of news
        """
        return len(self._dict_rec_input['idx2url'])

    def get_user_size(self):
        """
            get total number of users
            :return: total number of users
        """
        return self._dict_rec_input['user_size']

    def load_rec_input(self, dict_url2vec={}, dict_rec_input={}, options={}):
        """
            load basis of rnn input
            :dict_url2vec: dictionary to translate from url to its relavent vector
            :dict_rec_input: source of dataset contents
            :options: dictionary containing additional arguments
            :return: none
        """
        assert(dict_rec_input and dict_url2vec)

        self._dict_url2vec = dict_url2vec
        self._dict_rec_input = dict_rec_input

        self._trendy_count = options.trendy_count
        self._recency_count = options.recency_count

    def get_pad_idx(self):
        """
            get index of dummy news article
            :return: index of dummy news article
        """
        return self._dict_rec_input['pad_idx']

    def idx2url(self, idx):
        """
            get url of news from its index
            :idx: index of news
            :return: url of the news
        """
        return self._dict_rec_input['idx2url'][str(idx)]

    def idx2vec(self, idx):
        """
            get vectorized representations of news from its index
            :idx: index of news
            :return: vectorized representations of news from its index
        """
        return self._dict_url2vec[self.idx2url(idx)]

    def get_trendy(self, cur_time=-1, padding=0):
        """
            get popular news article at specific time
            :cur_time: a timestamp
            :padding: padding index of news
            :return: list of popular news indices
        """
        trendy_list = self._dict_rec_input['trendy_idx'].get(str(cur_time), None)
        recency_list = self._dict_rec_input['recency_idx'].get(str(cur_time), None)

        x2_list = []

        if trendy_list == None:
            trendy_list = [[padding, 0]] * self._trendy_count

        if recency_list == None:
            recency_list = [[padding, 0]] * self._recency_count

        assert(len(trendy_list) >= self._trendy_count)
        assert(len(recency_list) >= self._recency_count)

        return trendy_list[:self._trendy_count] + recency_list[:self._recency_count]

    def get_candidates(self, cur_time=-1, padding=0):
        """
            get candidate articles at specific time
            :cur_time: a time
            :padding: padding index of news
            :return: list of candidate news indices
        """
        candidates_max = 100

        trendy_list = self._dict_rec_input['trendy_idx'].get(str(cur_time), None)

        if trendy_list == None:
            trendy_list = [[padding, 0]] * candidates_max

        if len(trendy_list) < candidates_max:
            trendy_list += [[padding, 0]] * (candidates_max - len(trendy_list))
        elif len(trendy_list) > candidates_max:
            trendy_list = trendy_list[:candidates_max]

        assert(len(trendy_list) == candidates_max)

        return trendy_list

    def get_candidates_(self, start_time=-1, end_time=-1, idx_count=0):
        """
            support function of get_candidates
            :start_time: start of time window
            :end_time: end of time window
            :idx_count: number of maximum candidates
            :return: list of candidate news indices
        """
        if (start_time < 0) or (end_time < 0) or (idx_count <= 0):
            return []

        #    entry of : dict_rec_input['time_idx']
        #    (timestamp) :
        #    {
        #        prev_time: (timestamp)
        #        next_time: (timestamp)
        #        'indices': { idx:count, ... }
        #    }

        # swap if needed
        if start_time > end_time:
            tmp_time = start_time
            start_time = end_time
            end_time = tmp_time

        cur_time = start_time

        dict_merged = {}
        while(cur_time < end_time):
            cur_time = self._dict_rec_input['time_idx'][str(cur_time)]['next_time']
            for idx, count in self._dict_rec_input['time_idx'][str(cur_time)]['indices'].items():
                dict_merged[idx] = dict_merged.get(idx, 0) + count

        steps = 0
        time_from_start = start_time
        time_from_end = end_time
        while(len(dict_merged.keys()) < idx_count):
            if time_from_start == None and time_from_end == None:
                break

            if steps % 3 == 0:
                if time_from_end == None:
                    steps += 1
                    continue
                cur_time = self._dict_rec_input['time_idx'][str(time_from_end)]['next_time']
                time_from_end = cur_time
            else:
                if time_from_start == None:
                    steps += 1
                    continue
                cur_time = self._dict_rec_input['time_idx'][str(time_from_start)]['prev_time']
                time_from_start = cur_time

            if cur_time == None:
                continue

            for idx, count in self._dict_rec_input['time_idx'][str(cur_time)]['indices'].items():
                dict_merged[idx] = dict_merged.get(idx, 0) + count

        ret_sorted = sorted(dict_merged.items(), key=lambda x:x[1], reverse=True)
        if len(ret_sorted) > idx_count:
            ret_sorted = ret_sorted[:idx_count]
        return list(map(lambda x: int(x[0]), ret_sorted))

    def get_mrr_recency_candidates(self, cur_time=-1, padding=0):
        """
            get fresh articles at specific time
            :cur_time: a time
            :padding: padding index of news
            :return: list of fresh news indices
        """
        candidates_max = 100

        recency_candidates = []

        trendy_list = self._dict_rec_input['trendy_idx'].get(str(cur_time), None)
        recency_list = self._dict_rec_input['recency_idx'].get(str(cur_time), None)

        if trendy_list == None:
            trendy_list = []

        if recency_list == None:
            recency_list = []

        recency_articles = [r for r, r_c in recency_list]
        remains = []
        for t, t_c in trendy_list:
            if t in recency_articles:
                recency_candidates.append([t, t_c])
            else:
                remains.append(t)

        for r in remains:
            recency_candidates.append([r,0])

        if len(recency_candidates) < candidates_max:
            recency_candidates += [[padding, 0]] * (candidates_max - len(recency_candidates))
        elif len(recency_candidates) > candidates_max:
            recency_candidates = recency_candidates[:candidates_max]

        return recency_candidates


class AdressaRNNInput(RecInputCategoryMixin, RecInputMixin):
    """
        warpper function of torch model to train, validate, and test the models.
    """
    def __init__(self, rec_input_json_path, dict_url2vec, options, \
            dict_url2info={}, dict_glove={}):
        """
            initializer function
        """

        # initialize mixins
        self.load_rec_input(dict_url2vec=dict_url2vec,
                dict_rec_input=load_json(rec_input_json_path), options=options)

        self.load_category(dict_url2vec=dict_url2vec,
                dict_url2info=dict_url2info)

        # datasets will be updated lazily
        self._dataset = {}

    def idx2cate(self, idx):
        """
            get one-hot vector of news category from its index
            :idx: index of news
            :return: one-hot vector of news category
        """
        return self.url2cate(self.idx2url(idx))

    def get_dataset(self, data_type='test'):
        """
            get torch dataset
            :data_type: have to be one of 'train', 'valid', and 'test'
            :return: torch dataset of relavent type
        """
        if data_type not in ['train', 'valid', 'test']:
            data_type = 'test'

        max_seq = 20

        if hasattr(self._dataset, data_type):
            return self._dataset[data_type]

        def pad_sequence(sequence, padding):
            len_diff = max_seq - len(sequence)

            if len_diff < 0:
                return sequence[:max_seq]
            elif len_diff == 0:
                return sequence

            padded_sequence = sequence + [padding] * len_diff

            return padded_sequence

        datas = []

        for timestamp_start, timestamp_end, user_ids, sequence, time_sequence in \
                self._dict_rec_input['dataset'][data_type]:
            pad_indices = [idx for idx in pad_sequence(sequence, self.get_pad_idx())]
            pad_time_indices = [idx for idx in pad_sequence(time_sequence, -1)]
#                pad_seq = [normalize([self.idx2vec(idx)], norm='l2')[0] for idx in pad_indices]
            pad_seq = [self.idx2vec(idx) for idx in pad_indices]

            seq_len = min(len(sequence), max_seq) - 1
            seq_x = pad_seq[:-1]
            seq_y = pad_seq[1:]
            seq_cate = [self.idx2cate(idx) for idx in pad_indices][:-1]
            seq_cate_y = [self.idx2cate(idx) for idx in pad_indices][1:]

            idx_x = pad_indices[:-1]
            idx_y = pad_indices[1:]

            trendy_infos = [self.get_trendy(timestamp, self.get_pad_idx()) \
                    for timestamp in pad_time_indices]

            seq_trendy = [[self.idx2vec(idx) for idx, count in trendy] \
                    for trendy in trendy_infos][1:]
            idx_trendy = [[idx for idx, count in trendy] for trendy in trendy_infos][1:]

            candidate_infos = [self.get_candidates(timestamp, self.get_pad_idx()) \
                    for timestamp in pad_time_indices]

            seq_candi = [[self.idx2vec(idx) for idx, count in candi] \
                    for candi in candidate_infos][1:]
            idx_candi = [[idx for idx, count in candi] for candi in candidate_infos][1:]

            datas.append(
                (seq_x, seq_y, seq_cate, seq_cate_y, seq_len, idx_x, idx_y, seq_trendy, idx_trendy, \
                    seq_candi, idx_candi, timestamp_start, timestamp_end, user_ids)
            )

        self._dataset[data_type] = AdressaDataset(datas)

        return self._dataset[data_type]


def adressa_collate_train(batch):
    """
        collate function of train.
        this function is executed before every batch generation
    """
    batch.sort(key=lambda x: x[4], reverse=True)

    seq_x, seq_y, seq_cate, seq_cate_y, seq_len, x_indices, y_indices, seq_trendy, \
        trendy_indices, seq_candi, candi_indices, \
        timestamp_starts, timestamp_ends, user_ids = zip(*batch)

    return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), torch.FloatTensor(seq_trendy), \
        torch.FloatTensor(seq_cate), torch.FloatTensor(seq_cate_y), torch.IntTensor(seq_len), \
        timestamp_starts, timestamp_ends, \
        x_indices, y_indices, trendy_indices, \
        torch.LongTensor(user_ids)


def adressa_collate(batch):
    """
        collate function of validation, and test.
        this function is executed before every batch generation
    """
    batch.sort(key=lambda x: x[4], reverse=True)

    seq_x, seq_y, seq_cate, seq_cate_y, seq_len, x_indices, y_indices, seq_trendy, \
        trendy_indices, seq_candi, candi_indices, \
        timestamp_starts, timestamp_ends, user_ids = zip(*batch)

    return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), torch.FloatTensor(seq_trendy), \
        torch.FloatTensor(seq_candi), torch.FloatTensor(seq_cate), torch.FloatTensor(seq_cate_y), \
        torch.IntTensor(seq_len), timestamp_starts, timestamp_ends, \
        x_indices, y_indices, trendy_indices, candi_indices, \
        torch.LongTensor(user_ids)


class AdressaRec(object):
    """
        news recommendation class for testing, evaluating, and managing rnn based models
    """
    def __init__(self, model_class, ws_path, torch_input_path, \
            dict_url2vec, options, dict_url2info=None, dict_glove=None, hram_mode=False):
        """
            initializer function.
        """
        super(AdressaRec, self).__init__()

        print("AdressaRec generating ...")
        self._hram_mode = hram_mode

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._ws_path = ws_path
        self._options = options

        dim_article = len(next(iter(dict_url2vec.values())))
        learning_rate = options.learning_rate

        dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)
        self._rnn_input = AdressaRNNInput(dict_rnn_input_path, dict_url2vec, \
                options, dict_url2info=dict_url2info, \
                dict_glove=dict_glove)

        self._train_dataloader, self._valid_dataloader, self._test_dataloader = \
                                self.get_dataloader(dict_url2vec)

        # pass size of users, articles for HRAM
        setattr(options, 'user_size', self._rnn_input.get_user_size())
        setattr(options, 'article_size', self._rnn_input.get_article_size())
        setattr(options, 'article_pad_idx', self._rnn_input.get_pad_idx())

        self._model = model_class(dim_article, self._rnn_input._cate_dim, options).to(self._device)
        self._model.apply(weights_init)

#self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
#self._criterion = nn.MSELoss()
        if self._hram_mode:
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._criterion = nn.BCELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        self._saved_model_path = self._ws_path + '/predictor.pth.tar'


        print("AdressaRec generating Done!")

    def get_dataloader(self, dict_url2vec):
        """
            get torch dataloader for each data_type
            :dict_url2vec: dictionary for translating url to vector
            :return: torch dataloader for each data_type
        """
        train_dataloader = torch.utils.data.DataLoader(self._rnn_input.get_dataset(data_type='train'),
                batch_size=512, shuffle=True, num_workers=16,
                collate_fn=adressa_collate_train)

        valid_dataloader = torch.utils.data.DataLoader(self._rnn_input.get_dataset(data_type='valid'),
                batch_size=512, shuffle=False, num_workers=16,
                collate_fn=adressa_collate_train)

        test_dataloader = torch.utils.data.DataLoader(self._rnn_input.get_dataset(data_type='test'),
                batch_size=64, shuffle=False, num_workers=16,
                collate_fn=adressa_collate)

        return train_dataloader, valid_dataloader, test_dataloader

    def do_train(self, total_epoch=200, early_stop=10):
        """
            train a rnn-based model.
            :total_epoch: maximum number of epoch
            :early_stop: early_stop parametor
            :return: none
        """
        print('start traninig!!')

#start_epoch, best_valid_loss = self.load_model()
        start_epoch = 0
        best_valid_loss = sys.float_info.max

        best_hit_5 = -1.0
        best_mrr_20 = -1.0
        best_auc_20 = -1.0

        sim_cate = getattr(self._options, 'cate_mrr_mode', False)

        if start_epoch < total_epoch:
            endure = 0
            for epoch in range(start_epoch, total_epoch):
                start_time = time.time()
                if endure > early_stop:
                    print('Early stop!')
                    break

                train_loss = self.train()
                valid_loss = self.test()
                hit_5, auc_20, mrr_20 = self.test_mrr_trendy(metric_count=20,
                        candidate_count=20, sim_cate=sim_cate)
        
                best_hit_5 = max(best_hit_5, hit_5)
                best_mrr_20 = max(best_mrr_20, mrr_20)
                best_auc_20 = max(best_auc_20, auc_20)

                print('epoch {} - train loss({:.8f}) valid loss({:.8f})\n \
    test hit_5({:.4f}) best hit_5({:.4f})\n \
    test auc_20({:.4f}) best auc_20({:.4f})\n \
    test mrr_20({:.4f}) best mrr_20({:.4f}) tooks {:.2f}'.format(
                    epoch, train_loss, valid_loss, \
                    hit_5, best_hit_5, \
                    auc_20, best_auc_20, \
                    mrr_20, best_mrr_20, \
                    time.time() - start_time))

#if self._args.save_model and best_mrr_20 == mrr_20:
                if self._options.save_model and valid_loss < best_valid_loss:
                    self.save_model(epoch, valid_loss)
                    print('Model saved! - test mrr_20({}) best mrr_20({})'.format(mrr_20, best_mrr_20))

                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    endure = 0
                else:
                    endure += 1

        return best_hit_5, best_auc_20, best_mrr_20

    def to_device(self, data):
        """
            transfer a tensor to cpu of gpu
            :data: tensor
            :return: transfered tensor
        """
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        else:
            return data

    def get_samples(self, seq_max, indices_y, seq_lens):
        """
            get positive and negative sampled
            :seq_max: maximum number of sequence
            :inddices_y: positive news sample
            :seq_len: length of the sequence
            :return: negative samples
        """
        # generate pos/neg samples
        sample_count = 5
        sample_dummy = np.array([0] * sample_count)

        sample_indices = []
        for b, seq_len in enumerate(seq_lens):
            samples = []
            for s in range(seq_len):
                forbid = [self._rnn_input.get_pad_idx(), indices_y[b][s]]
                while True:
                    sample = np.random.permutation(self._rnn_input.get_article_size())[:sample_count-1]
                    if not np.any(np.isin(sample, forbid)):
                        break;
                sample = np.concatenate([np.array([indices_y[b][s],]), sample], axis=0)
                samples.append(sample)

            if len(samples) < seq_max:
                samples += [sample_dummy] * (seq_max - len(samples))

            sample_indices.append(samples)
        sample_indices = np.stack(sample_indices, axis=0)

        # sample: idx2vec
        sample_vecs = []
        for b in range(sample_indices.shape[0]):
            samples = []
            for s in range(sample_indices.shape[1]):
                sample = []
                for i in range(sample_indices.shape[2]):
                     sample.append(self._rnn_input.idx2vec(i))
                samples.append(sample)
            sample_vecs.append(samples)

        sample_vecs = self.to_device(torch.FloatTensor(sample_vecs))
        sample_indices = self.to_device(torch.LongTensor(sample_indices))

        #correct = self.to_device(torch.LongTensor([1] + [0] * (sample_count - 1)))
        correct = self.to_device(torch.LongTensor([1,]))

        return sample_indices, sample_vecs, correct

    def train(self):
        """
            train a model
            :return: none
        """
        self._model.train()
        train_loss = 0.0
        batch_count = len(self._train_dataloader)

        seq_sums = 0
        seq_count = 0
        for batch_idx, train_input in enumerate(self._train_dataloader):
            input_x_s, input_y_s, input_trendy, input_cate, input_cate_y, seq_lens, \
                timestamp_starts, timestamp_ends, \
                indices_x, indices_y, indices_trendy, user_ids\
                = [self.to_device(input_) for input_ in train_input]

            self._model.zero_grad()
            self._optimizer.zero_grad()

            seq_sums += np.sum(seq_lens.cpu().numpy())
            seq_count += len(seq_lens)

            if self._hram_mode:
                seq_max = input_x_s.size(1)
                sample_indices, sample_vecs, correct = self.get_samples(seq_max, indices_y, seq_lens)

                # inferences
                outputs = self._model(input_x_s, input_trendy, input_cate,
                        seq_lens, user_ids, sample_indices, sample_vecs)

                # loss
                packed_outputs = pack(outputs, seq_lens, batch_first=True).data
                correct = correct.expand(packed_outputs.shape[0])

                loss = self._criterion(packed_outputs, correct)
            else:
                outputs = self._model(input_x_s, input_trendy, input_cate,
                        seq_lens, user_ids)
                packed_outputs = pack(outputs, seq_lens, batch_first=True).data
                packed_y_s = pack(input_y_s, seq_lens, batch_first=True).data

                loss = self._criterion(F.softmax(packed_outputs, dim=1), F.softmax(packed_y_s, dim=1))
            loss.backward()
            self._optimizer.step()

            train_loss += loss.item()
        print('seq len', seq_sums/seq_count)

        return train_loss / batch_count

    def test(self):
        """
            test a model.
            :return: none
        """
        self._model.eval()

        test_loss = 0.0
        sampling_count = 0

        for batch_idx, test_input in enumerate(self._valid_dataloader):
            input_x_s, input_y_s, input_trendy, input_cate, input_cate_y, seq_lens, \
                timestamp_starts, timestamp_ends, \
                indices_x, indices_y, indices_trendy, user_ids\
                = [self.to_device(input_) for input_ in test_input]

            batch_size = input_x_s.shape[0]

            if self._hram_mode:
                seq_max = input_x_s.size(1)
                sample_indices, sample_vecs, correct = self.get_samples(seq_max, indices_y, seq_lens)

                # inferences
                outputs = self._model(input_x_s, input_trendy, input_cate,
                        seq_lens, user_ids, sample_indices, sample_vecs)

                # loss
                packed_outputs = pack(outputs, seq_lens, batch_first=True).data
                #correct = torch.unsqueeze(correct, dim=0).expand(*packed_outputs.shape)
                correct = correct.expand(packed_outputs.shape[0])

                loss = self._criterion(packed_outputs, correct)
            else:
                outputs = self._model(input_x_s, input_trendy, input_cate, seq_lens, user_ids)
                packed_outputs = pack(outputs, seq_lens, batch_first=True).data
                packed_y_s = pack(input_y_s, seq_lens, batch_first=True).data

                loss = self._criterion(F.softmax(packed_outputs, dim=1), F.softmax(packed_y_s, dim=1))

            test_loss += loss.item() * batch_size
            sampling_count += batch_size

        return test_loss / sampling_count

    def test_mrr_trendy(self, metric_count=20, candidate_count=20, max_sampling_count=2000,
            sim_cate=False, attn_mode=False, length_mode=False):
        """
            evaluate a model with hit@5 and mrr@20
            :metric_count: @ number
            :candidate_count: number of candidate articles to test
            :max_sampling_count: limit of evaluation
            :sim_cate: category test for comp_naver
            :attn_mode: flag of attention test mode
            :length_mode: flag of length test mode
            :return: none
        """
        self._model.eval()

        predict_count = 0

        predict_auc = 0.0
        predict_mrr = 0.0
        predict_hit = 0

        sampling_count = 0

        if attn_mode:
            data_by_attn = []
            data_by_attn_count = []
            for _ in range(20):
                data_by_attn.append(0.0)
                data_by_attn_count.append(0)

        if length_mode:
            data_by_length = []
            data_by_length_count = []
            for _ in range(20):
                data_by_length.append(0.0)
                data_by_length_count.append(0)

        for i, data in enumerate(self._test_dataloader, 0):
#            if not attn_mode and sampling_count >= max_sampling_count:
#                continue

            input_x_s, input_y_s, input_trendy, input_candi, input_cate, input_cate_y, seq_lens, \
                timestamp_starts, timestamp_ends, _, indices_y, \
                indices_trendy, indices_candi, user_ids = \
                [self.to_device(i_) for i_ in data]

            outputs = None
            attns = None

            with torch.no_grad():
#                if sim_cate:
#                    outputs, cate_pref = self._model.forward_with_cate(input_x_s,
#                            input_trendy, input_cate, seq_lens, user_ids)
#                elif attn_mode:
#                    outputs = self._model(input_x_s, input_trendy, input_cate,
#                            seq_lens, user_ids, attn_mode=True)
#                else:
#                    outputs = self._model(input_x_s, input_trendy, input_cate,
#                            seq_lens, user_ids)
                if self._hram_mode:
                    sample_indices = torch.cat([torch.unsqueeze(torch.LongTensor(indices_y), dim=2),
                        torch.LongTensor(indices_candi)], dim=2)
                    sample_vecs = torch.cat([torch.unsqueeze(input_y_s, dim=2), input_candi], dim=2)

                    sample_indices = self.to_device(sample_indices)
                    sample_vecs = self.to_device(sample_vecs)

                    # inferences
                    outputs = self._model(input_x_s, input_trendy, input_cate,
                            seq_lens, user_ids, sample_indices, sample_vecs)
                elif attn_mode:
                    outputs = self._model(input_x_s, input_trendy, input_cate, seq_lens, user_ids, attn_mode=True)
                else:
                    outputs = self._model(input_x_s, input_trendy, input_cate, seq_lens, user_ids)

            batch_size = seq_lens.size(0)
            seq_lens = seq_lens.cpu().numpy()
    
            for batch in range(batch_size):
#                if seq_lens[batch] < 2:
#                    continue

                for seq_idx in range(seq_lens[batch]):

#                    if seq_idx < 1:
#                        continue

                    next_idx = indices_y[batch][seq_idx]
                    candidates = indices_candi[batch][seq_idx]

### recency candidate mode
#                    if next_idx not in candidates:
#                        continue
### recency candidate mode : end

                    sampling_count += 1

                    if next_idx in candidates[:candidate_count]:
                        candidates_cut = candidate_count
                    else:
                        candidates_cut = candidate_count - 1

                    if self._hram_mode:
                        scores = outputs[batch][seq_idx][:candidates_cut].cpu().numpy()
                        candidates = [next_idx] + candidates[:candidates_cut-1]
                    else:
                        scores = 1.0 / torch.mean(((input_candi[batch][seq_idx])[:candidates_cut] - \
                                outputs[batch][seq_idx]) ** 2, dim=1)

                        candidates = candidates[:candidates_cut]

                        scores = scores.cpu().numpy()
                        if next_idx not in candidates:
                            next_score = 1.0 / np.mean((np.array(self._rnn_input.idx2vec(next_idx)) - \
                                        outputs[batch][seq_idx].cpu().numpy()) ** 2)

                            candidates = [next_idx] + candidates
                            scores = np.append(next_score, scores)

                    # Naver, additional score as the similarity with category
                    if sim_cate:
                        cate_candi = np.array([self._rnn_input.idx2cate(idx) for idx in candidates])
                        cate_scores = np.dot(cate_candi, np.array(cate_pref[batch][seq_idx]))

                        scores += self._options.cate_weight * scores * cate_scores
            
                    top_indices = (np.array(candidates)[list(filter(lambda x: \
                                    candidates[x] != self._rnn_input.get_pad_idx(), \
                                    scores.argsort()[::-1]))]).tolist()

                    if len(top_indices) < candidate_count:
                        continue

                    hit_index = top_indices.index(next_idx)

                    predict_count += 1

                    if hit_index < 5:
                        predict_hit += 1

                    predict_auc += (candidate_count - 1 - hit_index) / (candidate_count - 1)

                    if hit_index < metric_count:
                        predict_mrr += 1.0 / float(hit_index + 1)
        
                    if length_mode:
                        if hit_index < metric_count:
                            data_by_length[seq_idx] += 1.0 / float(hit_index + 1)
                        data_by_length_count[seq_idx] += 1

        if length_mode:
            length_mode_datas = []
            for idx in range(len(data_by_length)):
                if data_by_length_count[idx] > 0:
                    length_mode_datas.append(str(data_by_length[idx] / data_by_length_count[idx]))
                else:
                    length_mode_datas.append(str(0.0))
            print('=========length_mode=============')
            print(','.join(length_mode_datas))

        return ((predict_hit / float(predict_count)), (predict_auc / float(predict_count)), (predict_mrr / float(predict_count))) if predict_count > 0 else (0.0, 0.0, 0.0)

    def pop(self, metric_count=20, candidate_count=20, length_mode=False):
        """
            evaluate the POP
            :metric_count: @ number
            :candidate_count: number of candidate articles to test
            :length_mode: flag of length test mode
            :return: none
        """
        predict_count = 0
        predict_mrr = 0.0
        predict_hit = 0

        if length_mode:
            data_by_length = []
            data_by_length_count = []
            for _ in range(20):
                data_by_length.append(0.0)
                data_by_length_count.append(0)

        for i, data in enumerate(self._test_dataloader, 0):
            input_x_s, input_y_s, input_trendy, input_candi, input_cate, input_cate_y, seq_lens, \
                timestamp_starts, timestamp_ends, _, \
                indices_y, indices_trendy, indices_candi, user_ids = \
                [self.to_device(i_) for i_ in data]

            batch_size = seq_lens.size(0)
            seq_lens = seq_lens.cpu().numpy()

            for batch in range(batch_size):
                for seq_idx in range(seq_lens[batch]):

#                    if seq_idx < 1:
#                        continue

                    next_idx = indices_y[batch][seq_idx]
                    candidates = indices_candi[batch][seq_idx]

                    #POP@
                    top_indices = candidates[:candidate_count]
                    if next_idx not in top_indices:
                        top_indices = top_indices[:candidate_count-1] + [next_idx]

                    if len(top_indices) < candidate_count:
                        continue

                    hit_index = top_indices.index(next_idx)

                    if hit_index < 5:
                        predict_hit += 1

                    predict_count += 1
                    if hit_index < metric_count:
                        predict_mrr += 1.0 / float(top_indices.index(next_idx) + 1)

                    if length_mode:
                        if hit_index < metric_count:
                            data_by_length[seq_lens[batch]] += 1.0 / float(hit_index + 1)
                        data_by_length_count[seq_lens[batch]] += 1

        if length_mode:
            length_mode_datas = []
            for idx in range(len(data_by_length)):
                if data_by_length_count[idx] > 0:
                    length_mode_datas.append(str(data_by_length[idx] / data_by_length_count[idx]))
                else:
                    length_mode_datas.append(str(0.0))
            print(','.join(length_mode_datas))

        return ((predict_hit / float(predict_count)), (predict_mrr / float(predict_count))) if predict_count > 0 else (0.0, 0.0)

    def save_model(self, epoch, valid_loss):
        """
            save current model to file
            :epoch: current trained epoch
            :valid_loss: current validation loss
            :return: none
        """
        dict_states = {
            'epoch': epoch,
            'valid_loss': valid_loss,
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

        torch.save(dict_states, self._saved_model_path)

    def load_model(self):
        """
            load saved model from the file
            :return: none
        """
        if not os.path.exists(self._saved_model_path):
            return 0, sys.float_info.max

        dict_states = torch.load(self._saved_model_path)
        self._model.load_state_dict(dict_states['model'])
        self._optimizer.load_state_dict(dict_states['optimizer'])

        return dict_states['epoch'], dict_states['valid_loss']


def main():
    """
    main function.
    """
    arr = np.array([5, 0, 1])
    print(arr[arr.argsort()[::-1]].tolist())

if __name__ == '__main__':
    main()
