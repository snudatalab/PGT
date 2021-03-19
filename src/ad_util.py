""" PGT: Accurate News Recommendation Coalescing Personal and Global Temporal Preferences

Code Authors:
    - Bonhun Koo, (darkgs@snu.ac.kr) Data Mining Lab. at Seoul National University.
    - U Kang, (ukang@snu.ac.kr) Associate Professor.

    File: src/ad_util.py
    - Utility function used in the poject generally.

"""

import os
import json
import datetime

import numpy as np

import torch
import torch.nn as nn

def find_best_url(event_dict=None):
    """
        find the best url from the adressa dataset
        :event_dict: dictionary containing adressa user interaction events
        :return: the best url
    """
    if event_dict == None:
        return None

    url_keys = ['url', 'cannonicalUrl', 'referrerUrl']
    black_list = ['http://google.no', 'http://facebook.com', 'http://adressa.no/search']

    best_url = None
    for key in url_keys:
        url = event_dict.get(key, None)
        if url == None:
            continue

        if url.count('/') < 3:
            continue

        black_url = False
        for black in black_list:
            if url.startswith(black):
                black_url = True
                break
        if black_url:
            continue

        if (best_url == None) or (len(best_url) < len(url)):
            best_url = url

    return best_url

def write_log(log):
    """
        write a log to file
        :log: string of log
    """
    with open('log.txt', 'a') as log_f:
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_f.write(time_stamp + ' ' + log + '\n')


def get_files_under_path(p_path=None):
    """
        get all files including sub-directories
        :p_path: parent path
        :return: all files
    """
    ret = []

    if p_path == None:
        return ret

    for r, d, files in os.walk(p_path):
        for f in files:
            file_path = os.path.join(r,f)
            if not os.path.isfile(file_path):
                continue

            ret.append(file_path)

    return ret

def weights_init(m):
    """
        initialization function to weight of nodes
        :m: a node
    """
    if isinstance(m, nn.Conv1d):
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)


def load_json(dict_path=None):
    """
        load a json from to dictinary
        :dict_path: path of json
        :return: relavent dictionary object
    """
    dict_ret = {}

    if dict_path == None:
        return

    with open(dict_path, 'r') as f_dict:
        dict_ret = json.load(f_dict)

    return dict_ret

def option2str(options):
    """
        translate option dictionary to string representation
        :options: option dictionary
        :return: unique string of the option
    """
    items = [(key, option) for key, option in options.__dict__.items() if '/' not in str(option) ]
    items.sort(key=lambda x: x[0])
    items = [key + '-' + str(option) for key, option in items]
    return '__'.join(items)


