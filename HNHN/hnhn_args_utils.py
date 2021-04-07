"""
    Utilities for the framework.
"""
import pandas as pd
import numpy as np
import os
import random
import torch
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from argparse import Namespace

import pickle
import warnings
import sklearn.metrics
warnings.filterwarnings('ignore')

res_dir = 'results'
data_dir = 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args(params):
    config = Namespace()

    config.batch_size = params.batch_size
    config.valid_batch_size = params.valid_batch_siz
    config.valid_dim = 101
    config.num_ng = params.num_negatives
    config.dataset_name = params.dataset
    config.learning_rate = params.hnhn_lr

    config.verbose = False
    config.exp_wt = False
    config.do_svd = False
    config.method = 'hypergcn'
    config.kfold = 1
    config.predict_edge = True
    config.edge_linear = False
    config.alpha_e = -0.1
    config.alpha_v = -0.1
    config.dropout_p = 0.3
    config.n_layers = 1
    config.seed = 20
    config.top_k = 10
    config.embed_dim = 300

    if config.dataset_name == 'zhihu':
        config.num_users = 1103
        config.num_items = 10936
        config.num_lists = 8996
        config.train_len = 178208
        config.val_len = 8186
        config.test_len = 8472

    return config


def readlines(path):
    with open(path, 'r') as f:
        return f.readlines()
    

def get_label_percent(dataset_name):
    if dataset_name == 'cora':
        return .052
    elif dataset_name == 'citeseer':
        return .15 
    elif dataset_name == 'dblp':
        return .04
    else:
        return .15
    
    
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
