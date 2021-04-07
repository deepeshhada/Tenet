import os
import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_embeddings(args, mode):
    load_path = os.path.join('HNHN', args.dataset_name + "_raw", mode + '_embeddings.npy')

    start_idx = 0
    if mode == "item":
        start_idx = args.num_users
    elif mode == "list":
        start_idx = args.num_users + args.num_items

    embed_vec = np.load(load_path)
    embed_dict = dict(enumerate(embed_vec, start_idx))
    return embed_dict


def load_data_dict(root_path):
    n_edges = torch.load(os.path.join(root_path, 'n_author.pth'))
    n_nodes = torch.load(os.path.join(root_path, 'n_paper.pth'))
    classes = torch.load(os.path.join(root_path, 'classes.pth'))
    edge_classes = torch.load(os.path.join(root_path, 'author_classes.pth'))
    node_edge = torch.load(os.path.join(root_path, 'paper_author.pth'))
    edge_node = torch.load(os.path.join(root_path, 'author_paper.pth'))
    nodewt = torch.load(os.path.join(root_path, 'paperwt.pth'))
    edgewt = torch.load(os.path.join(root_path, 'authorwt.pth'))
    node_X = torch.load(os.path.join(root_path, 'paper_X.pth'))
    train_len = torch.load(os.path.join(root_path, 'train_len.pth'))
    val_len = torch.load(os.path.join(root_path, 'val_len.pth'))
    test_len = torch.load(os.path.join(root_path, 'test_len.pth'))
    cls2idx = torch.load(os.path.join(root_path, 'user_item_cls_map.pth'))
    train_negatives = torch.load(os.path.join(root_path, 'train_negatives.pth'))

    data_dict = {
        'n_author': n_edges,
        'n_paper': n_nodes,
        'classes': classes,
        'author_classes': edge_classes,
        'paper_author': node_edge,
        'author_paper': edge_node,
        'paperwt': nodewt,
        'authorwt': edgewt,
        'paper_X': node_X,
        'train_len': train_len,
        'val_len': val_len,
        'test_len': test_len,
        'user_item_cls_map': cls2idx,
        'train_negatives': train_negatives
    }

    return data_dict

