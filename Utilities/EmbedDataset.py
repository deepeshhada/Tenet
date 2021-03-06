import pickle,itertools,pdb
from time import time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

from Dataset import Dataset
from TenetDataset import TenetDataset
import utils
import torch
#from torch_cluster import knn_graph

class EmbedDataset(TenetDataset):
    def __init__(self,args):
        TenetDataset.__init__(self,args)
        path = args.path+args.dataset
        cosine = args.cosine=='True'
        print('Inside Embed Dataset')

        '''
        self.user_embed = torch.from_numpy(utils.load_npy(path + '.user_embed.npy')).to(args.device)
        self.list_embed = torch.from_numpy(utils.load_npy(path + '.list_embed.npy')).to(args.device)
        self.item_embed = torch.from_numpy(utils.load_npy(path + '.item_embed.npy')).to(args.device)
        #pdb.set_trace()

        #batch = torch.tensor([0, 0, 0, 0])
        #edge_index = knn_graph(x, k=3, batch=batch, loop=False)
        print("Cosine:",cosine)
        self.user_edge_index = knn_graph(self.user_embed, k=args.knn_k, batch=None, loop=False,cosine=cosine)
        self.list_edge_index = knn_graph(self.list_embed, k=args.knn_k, batch=None, loop=False,cosine=cosine)
        self.item_edge_index = knn_graph(self.item_embed, k=args.knn_k, batch=None, loop=False,cosine=cosine)
        '''

        if cosine == False:
            self.user_edge_index, self.list_edge_index, self.item_edge_index = utils.load_pickle(args.path + '/' + args.embed_type + '/' + str(args.knn_k) + '/' + args.dataset + '.user_list_item_knn.pkl')
        else:
            self.user_edge_index, self.list_edge_index, self.item_edge_index = utils.load_pickle(args.path + '/' + args.embed_type + '/' + str(args.knn_k) + '_cosine/' + args.dataset + '.user_list_item_knn.pkl')
        # ==========================================
