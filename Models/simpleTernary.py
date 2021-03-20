import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math, pdb

device = 'cpu' ##

class HyperSAGNN_Model(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            node_embedding,
            diag_mask,
            bottle_neck,#device,
            **args):
        super().__init__()

        self.node_embedding = node_embedding
        self.sigmoid               = torch.nn.Sigmoid()

    def forward(self, x, mask=None, get_outlier=None, return_recon = False):
        x = x.long()
        #pdb.set_trace()
        sz_b, num_emb = x.shape
        emb = self.node_embedding[x.view(-1)]
        y = emb.view(sz_b, num_emb, -1)
        output = y[:,0] * y[:,1] * y[:,2]
        output = torch.sum(output, axis = 1)
        output = self.sigmoid(output)

        return output