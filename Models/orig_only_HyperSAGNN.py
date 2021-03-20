import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math, pdb

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids = [0, 1]
#device = 'cpu' ##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from EncoderLayer import EncoderLayer
from ScaledDotProductAttention import ScaledDotProductAttention
from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward
#from MultipleEmbedding import MultipleEmbedding
from utils_HyperSAGNN import get_non_pad_mask,get_attn_key_pad_mask
    
class HyperSAGNN_Model(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            node_embedding,
            diag_mask,
            bottle_neck,
            **args):
        super().__init__()
        
        self.pff_classifier = PositionwiseFeedForward(
            [d_model, 1], reshape=True, use_bias=True)
        
        self.node_embedding = node_embedding
        self.encode1 = EncoderLayer(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout_mul=0.3,
            dropout_pff=0.4,
            diag_mask=diag_mask,
            bottle_neck=bottle_neck)
        # self.encode2 = EncoderLayer(n_head, d_model, d_k, d_v, dropout_mul=0.0, dropout_pff=0.0, diag_mask = diag_mask, bottle_neck=bottle_neck)
        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def get_node_embeddings(self, x,return_recon = False):
        
        # shape of x: (b, tuple)
        sz_b, len_seq = x.shape
        # print(torch.max(x), torch.min(x))
        
        x, recon_loss = self.node_embedding(x.view(-1))
        if return_recon:
            return x.view(sz_b, len_seq, -1), recon_loss
        else:
            return x.view(sz_b, len_seq, -1)
    
    def get_embedding(self, x, slf_attn_mask, non_pad_mask,return_recon = False):
        if return_recon:
            x, recon_loss = self.get_node_embeddings(x,return_recon)
        else:
            x = self.get_node_embeddings(x, return_recon)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        # dynamic, static1, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
        if return_recon:
            return dynamic, static, attn, recon_loss
        else:
            return dynamic, static, attn
    
    def get_embedding_static(self, x):
        if len(x.shape) == 1:
            x = x.view(-1, 1)
            flag = True
        else:
            flag = False
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)
        x = self.get_node_embeddings(x)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        # dynamic, static, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
        if flag:
            return static[:, 0, :]
        return static
    
    def forward(self, x, mask=None, get_outlier=None, return_recon = False):
        x = x.long()
            
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)
        
        if return_recon:
            dynamic, static, attn, recon_loss = self.get_embedding(x, slf_attn_mask, non_pad_mask,return_recon)
        else:
            dynamic, static, attn = self.get_embedding(x, slf_attn_mask, non_pad_mask, return_recon)
        dynamic = self.layer_norm1(dynamic)
        static = self.layer_norm2(static)
        sz_b, len_seq, dim = dynamic.shape
        
        if self.diag_mask_flag == 'True':
            output = (dynamic - static) ** 2
        else:
            output = dynamic
        
        output = self.pff_classifier(output)
        output = torch.sigmoid(output)
        
        
        if get_outlier is not None:
            k = get_outlier
            outlier = (
                    (1 -
                     output) *
                    non_pad_mask).topk(
                k,
                dim=1,
                largest=True,
                sorted=True)[1]
            return outlier.view(-1, k)
        
        mode = 'sum'
        
        if mode == 'min':
            output, _ = torch.max(
                (1 - output) * non_pad_mask, dim=-2, keepdim=False)
            output = 1 - output
        
        elif mode == 'sum':
            output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
            mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
            output /= mask_sum
        elif mode == 'first':
            output = output[:, 0, :]
            
        if return_recon:
            return output, recon_loss
        else:
            return output
