import torch, pdb
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

class BPR(torch.nn.Module):
    def __init__(self,params,device='cuda:0'):
        super(BPR,self).__init__()
        print("Method: ", params.method.upper())
        self.params                = params

        # embedding matrices
        self.user_embeddings       = torch.nn.Embedding(params.num_user, params.num_factors)
        self.list_embeddings       = torch.nn.Embedding(params.num_list, params.num_factors)
        self.item_embeddings       = torch.nn.Embedding(params.num_item, params.num_factors)

        self.user_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.list_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_neg_dropout      = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob

        self.sigmoid               = torch.nn.Sigmoid()

        # weight initialization
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.list_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_pos_indices, user_neg_indices, list_indices, item_indices, item_neg_indices=None, param5=None, param6=None, train=False):
        self.list_embeds                     = self.list_embeddings(list_indices)
        self.list_embeds                     = self.list_dropout(self.list_embeds)
        self.item_embeds                     = self.item_embeddings(item_indices)
        self.item_embeds                     = self.item_dropout(self.item_embeds)

        self.multiplied_output_pos           = self.list_embeds * self.item_embeds
        self.pred_rating_pos                 = torch.sum(self.multiplied_output_pos,axis=1)
        #self.pred_rating                     = torch.maximum(self.pred_rating_pos, 0) ## can be removed

        if train == True:
            self.item_neg_embeds             = self.item_embeddings(item_neg_indices)
            self.item_neg_embeds             = self.item_neg_dropout(self.item_neg_embeds)
            self.multiplied_output_neg       = self.list_embeds * self.item_neg_embeds
            self.pred_rating_neg             = torch.sum(self.multiplied_output_neg,axis=1)

        if train == True:
            return self.pred_rating_pos, self.pred_rating_neg
        else:
            return self.pred_rating_pos

    def loss(self,):
        pass
