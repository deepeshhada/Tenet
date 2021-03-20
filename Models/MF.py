import torch, pdb
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

class MF(torch.nn.Module):
    def __init__(self,params,device='cuda:0'):
        super(MF,self).__init__()
        print("Method: ", params.method.upper())
        self.params                = params

        # embedding matrices
        self.user_embeddings       = torch.nn.Embedding(params.num_user, params.num_factors)
        self.list_embeddings       = torch.nn.Embedding(params.num_list, params.num_factors)
        self.item_embeddings       = torch.nn.Embedding(params.num_item, params.num_factors)
        self.fc                    = torch.nn.Linear(params.num_factors,1)

        self.user_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.list_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob

        self.sigmoid               = torch.nn.Sigmoid()

        # weight initialization
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.list_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_indices, list_indices, item_indices, param4=None, param5=None, param6=None):
        self.user_embeds                     = self.user_embeddings(user_indices)
        self.user_embeds                     = self.user_dropout(self.user_embeds)
        self.list_embeds                     = self.list_embeddings(list_indices)
        self.list_embeds                     = self.list_dropout(self.list_embeds)
        self.item_embeds                     = self.item_embeddings(item_indices)
        self.item_embeds                     = self.item_dropout(self.item_embeds)

        self.list_item_interactions          = self.list_embeds * self.item_embeds
        #rating_pred                          = self.sigmoid(self.fc(self.list_item_interactions).reshape(-1))
        rating_pred                          = self.sigmoid(torch.sum(self.list_item_interactions,axis=1))

        return rating_pred

    def loss(self,):
        pass
