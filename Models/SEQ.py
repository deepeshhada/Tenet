import torch, pdb
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from PositionalEncoding import PositionalEncoding
from TransformerModel import TransformerModel

class SEQ(torch.nn.Module):
    def __init__(self,params,device='cuda:0'):
        super(SEQ,self).__init__()
        print("Method: ", params.method.upper())
        self.params                = params
        self.device                = device

        # embedding matrices
        self.user_embeddings       = torch.nn.Embedding(params.num_user, params.num_factors)
        self.list_embeddings       = torch.nn.Embedding(params.num_list, params.num_factors)
        self.item_embeddings       = torch.nn.Embedding(params.num_item, params.num_factors)
        self.pos_embeddings        = torch.nn.Embedding(params.max_item_seq_length, params.num_factors)
        self.fc1                   = torch.nn.Linear(params.num_factors,1)
        self.fc2                   = torch.nn.Linear(params.num_factors,1)

        self.user_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.list_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout          = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout2         = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout3         = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout4         = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.pos_dropout           = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob

        self.sigmoid               = torch.nn.Sigmoid()

        # weight initialization
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.list_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.pos_embeddings.weight)

        torch.nn.init.zeros_(self.user_embeddings.weight[0])
        torch.nn.init.zeros_(self.list_embeddings.weight[0])
        torch.nn.init.zeros_(self.item_embeddings.weight[0])

        # transormer model
        self.trans_model           = TransformerModel(params.num_item, params.num_factors, 1, params.num_factors, 1, 0.4)
        self.layer_norm            = nn.LayerNorm(params.num_factors)

    def forward(self, user_indices, list_indices, item_seq, item_seq_pos=None, item_seq_neg=None, test_item_indices=None, param5=None, train=True):
        self.user_embeds                     = self.user_embeddings(user_indices)
        self.user_embeds                     = self.user_dropout(self.user_embeds)
        self.list_embeds                     = self.list_embeddings(list_indices)
        self.list_embeds                     = self.list_dropout(self.list_embeds)

        self.out_trans                       = self.trans_model(item_seq.T).transpose(1,0)
        self.item_seq_embeds                 = self.out_trans
        self.item_seq_embeds                 = self.item_dropout2(self.item_seq_embeds)

        if train == True:
            self.item_seq_pos_embeds         = self.item_embeddings(item_seq_pos)
            self.item_seq_pos_embeds         = self.item_dropout3(self.item_seq_pos_embeds)
            self.item_seq_neg_embeds         = self.item_embeddings(item_seq_neg)
            self.item_seq_neg_embeds         = self.item_dropout4(self.item_seq_neg_embeds)
            self.is_target                   = (item_seq_pos != 0).float()

            self.pos_logits                  = self.sigmoid(torch.sum((self.item_seq_embeds ) * self.item_seq_pos_embeds,axis=-1))
            self.neg_logits                  = self.sigmoid(torch.sum((self.item_seq_embeds ) * self.item_seq_neg_embeds,axis=-1))
            '''
            self.pos_logits                  = self.sigmoid(torch.sum((self.item_seq_embeds + self.list_embeds.reshape(self.item_seq_embeds.shape[0],1, self.item_seq_embeds.shape[2])) *
                                                                      self.item_seq_pos_embeds,axis=-1))
            self.neg_logits                  = self.sigmoid(torch.sum((self.item_seq_embeds + self.list_embeds.reshape(self.item_seq_embeds.shape[0], 1,self.item_seq_embeds.shape[2])) *
                                                                      self.item_seq_neg_embeds,axis=-1))
            '''
            return self.pos_logits, self.neg_logits, self.is_target

        elif train == False:
            self.test_item_embeds            = self.item_embeddings(test_item_indices)
            self.pos_logits                  = self.sigmoid(torch.sum((self.item_seq_embeds[:,-1,:] ) * self.test_item_embeds,axis=-1))
            return self.pos_logits

    def loss(self,):
        pass
