import torch, pdb
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from PositionalEncoding import PositionalEncoding
from TransformerModelMine import TransformerModelMine,PositionalEncoding

class TenetSeq(torch.nn.Module):
    def __init__(self,params,device='cuda:0'):
        super(TenetSeq,self).__init__()
        print("Method: ", params.method.upper() + '-101')
        self.params                          = params
        self.device                          = device

        # embedding matrices
        #self.user_list_item_embeddings       = torch.nn.Embedding(params.num_user + params.num_list + params.num_item, params.num_factors)
        self.user_embeddings                 = torch.nn.Embedding(params.num_user, params.num_factors)
        self.list_embeddings                 = torch.nn.Embedding(params.num_list, params.num_factors)
        self.item_embeddings                 = torch.nn.Embedding(params.num_item, params.num_factors)

        self.pos_embeddings                  = torch.nn.Embedding(params.max_item_seq_length, params.num_factors)
        self.pos_encoder                     = PositionalEncoding(params.num_factors, dropout=0.2) ##fixed
        self.fc1                             = torch.nn.Linear(params.num_factors,1)
        self.fc2                             = torch.nn.Linear(params.num_factors,1)

        self.user_dropout                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.list_dropout                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout1                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout2                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout3                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout4                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout5                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout6                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.item_dropout7                   = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.pos_dropout                     = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob

        self.sigmoid                         = torch.nn.Sigmoid()

        # weight initialization
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.list_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.pos_embeddings.weight)

        torch.nn.init.zeros_(self.user_embeddings.weight[0])
        torch.nn.init.zeros_(self.list_embeddings.weight[0])##careful about this. Perhaps use concatenation and grad_poob=False,
        torch.nn.init.zeros_(self.item_embeddings.weight[0]) ##Regularization loss on embeddings are used in sasrec code

        # transormer model
        self.trans_model                     = TransformerModelMine(ntoken=params.num_item, ninp=params.num_factors, nhead=1, nhid=params.num_factors, nlayers=1, dropout=0.3)
        self.layer_norm                      = nn.LayerNorm(params.num_factors)

    def forward(self, user_indices, list_indices, item_seq, item_seq_pos=None, item_seq_neg=None, test_item_indices=None, param5=None, train=True,network='gnn'):
        ## flag should removed
        flag = 2
        flag_tran = True

        if train == False:
            user_indices                     = user_indices.reshape(-1,101)[:,0]
            list_indices                     = list_indices.reshape(-1,101)[:,0]
            item_seq                         = item_seq.reshape(-1,101,self.params.max_item_seq_length)[:,0,:] ##101

        self.user_embeds                     = self.user_embeddings(user_indices)
        self.user_embeds                     = self.user_dropout(self.user_embeds)
        self.list_embeds                     = self.list_embeddings(list_indices)
        self.list_embeds                     = self.list_dropout(self.list_embeds)
        self.mask                            = (item_seq != 0).float()

        self.item_seq_embeds                 = self.item_embeddings(item_seq)
        #self.item_seq_embeds                += self.list_embeds.reshape(-1,1,self.params.num_factors)
        self.item_seq_embeds                += (self.user_embeds.reshape(-1,1,self.params.num_factors) + self.list_embeds.reshape(-1,1,self.params.num_factors))
        self.item_seq_embeds                += self.pos_embeddings.weight ##check this carefullly
        self.item_seq_embeds                *= self.mask.reshape(item_seq.shape[0], item_seq.shape[1], 1)

        if flag_tran == True:
            self.out_trans                   = self.trans_model(self.item_seq_embeds.transpose(1,0)).transpose(1,0) ##posemb
            self.item_seq_embeds             = self.out_trans

        if train == True:
            self.item_seq_pos_embeds         = self.item_embeddings(item_seq_pos)
            self.item_seq_pos_embeds         = self.item_dropout3(self.item_seq_pos_embeds)
            self.item_seq_neg_embeds         = self.item_embeddings(item_seq_neg)
            self.item_seq_neg_embeds         = self.item_dropout4(self.item_seq_neg_embeds)
            self.is_target                   = (item_seq_pos != 0).float()

            self.user_item_seq_pos_embeds    = self.user_embeds.reshape(-1,1,self.params.num_factors) * self.item_seq_pos_embeds
            #self.user_item_seq_pos_embeds    = self.item_dropout1(self.user_item_seq_pos_embeds)
            self.list_item_seq_pos_embeds    = self.list_embeds.reshape(-1,1,self.params.num_factors) * self.item_seq_pos_embeds
            #self.list_item_seq_pos_embeds    = self.item_dropout2(self.list_item_seq_pos_embeds)
            self.item_seq_and_seq_pos_embeds = self.item_seq_embeds * self.item_seq_pos_embeds
            #self.item_seq_and_seq_pos_embeds = self.item_dropout3(self.item_seq_and_seq_pos_embeds)

            self.user_item_seq_neg_embeds    = self.user_embeds.reshape(-1,1,self.params.num_factors) * self.item_seq_neg_embeds
            #self.user_item_seq_neg_embeds    = self.item_dropout4(self.user_item_seq_neg_embeds)
            self.list_item_seq_neg_embeds    = self.list_embeds.reshape(-1,1,self.params.num_factors) * self.item_seq_neg_embeds
            #self.list_item_seq_neg_embeds    = self.item_dropout5(self.list_item_seq_neg_embeds)
            self.item_seq_and_seq_neg_embeds = self.item_seq_embeds * self.item_seq_neg_embeds
            #self.item_seq_and_seq_neg_embeds = self.item_dropout6(self.item_seq_and_seq_neg_embeds)

            # only last-item =======================================
            if flag == 0:
                self.pos_logits              = self.sigmoid(torch.sum(self.item_seq_and_seq_pos_embeds, axis=-1))
                self.neg_logits              = self.sigmoid(torch.sum(self.item_seq_and_seq_neg_embeds, axis=-1))

            # only last-item and list  =========================
            if flag == 1:
                self.pos_logits              = self.sigmoid(torch.sum(self.list_item_seq_pos_embeds + self.item_seq_and_seq_pos_embeds, axis=-1))
                self.neg_logits              = self.sigmoid(torch.sum(self.list_item_seq_neg_embeds + self.item_seq_and_seq_neg_embeds, axis=-1))

            # only last-item, list and user ====================
            if flag == 2:
                self.pos_logits              = self.sigmoid(torch.sum(self.user_item_seq_pos_embeds + self.list_item_seq_pos_embeds + self.item_seq_and_seq_pos_embeds, axis=-1))
                self.neg_logits              = self.sigmoid(torch.sum(self.user_item_seq_neg_embeds + self.list_item_seq_neg_embeds + self.item_seq_and_seq_neg_embeds, axis=-1))

            ## Need to remove the bias associated with the attention projection matrices?
            ## normalized seq embeddings for query embeddings?
            ## MultiheadAttention without bias, without weights and with list_embedding as query?
            ## MultiheadAttention in sasrec may not have final projection matrices
            ## residual is done right after multihead_attention inside multiheadattn in sasrec
            ### TEst time one-extra dimension is added. pos_emb is trained on different item emb. Careful about it
            return self.pos_logits, self.neg_logits, self.is_target

        elif train == False:
            self.test_item_embeds            = self.item_embeddings(test_item_indices)
            self.item_seq_embeds             = self.item_seq_embeds.view(-1,1,self.params.max_item_seq_length,self.params.num_factors).repeat(1,101,1,1).view(-1,self.params.max_item_seq_length,self.params.num_factors)
            self.list_embeds                 = self.list_embeds.view(-1,1,self.params.num_factors).repeat(1,101,1).view(-1,self.params.num_factors)
            self.user_embeds                 = self.user_embeds.view(-1,1,self.params.num_factors).repeat(1,101,1).view(-1,self.params.num_factors)

            # only last-item =======================================
            if flag == 0:
                self.pos_logits              = self.sigmoid(torch.sum((self.item_seq_embeds[:,-1,:]) * self.test_item_embeds,axis=-1))

            # only last-item and list  =========================
            if flag == 1:
                self.pos_logits              = self.sigmoid(torch.sum((self.item_seq_embeds[:,-1,:] + self.list_embeds) * self.test_item_embeds,axis=-1))

            # only last-item, list and user ====================
            if flag == 2:
                ## should be modified similar to car
                self.pos_logits                  = self.sigmoid(torch.sum((self.item_seq_embeds[:,-1,:] + self.list_embeds + self.user_embeds) * self.test_item_embeds,axis=-1))
            return self.pos_logits

    def loss(self,):
        pass

