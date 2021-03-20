import torch, pdb
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from HyperSAGNN import HyperSAGNN_Model
#from simpleTernary import HyperSAGNN_Model

# gnn
from gnn_utils import normalize_adj
# from torch_geometric.nn import GCNConv, ChebConv, GATConv  # noqa

class GMF(torch.nn.Module):
    def __init__(self,params,device='cuda:0'):
        super(GMF,self).__init__()
        print("Method: ", params.method.upper())
        self.params                      = params

        # embedding matrices
        self.user_list_item_embeddings   = torch.nn.Embedding(params.num_user + params.num_list + params.num_item, params.num_factors)
        self.fc1                         = torch.nn.Linear(params.num_factors, 1)
        self.fc2                         = torch.nn.Linear(params.num_factors, 1)

        self.user_item_list_dropout      = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout1                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout2                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.sigmoid                     = torch.nn.Sigmoid()

        # weight initialization
        torch.nn.init.xavier_uniform_(self.user_list_item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[0])
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[self.params.num_user])
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[self.params.num_user + self.params.num_list])

        # hypersagnn ==================
        self.hypersagnn_model            = HyperSAGNN_Model(n_head=params.n_heads[0], d_model=params.num_factors, d_k=params.d_k, d_v=params.d_k,
                                                 node_embedding=self.user_list_item_embeddings,
                                                 diag_mask=True, bottle_neck=params.num_factors,
                                                 dropout=1.0-params.net_keep_prob).to(device) ## d_k * n_head should be equal to num_factors
        #classifier_model = Classifier(n_head=8,d_model=args.dimensions,d_k=16,d_v=16,node_embedding=node_embedding,
        #diag_mask=args.diag,bottle_neck=bottle_neck).to(device)

    def get_emb_user(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        #emb    = self.user_item_list_dropout(emb)
        output = emb[:,0] * emb[:,2] #user-item
        #output = emb[:,1] * emb[:,2] #list-item
        #output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.user_item_list_dropout(output)
        #output = self.sigmoid(torch.sum(output,axis=1)) #self.user_item_list_dropout(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_list(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        #emb    = self.user_item_list_dropout(emb)
        output = emb[:,1] * emb[:,2] #user-item
        #output = emb[:,1] * emb[:,2] #list-item
        #output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.user_item_list_dropout(output)
        #output = self.sigmoid(torch.sum(output,axis=1)) #self.user_item_list_dropout(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_user_list(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output_user = emb[:,0] * emb[:,2] #user-item
        output_list = emb[:,1] * emb[:,2] #list-item
        output_user = self.dropout1(output_user)
        output_list = self.dropout2(output_list)
        ##output = self.sigmoid(self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1)) #self.user_item_list_dropout(output)
        output = self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1) #self.user_item_list_dropout(output)
        return output

    def get_emb_all_mult(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.dropout1(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_user_list_mf(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output_user = emb[:,0] * emb[:,2] #user-item
        output_list = emb[:,1] * emb[:,2] #list-item
        output_user = self.dropout1(output_user)
        output_list = self.dropout2(output_list)
        output = self.sigmoid(torch.sum(output_user,axis=1) + torch.sum(output_list,axis=1)) #self.user_item_list_dropout(output)
        return output

    def forward(self, user_indices, list_indices, item_indices, param4=None, param5=None, param6=None):
        ##self.hypersagnn_model.node_embedding = self.user_list_item_embeddings ## every time update the node_embeddings ## check whether updation happening

        x = torch.cat([user_indices.reshape(-1,1),
                       list_indices.reshape(-1,1) + self.params.num_user,
                       item_indices.reshape(-1,1) + self.params.num_user + self.params.num_list],
                       dim=1)

        #self.edge_probs = self.get_emb_list(x)
        #self.edge_probs = self.get_emb2(x)
        #self.edge_probs = self.get_emb_user_list(x)
        #self.edge_probs = self.hypersagnn_model(x).reshape(-1)
        #self.edge_probs = self.hypersagnn_model(x).reshape(-1)
        #self.edge_probs = F.sigmoid(self.hypersagnn_model(x).reshape(-1)) ## remove sigmoid

        #self.edge_probs = self.get_emb_user_list_mf(x)
        #self.edge_probs = self.get_emb_user_list(x)
        self.edge_probs = self.get_emb_list(x)
        return self.edge_probs

    def loss(self,):
        pass

