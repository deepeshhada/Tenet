## relu to tanh in gcnconv
## some parts are in abc2
import torch, pdb
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from HyperSAGNN import HyperSAGNN_Model
#from simpleTernary import HyperSAGNN_Model

# gnn
from gnn_utils import normalize_adj
#from torch_geometric.nn import GCNConv, ChebConv, GATConv  # noqa
from my_GCNConv import GCNConv

class Tenet(torch.nn.Module):
    def __init__(self,params,device='cuda:0'):
        super(Tenet,self).__init__()
        print("Method: ", params.method.upper()+ '-HGNN')
        self.params                      = params

        # embedding matrices
        self.user_list_item_embeddings   = torch.nn.Embedding(params.num_user + params.num_list + params.num_item, params.num_factors)
        self.fc1                         = torch.nn.Linear(params.num_factors, 1)
        self.fc2                         = torch.nn.Linear(params.num_factors, 1)
        self.fc3                         = torch.nn.Linear(params.hid_units[-1], 1)
        self.fc4                         = torch.nn.Linear(params.hid_units[-1], 1)

        self.user_item_list_dropout      = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout1                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.dropout2                    = torch.nn.Dropout(1.0 - params.keep_prob) ## keep_prob
        self.sigmoid                     = torch.nn.Sigmoid()

        # weight initialization
        ##torch.nn.init.xavier_uniform_(self.user_list_item_embeddings.weight)
        torch.nn.init.xavier_normal_(self.user_list_item_embeddings.weight)
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[0])
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[self.params.num_user])
        torch.nn.init.zeros_(self.user_list_item_embeddings.weight[self.params.num_user + self.params.num_list])

        # gnn ==========================
        self.user_indices           = torch.tensor(np.array(range(params.num_user))).to(device)
        self.list_indices           = torch.tensor(np.array(range(params.num_list))).to(device)
        self.item_indices           = torch.tensor(np.array(range(params.num_item))).to(device)

        self.user_conv1             = GCNConv(params.num_factors,   params.hid_units[-2], cached=True, normalize=True,add_self_loops=True) ##normalize=True
        self.user_conv2             = GCNConv(params.hid_units[-2], params.hid_units[-1], cached=True, normalize=True,add_self_loops=True)

        self.list_conv1             = GCNConv(params.num_factors,   params.hid_units[-2], cached=True, normalize=True,add_self_loops=True)
        self.list_conv2             = GCNConv(params.hid_units[-2], params.hid_units[-1], cached=True, normalize=True,add_self_loops=True)

        self.item_conv1             = GCNConv(params.num_factors,   params.hid_units[-2], cached=True, normalize=True,add_self_loops=True)
        self.item_conv2             = GCNConv(params.hid_units[-2], params.hid_units[-1], cached=True, normalize=True,add_self_loops=True)


        if params.args.knn_graph == 'True':
            self.user_param_indices     = params.dataset_obj.user_edge_index
            self.list_param_indices     = params.dataset_obj.list_edge_index
            self.item_param_indices     = params.dataset_obj.item_edge_index
            self.user_param_weights, self.list_param_weights, self.item_param_weights = None, None, None ##crucial to note
        else:
            self.user_adj_mat           = params.user_adj_mat.tocoo()
            self.user_adj_mat.setdiag(0); self.user_adj_mat.eliminate_zeros()
            #pdb.set_trace()
            self.user_param_indices     = torch.LongTensor(self.user_adj_mat.nonzero()).to(device)
            self.user_param_weights     = torch.FloatTensor(self.user_adj_mat.data).to(device) ##weight check

            self.list_adj_mat           = params.list_adj_mat.tocoo()
            self.list_adj_mat.setdiag(0); self.list_adj_mat.eliminate_zeros()
            self.list_param_indices     = torch.LongTensor(self.list_adj_mat.nonzero()).to(device)
            self.list_param_weights     = torch.FloatTensor(self.list_adj_mat.data).to(device) ##weight check

            self.item_adj_mat           = params.item_adj_mat.tocoo()
            self.item_adj_mat.setdiag(0); self.item_adj_mat.eliminate_zeros()
            self.item_param_indices     = torch.LongTensor(self.item_adj_mat.nonzero()).to(device)
            self.item_param_weights     = torch.FloatTensor(self.item_adj_mat.data).to(device) ##weight check
            if params.args.user_adj_weights == 'False':
                self.user_param_weights, self.list_param_weights, self.item_param_weights = None, None, None ##crucial to note

        # dropouts gnn part
        self.user_gnn_dropout       = torch.nn.Dropout(1.0 - params.gnn_keep_prob) ## keep_prob
        self.list_gnn_dropout       = torch.nn.Dropout(1.0 - params.gnn_keep_prob) ## keep_prob
        self.item_gnn_dropout       = torch.nn.Dropout(1.0 - params.gnn_keep_prob) ## keep_prob

        # hgnn ==================================
        self.hypersagnn_model       = HyperSAGNN_Model(n_head=params.n_heads[0], d_model=params.hid_units[-1], d_k=params.hid_units[-1], d_v=params.hid_units[-1],
                                                 node_embedding=self.user_list_item_embeddings,
                                                 diag_mask=True, bottle_neck=params.hid_units[-1],
                                                 dropout=1.0-params.net_keep_prob).to(device)

        self.ind = 0



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
        output = self.sigmoid(self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_all_mult(self, x, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.dropout1(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_all_mult2(self, x, user_list_item_embeddings, mask=None, get_outlier=None, return_recon = False):
        emb    = self.user_list_item_embeddings(x)
        output = emb[:,0] * emb[:,1] * emb[:,2] #user-list-item
        output = self.dropout1(output)
        output = self.sigmoid(self.fc1(output).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def get_emb_user_list2(self, x, user_list_item_embeddings, mask=None, get_outlier=None, return_recon = False):
        emb    = user_list_item_embeddings[x]
        output_user = emb[:,0] * emb[:,2] #user-item
        output_list = emb[:,1] * emb[:,2] #list-item
        output_user = self.dropout1(output_user)
        output_list = self.dropout2(output_list)
        output = self.sigmoid(self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1)) #self.user_item_list_dropout(output)
        #output = self.fc1(output_user).reshape(-1) + self.fc2(output_list).reshape(-1) #self.user_item_list_dropout(output)
        return output

    def get_emb_user_list3(self, x, user_list_item_embeddings, mask=None, get_outlier=None, return_recon = False):
        emb    = user_list_item_embeddings[x]
        output_user = emb[:,0] * emb[:,2] #user-item
        output_list = emb[:,1] * emb[:,2] #list-item
        output_user = self.dropout1(output_user)
        output_list = self.dropout2(output_list)
        output = self.sigmoid(self.fc3(output_user).reshape(-1) + self.fc4(output_list).reshape(-1)) #self.user_item_list_dropout(output)
        return output

    def forward(self, user_indices, list_indices, item_indices, param4=None, param5=None, param6=None, network='gnn', include_hgnn=False):

        # gnn_user ==============================
        user_x = self.user_list_item_embeddings(self.user_indices)
        user_x = F.relu(self.user_conv1(user_x, self.user_param_indices, self.user_param_weights))
        user_x = self.user_gnn_dropout(user_x)
        user_x = self.user_conv2(user_x, self.user_param_indices, self.user_param_weights)

        # gnn_list ==============================
        list_x = self.user_list_item_embeddings(self.params.num_user+self.list_indices)
        list_x = F.relu(self.list_conv1(list_x, self.list_param_indices, self.list_param_weights))
        list_x = self.list_gnn_dropout(list_x)
        list_x = self.list_conv2(list_x, self.list_param_indices, self.list_param_weights)

        # gnn_item ==============================
        item_x = self.user_list_item_embeddings(self.params.num_user+self.params.num_list+self.item_indices)
        item_x = F.relu(self.item_conv1(item_x, self.item_param_indices, self.item_param_weights))
        item_x = self.item_gnn_dropout(item_x)
        item_x = self.item_conv2(item_x, self.item_param_indices, self.item_param_weights)

        # residual or concatenation

        user_list_item_gnn_emb = torch.cat([user_x, list_x, item_x],dim=0)
        x = torch.cat([user_indices.reshape(-1,1),
                       list_indices.reshape(-1,1) + self.params.num_user,
                       item_indices.reshape(-1,1) + self.params.num_user + self.params.num_list],
                       dim=1)
        self.edge_probs_gnn    = self.get_emb_user_list3(x,user_list_item_gnn_emb) # return this for gnn not hgnn

        # hgnn =======================
        if include_hgnn == True:
            self.edge_probs_hgnn   = self.hypersagnn_model(x, user_list_item_gnn_emb).reshape(-1)
            self.edge_probs        = (self.edge_probs_hgnn + self.edge_probs_gnn)/2
        else:
            self.edge_probs        = self.edge_probs_gnn
        ##self.edge_probs        = self.edge_probs_hgnn
        ##self.edge_probs        = self.edge_probs_gnn

        return self.edge_probs
