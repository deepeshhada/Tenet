from BPR import BPR
from MF import MF
from GMF import GMF
from Tenet_GNN_SEQ import Tenet_Gnn_Seq

class Models(object):
    def __init__(self,params,device='cuda:0'):
        if params.method == 'tenet' and 'gnn' in params.include_networks and 'seq' in params.include_networks:
            self.model   = Tenet_Gnn_Seq(params, device)
        elif params.method == 'tenet' and 'gnn' in params.include_networks:
            self.model = Tenet_Gnn_Seq(params, device)
        elif params.method == 'tenet' and 'seq' in params.include_networks:
            self.model = Tenet_Gnn_Seq(params, device)
        elif params.method == 'bpr':
            self.model = BPR(params)
        elif params.method == 'mf':
            self.model     = MF(params)
        elif params.method == 'gmf':
            self.model     = GMF(params)

    def get_model(self):
        return self.model
