import numpy as np
import scipy.sparse as sp
import ast,sys,operator,os,argparse,re,pdb
from time import time

sys.path.append('../model_code/Utilities/.')
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='''convert to zero-indexed ids.''')
    parser.add_argument('--path', nargs='?',default='/home/vijai/tenet/data_tenet/aotm_small_small/tenet/aotm/',help   ='Input data path.')
    parser.add_argument('--dataset', nargs='?',default='aotm',help='Choose a dataset.')
    return parser.parse_args()

class similarity_calculation:
    def __init__(self,args):
        dirname     = args.path +'/'
        filename    = args.dataset
        path        = dirname + filename

        self.user_emb       = utils.load_npy(path+'.user_embed.npy')
        self.list_emb       = utils.load_npy(path+'.list_embed.npy')
        self.item_emb       = utils.load_npy(path+'.item_embed.npy')

        utils.store_pickle(path + '.dummy.npy', [self.user_emb,self.list_emb,[1,2,4]])
        pdb.set_trace()
        self.z       = utils.load_pickle(path+'.dummy.npy')

if __name__ == '__main__':

    args = parse_args()
    print(args , "\n")
    id_assign = similarity_calculation(args)
    print("Finished.")
