import numpy as np
import scipy.sparse as sp
import ast,sys,operator,os,argparse,re
from time import time
import pdb,pickle

sys.path.append('/home/vijai/tenet/tenet_comparison/tenet/model_code/Utilities/.')
import utils

#==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="threshold dataset.")
    parser.add_argument('--path', nargs='?',
                        default='/home/vijai/tenet/tenet_comparison/SASRec/data/',help   ='Input data path.')
    parser.add_argument('--dataset', nargs='?',
                        default='sample_small',help='Choose a dataset.')

    parser.add_argument('--verbose',                            type=int, default=1, help='verbose.')
    return parser.parse_args()

def convert_to_sasrec_format(fname, dct):
    with open(fname,'w') as fout:
        for lst in dct:
            for item in dct[lst]:
                fout.write(str(lst+1) + ' '+ str(item+1) + '\n')
'''
def convert_to_sasrec_negatives(fname, neg_dct):
    for key in neg_dct:
        for val_list in neg_dct[

    utils.store_pickle(fname, neg_dct)
    print(dct)
    with open(fname,'w') as fout:
        for lst in dct:
            for item in dct[lst]:
                fout.write(str(lst+1) + ' ' + str(item+1) + '\n')
'''

if __name__ == '__main__':

    args = parse_args()
    print(args)

    path                     = args.path
    dataset                  = args.dataset
    dataset_path             = os.path.join(path,dataset)

    list_items_dct           = utils.load_pickle(dataset_path+'.list_items.pkl')
    dumm_dct                 = dict()
    neg_dct_valid                = {(0,8): [1,2,3,4,5] * 20,(1,7): [1,2,3,4,5] * 20,(2,4): [1,2,3,4,5] * 20,(3,6): [1,2,3,4,5] * 20,(4,2): [1,2,3,4,5] * 20,(5,9): [1,2,3,4,5] * 20,(6,5): [1,2,3,4,5] * 20,(7,7): [1,2,3,4,5] * 20,(8,5): [1,2,3,4,5] * 20}
    neg_dct_test                = {(0,1): [1,2,3,4,5] * 20,(1,9): [1,2,3,4,5] * 20,(2,5): [1,2,3,4,5] * 20,(3,8): [1,2,3,4,5] * 20,(4,1): [1,2,3,4,5] * 20,(5,10): [1,2,3,4,5] * 20,(6,12): [1,2,3,4,5] * 20,(7,8): [1,2,3,4,5] * 20,(8,11): [1,2,3,4,5] * 20}

    utils.store_pickle(dataset_path + '.list_items.train_valid_test.pkl', [dumm_dct, neg_dct_valid, neg_dct_test])
    sys.exit()
    '''
    print(list_items_dct)
    # ======================
    convert_to_sasrec_format(fname=dataset_path+'.sasrec.txt',dct=list_items_dct)

    convert_to_sasrec_negatives(fname=dataset_path+'.sasrec.negatives.pkl', valid_dct)
    convert_to_sasrec_negatives(fname=dataset_path+'.sasrec.negatives.pkl', test_dct)
    '''
