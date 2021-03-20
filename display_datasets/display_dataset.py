import sys,math,argparse,os,pdb,random
import numpy as np
import torch
from time import time

sys.path.append('./.')
sys.path.append('./Utilities/.')
import utils
dir = '/home/vijai/tenet/data_tenet/sample_small/'
dataset = 'sample_small'
path = dir + dataset

user_lists_dct = utils.load_pickle(path+'.user_lists.ided.pkl')
list_items_dct = utils.load_pickle(path+'.list_items.ided.pkl')
train,valid,test = utils.load_pickle(path+'.list_items.train_valid_test.pkl')

print('user_lists_dct')
print(user_lists_dct)
print('list_items_dct')
print(list_items_dct)

print('train')
print(train)
print('valid')
print(valid)
print('test')
print(test)
utils.store_pickle(path + 'sstt',[train,train,valid,test])
print('Hll')
print(utils.load_pickle(path+'sstt' ))
