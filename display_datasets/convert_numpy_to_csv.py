import numpy as np,pdb,sys

sys.path.append('./Utilities/.')
import utils

path = '/home/vijai/tenet/data_tenet/required/zhihu_small/tenet/zhihu/zhihu'
#path = '/home/vijai/tenet/all_data_tenet/sample_small/sample_small'
fname1 = path + '.user_lists.ided.pkl'
fname2 = path + '.list_items.train_valid_test.pkl'
fnameli = path + '.list_items.ided.pkl'
'''
fout1  = path + '.user_list.txt'
fout2  = path + '.train.txt'
fout3  = path + '.train2.txt'
fout4  = path + '.valid.txt'
fout5  = path + '.test.txt'
'''

user_lists = utils.load_pickle(fname1)
list_items = utils.load_pickle(fnameli)
#utils.store_npy_to_csv('',user_lists)


train,train2,valid,test = utils.load_pickle(fname2)

pdb.set_trace()
print("hello")
