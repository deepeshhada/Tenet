import sys,pdb
sys.path.append('./Utilities/.')
import utils

#obj = utils.load_pickle('/home/vijai/tenet/data_tenet/test/test.item_id.pkl')
obj1 = utils.load_pickle('/home/vijai/tenet_comparison/SASRec/data/sample_small/sample_small.list_items.ided.pkl')
obj2 = utils.load_pickle('/home/vijai/tenet_comparison/SASRec/data/sample_small/sample_small.user_lists.ided.pkl')
obj3 = utils.load_pickle('/home/vijai/tenet_comparison/SASRec/data/sample_small/sample_small.list_items.train_valid_test.pkl')
obj4 = utils.load_pickle('/home/vijai/tenet_comparison/SASRec/data/sample_small/sample_small_creator_list.dict')
obj5 = utils.load_pickle('/home/vijai/tenet_comparison/SASRec/data/sample_small/sample_small.ListItems_len5_item5_cut1000.negativeEvalTest')


print('\n\n1 \n\n')
print(obj1)
print('\n\n1 \n\n')
print(obj2)
print('\n\n1 \n\n')
print(obj3[2])
print('\n\n1 \n\n')
print(obj4)
print('\n\n5 \n\n')
print(obj5)
pdb.set_trace()
print('\n\n1 \n\n')
