import numpy as np
import pandas as pd
import torch,pdb,math
from Evaluation import evaluate_model
from time import time
from Batch import Batch

class Valid_Test_Error(object):
    def __init__(self,params):
        self.params               = params
        self.validNegativesDict   = params.validNegativesDict
        self.testNegativesDict    = params.testNegativesDict

        self.num_valid_instances  = params.num_valid_instances
        self.num_test_instances   = params.num_test_instances
        self.num_thread           = params.num_thread
        self.num_valid_negatives  = self.get_num_valid_negative_samples(self.validNegativesDict)
        self.valid_dim            = self.num_valid_negatives + 1

        self.epoch_mod            = params.epoch_mod
        self.valid_batch_siz      = params.valid_batch_siz
        self.at_k                 = params.at_k

        self.validArrDubles,self.valid_pos_items = self.get_dict_to_dubles(self.validNegativesDict)
        self.testArrDubles,self.test_pos_items   = self.get_dict_to_dubles(self.testNegativesDict)
        self.list_user_vec                       = params.list_user_vec

    def get_num_valid_negative_samples(self,validDict): ## some strange things are happening.
        for key in validDict:
            return len(self.validNegativesDict[key])
        return None

    def get_dict_to_dubles(self,dct):
        list_lst, item_lst = [],[]
        pos_item_lst = []
        for key,value in dct.items():
            lst_id, itm_id = key
            lists  = list(np.full(self.valid_dim,lst_id,dtype = 'int32'))#+1 to add pos item
            items  = [itm_id]
            pos_item_lst.append(itm_id)
            items += list(value) # first is positive item

            list_lst   += lists
            item_lst   += items

        return (np.array(list_lst),np.array(item_lst)),np.array(pos_item_lst)

    def get_update(self,model,epoch_num,device,valid_flag=True):
        model.eval()
        if valid_flag == True:
            (list_input,item_input) = self.validArrDubles
            num_inst   = self.num_valid_instances * self.valid_dim
            posItemlst = self.valid_pos_items # parameter for evaluate_model
            matShape   = (self.num_valid_instances, self.valid_dim)
        else:
            (list_input,item_input) = self.testArrDubles
            num_inst   = self.num_test_instances * self.valid_dim
            posItemlst = self.test_pos_items # parameter for evaluate_model
            matShape   = (self.num_test_instances, self.valid_dim)

        batch_siz      = self.valid_batch_siz * self.valid_dim
        #print("jello")
        #print(batch_siz, self.valid_batch_siz, self.valid_dim,num_inst)
        #pdb.set_trace()

        full_pred_torch_lst  = []
        list_input_ten       = torch.from_numpy(list_input.astype(np.long)).to(device) ## could be moved to gpu before-hand
        item_input_ten       = torch.from_numpy(item_input.astype(np.long)).to(device)
        user_input           = self.list_user_vec[list_input]
        user_input_ten       = torch.from_numpy(user_input.astype(np.long)).to(device)
        batch                = Batch(num_inst,batch_siz,shuffle=False)
        ##
        user_input_ten = user_input_ten - 1
        item_input_ten = item_input_ten + self.params.num_user - 2
        list_input_ten = list_input_ten + self.params.num_user + self.params.num_item - 3

        dataset = torch.stack((user_input_ten, item_input_ten, list_input_ten), dim=1).numpy()
        df = pd.DataFrame(data=dataset)
        if valid_flag:
            df.to_csv("valid_set.csv", sep="\t", index=False, header=False)
        else:
            df.to_csv("test_set.csv", sep="\t", index=False, header=False)
        return None, None, None

        ind = 0
        while batch.has_next_batch():
            batch_indices    = batch.get_next_batch_indices()
            if self.params.method == 'bpr' or self.params.loss == 'pairwise':
                user_neg_input = None
                y_pred           = model(user_input_ten[batch_indices],user_neg_input, list_input_ten[batch_indices],item_input_ten[batch_indices]) # first argument for user
            else:
                y_pred           = model(user_indices=user_input_ten[batch_indices],list_indices=list_input_ten[batch_indices],item_indices=item_input_ten[batch_indices]) # first argument for user
            full_pred_torch_lst.append(y_pred.detach().cpu().numpy())
            #pdb.set_trace()
            #print(len(full_pred_torch_lst))

        #full_pred_np         = torch.cat(full_pred_torch_lst).data.cpu().numpy()
        full_pred_np         = np.concatenate(full_pred_torch_lst) #.data.cpu().numpy()
        # ==============================

        predMatrix           = np.array(full_pred_np).reshape(matShape)
        itemMatrix           = np.array(item_input).reshape(matShape)
        '''
        print('predMatrix')
        print(predMatrix[0:20,0:20])
        print('itemMatrix')
        print(itemMatrix[0:20,0:20])
        '''

        (hits, ndcgs, maps)  = evaluate_model(posItemlst=posItemlst,itemMatrix=itemMatrix,predMatrix=predMatrix,k=self.at_k,num_thread=self.num_thread)
        return (hits, ndcgs, maps)
