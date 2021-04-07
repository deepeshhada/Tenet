import sys,math,argparse,os,pdb,random,datetime
import numpy as np
import pandas as pd
import torch
from time import time
from HNHN import preprocess_data_hnhn, hypergraph

sys.path.append('./.')
sys.path.append('./Utilities/.')
sys.path.append('./Models/.')
sys.path.append('./pygat/.')

#'''
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(7)
np.random.seed(20)
#random.seed(7)
torch.manual_seed(20)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#'''

from Arguments import parse_args
from TenetNegativeSamples import NegativeSamples
from ListNegativeSamples import ListNegativeSamples
from Dataset import Dataset
from TenetDataset import TenetDataset
from EmbedDataset import EmbedDataset
from Parameters import Parameters
from Valid_Test_Error import Valid_Test_Error
from Valid_Test_Error_seq import Valid_Test_Error_seq
from Evaluation import evaluate_model
from Error_plot import Error_plot
from Models import Models
from Batch import Batch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import torch.nn as nn
import utils
from sampler import WarpSampler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    args = parse_args()
    print(args)
    print("Main_Both2")
    print('Data loading...')
    t1, t_init = time(), time()
    args.device = device
    args.date_time = datetime.datetime.now()
    print(args.date_time)

    if args.method.lower() in ['tenet']:
        if args.knn_graph == 'True':
            dataset = EmbedDataset(args)  # change according to method
        else:
            dataset = TenetDataset(args)  # change according to method
    else:
        dataset = Dataset(args)

    params = Parameters(args,dataset)
    print("""Load data done [%.1f s]. #user:%d, #list:%d, #item:%d, #train:%d, #valid:%d, #test:%d""" %
          (time() - t1, params.num_user, params.num_list, params.num_item, params.num_train_instances,
           params.num_valid_instances, params.num_test_instances))

    args.args_str = params.get_args_to_string()
    t1 = time()
    print("args str: ", args.args_str)

    print("leng from list_items_list: ", len(utils.get_value_lists_as_list(params.list_items_dct)))
    print("leng from trainArrTriplets: ", len((params.trainArrTriplets[0])))
    print("non-zero entries in train_matrix: ", params.train_matrix.nnz)

    # process HNHN data
    hnhn_data_path = os.path.join("HNHN/processed", args.dataset)

    if not os.path.exists(hnhn_data_path) or len(os.listdir(hnhn_data_path)) == 0:
        print('preprocessing HNHN data dict...')
        preprocess_data_hnhn.preprocess()
    print('building HNHN hypergraph...')
    hnhn_args, hnhn, hnhn_train_set, hnhn_train_loader, hnhn_val_loader, hnhn_test_loader = hypergraph.build_hypergraph()

    # model-loss-optimizer defn =======================================================================
    models = Models(params, device=device)
    model = models.get_model()

    if params.loss not in ['bpr']: #bpr
        criterion_li     = torch.nn.BCELoss()
        #criterion_li     = torch.nn.BCEWithLogitsLoss() ## new change made
    if params.optimizer == 'adam':
        optimizer_gnn     = torch.optim.Adam(model.parameters(), lr=params.lr)
        optimizer_seq     = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif params.optimizer == 'rmsprop':
        optimizer_gnn     = torch.optim.RMSprop(model.parameters(), lr=params.lr)
        optimizer_seq     = torch.optim.RMSprop(model.parameters(), lr=params.lr)
    model.to(device)

    # training =======================================================================
    ## param =============================
    #=====================================
    vt_err_gnn           = Valid_Test_Error(params)
    vt_err_seq           = Valid_Test_Error_seq(params)

    include_networks     = eval(args.include_networks)
    if len(include_networks) == 1 and 'gnn' in include_networks:
        valid_type       = 'gnn'
    else:
        valid_type       = 'seq'
    vt_err               = vt_err_gnn if valid_type == 'gnn' else vt_err_seq
    error_plot           = Error_plot(save_flag=True,res_path=params.result_path,args_str=args.args_str,args=args,item_bundle_str='bundle')
    ns_gnn               = NegativeSamples(params.train_matrix,params.num_negatives,params)
    ns_seq               = ListNegativeSamples(params.train_matrix_item_seq, params.num_negatives,params)

    include_hgnn_flag = False
    for epoch_num in range(params.num_epochs+1):
        tt = time()
        model.train()
        for network in include_networks:
            t2 = time()
            ce_or_pairwise_loss, reg_loss, recon_loss = 0.0, 0.0, 0.0
            if network == 'gnn':
                if params.include_hgnn and epoch_num > params.warm_start_gnn:
                    print("including hgnn")
                    include_hgnn_flag = True
                user_input, list_input, item_input, train_rating = ns_gnn.generate_instances()
                user_input, list_input, item_input, train_rating = (
                    torch.from_numpy(user_input[params.num_train_instances:].astype(np.long)),
                    torch.from_numpy(list_input[params.num_train_instances:].astype(np.long)),
                    torch.from_numpy(item_input[params.num_train_instances:].astype(np.long)),
                    torch.from_numpy(train_rating.astype(np.float32))
                )

                user_input = user_input - 1
                item_input = item_input + params.num_user - 2
                list_input = list_input + params.num_user + params.num_item - 3
                hnhn_train_set.negatives = torch.stack((user_input, item_input, list_input), dim=1).long()

                num_inst = len(user_input)
            elif network == 'seq':
                list_input, item_seq, item_seq_pos, item_seq_neg                = ns_seq.generate_instances()
                user_input                                                      = params.list_user_vec[list_input]
                user_input, list_input, item_seq, item_seq_pos, item_seq_neg    =  (torch.from_numpy(user_input.astype(np.long)).to(device),
                                                                                    torch.from_numpy(list_input.astype(np.long)).to(device),
                                                                                    torch.from_numpy(item_seq.astype(np.long)).to(device),
                                                                                    torch.from_numpy(item_seq_pos.astype(np.long)).to(device),
                                                                                    torch.from_numpy(item_seq_neg.astype(np.long)).to(device))
                num_inst = len(list_input)
            # negative sampling end =======================================================================

            if network == 'gnn' and params.loss not in ['bpr']:
                batch = Batch(num_inst,params.batch_size,shuffle=True)
                while batch.has_next_batch():
                    batch_indices = batch.get_next_batch_indices()
                    optimizer_gnn.zero_grad()
                    hypertrain.train_one_epoch(hnhn_train_loader)

                    y_pred = model(
                        user_indices=user_input[batch_indices],
                        list_indices=list_input[batch_indices],
                        item_indices=item_input[batch_indices],
                        network=network,
                        include_hgnn=include_hgnn_flag
                    )
                    y_orig = train_rating[batch_indices]
                    loss = criterion_li(y_pred, y_orig)

                    loss.backward()
                    optimizer_gnn.step()
                    ce_or_pairwise_loss += loss * len(batch_indices)
            elif network == 'seq':
                batch = Batch(num_inst,params.batch_size_seq,shuffle=True)
                while batch.has_next_batch():
                    batch_indices = batch.get_next_batch_indices()
                    optimizer_seq.zero_grad()

                    y_pred_seq_pos, y_pred_seq_neg, is_target = model(
                        user_indices=user_input[batch_indices].long(),
                        list_indices=list_input[batch_indices].long(),
                        item_seq=item_seq[batch_indices].long(),
                        item_seq_pos=item_seq_pos[batch_indices].long(),
                        item_seq_neg=item_seq_neg[batch_indices].long(),
                        train=True,
                        network=network
                    )

                    first_flag = True

                    for ind_neg in range(params.num_negatives_seq - 1):
                        neg_indices = np.arange(0,num_inst)
                        np.random.shuffle(neg_indices)
                        neg_batch_indices = neg_indices[0:len(batch_indices)]

                        _, y_pred_seq_neg_arr_local, _ = model(
                            user_indices=user_input[batch_indices].long(),
                            list_indices=list_input[batch_indices].long(),
                            item_seq=item_seq[batch_indices].long(),
                            item_seq_pos=item_seq_pos[batch_indices].long(),
                            item_seq_neg=item_seq_neg[neg_batch_indices].long(),
                            train=True,
                            network=network
                        )  # neg_batch_indices

                        if first_flag:
                            first_flag = False
                            y_pred_seq_neg_sum = (1 - y_pred_seq_neg_arr_local + 1e-24).log() * is_target
                        else:
                            y_pred_seq_neg_sum += (1 - y_pred_seq_neg_arr_local + 1e-24).log() * is_target

                    if params.num_negatives_seq <= 1:
                        loss = (-(y_pred_seq_pos + 1e-24).log() * is_target - (1 - y_pred_seq_neg + 1e-24).log() * is_target).sum()/is_target.sum()
                    else:
                        loss = (-(y_pred_seq_pos + 1e-24).log() * is_target - (
                                1 - y_pred_seq_neg + 1e-24).log() * is_target - y_pred_seq_neg_sum).sum()/is_target.sum()

                    loss.backward()
                    optimizer_seq.step()
                    ce_or_pairwise_loss += loss
            # training end =======================================================================
            total_loss = ce_or_pairwise_loss + reg_loss + recon_loss
            print("""[%.2f s] %15s iter:%3i obj ==> total loss:%.4f ce/pairwise loss:%.4f reg loss:%.4f recon loss:%.4f """
                %(time()-t2,network, epoch_num,total_loss,ce_or_pairwise_loss,reg_loss,recon_loss))

        # validation and test =======================================================================
        if epoch_num > 0 and epoch_num % params.epoch_mod == 0: ##
            t3 = time()
            (valid_hits_lst,valid_ndcg_lst,valid_map_lst) = vt_err.get_update(model,epoch_num,device,valid_flag=True)
            (test_hits_lst,test_ndcg_lst,test_map_lst)    = vt_err.get_update(model,epoch_num,device,valid_flag=False)
            (valid_hr,valid_ndcg,valid_map)               = (np.mean(valid_hits_lst),np.mean(valid_ndcg_lst),np.mean(valid_map_lst))
            (test_hr,test_ndcg,test_map)                  = (np.mean(test_hits_lst),np.mean(test_ndcg_lst),np.mean(test_map_lst))
            print("[%.2f s] %15s Errors train %.4f valid hr: %.4f test hr: %.4f valid ndcg: %.4f test ndcg: %.4f valid map: %.4f test map: %.4f"%(time()-t3,'',ce_or_pairwise_loss/num_inst,valid_hr,test_hr,valid_ndcg,test_ndcg,valid_map,test_map))

            error_plot.append(loss,recon_loss,reg_loss,ce_or_pairwise_loss,valid_hr,test_hr,valid_ndcg,test_ndcg,valid_map,test_map)
            print('Time taken for this epoch: {:.2f} m'.format((time()-tt)/60))

    #=============================================================================================
    if args.store_embedding == 'True':
        print("store embeddings")
        user_list_item_embeddings_np = model.user_list_item_embeddings.weight.cpu().detach().numpy()
        utils.store_npy(args.path + args.dataset + '.user_embed.npy', user_list_item_embeddings_np[0:params.num_user])
        utils.store_npy(args.path + args.dataset + '.list_embed.npy', user_list_item_embeddings_np[params.num_user:params.num_user+params.num_list])
        utils.store_npy(args.path + args.dataset + '.item_embed.npy', user_list_item_embeddings_np[params.num_user+params.num_list:params.num_user+params.num_list+params.num_item])

    # best valid and test =======================================================================
    tot_time = time() - t_init
    args.total_time = '{:.2f}m'.format(tot_time/60)
    print('error plot: ')
    # (best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index,best_valid_hr,best_valid_ndcg,best_valid_map,best_test_hr,best_test_ndcg,best_test_map,best_test_hr_test,best_test_ndcg_test,best_test_map_test) = error_plot.get_best_valid_test_error()
    # args.hr_index,args.ndcg_index,args.map_index = best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index
    # print('[{:.2f} s] best_hr_index: {} best_ndcg_index: {} best_map_index: {} best_valid_hr: {:.4f} best_valid_ndcg: {:.4f} best_valid_map: {:.4f} best_test_hr: {:.4f} best_test_ndcg: {:.4f} best_test_map: {:.4f} best_test_hr_test: {:.4f} best_test_ndcg_test: {:.4f} best_test_map_test: {:.4f}'.format(tot_time,best_valid_hr_index,best_valid_ndcg_index,best_valid_map_index,best_valid_hr,best_valid_ndcg,best_valid_map,best_test_hr,best_test_ndcg,best_test_map,best_test_hr_test,best_test_ndcg_test,best_test_map_test))
    # error_plot.plot()
