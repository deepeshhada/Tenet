import subprocess
import numpy as np
import subprocess

def run_python(test,lr,batch_size,num_factors,num_negatives,margin,keep_prob,gnn_keep_prob,net_keep_prob,hid_units,d_k,n_heads,knn_k, num_negatives_seq):
    sys_path         = '/home/vijai/'
    #data_path        = 'tenet/data_tenet/required/spotify_small/tenet/spotify/' #+ test + '/fold1/'
    data_path        = 'tenet/data_tenet/required/zhihu_small/tenet/zhihu/' #+ test + '/fold1/'
    #data_path        = 'tenet/data_tenet/sample_small/' #+ test + '/fold1/'
    #method           = 'tenet'
    method           = 'tenet'
    #method           = 'seq'

    dataset          = 'zhihu'
    #dataset          = 'sample_small'
    res_folder       = 'gnn_seq_training/' # 'training_negative' ,'training_knn'
    comment          = 'training gnn_seq networks combinedly.'

    # ===========================================
    include_networks = "['gnn','seq']" #"['gnn','seq']"
    num_epochs       = '300' ##400
    num_layers       = '2'
    batch_size_seq   = '256'
    valid_batch_siz  = '128'
    at_k             = '5'
    embed_type       = 'node2vec'
    warm_start_seq   = '1'
    #knn_k             = '20'
    #d_k              = '64'

    # ===========================================
    path             = sys_path + data_path
    res_path         = sys_path + '/result/'

    print("path: ",path,"res_path: ",res_path)
    print("test,lr,batch_size,num_factors,num_negatives,margin,keep_prob,gnn_keep_prob,net_keep_prob,hid_units,n_heads: ",
           test,lr,batch_size,num_factors,num_negatives,margin,keep_prob,gnn_keep_prob,net_keep_prob,hid_units,n_heads)
    print("\n")
    #'''
    #subprocess.call(['python','seq_mine_Main.py', ## only-seq (can be done using Main_Both2)
    #subprocess.call(['python','Main_Both.py', ## old-version
    subprocess.call(['python','Main_Both2.py', ## ##new-version
    #subprocess.call(['python','Main.py', ## old-version
                        '--method',method,
                        '--path',path,
                        '--dataset',dataset,
                        '--res_path',res_path,
                        '--res_folder',res_folder,
                        '--include_networks',include_networks,

                        '--num_epochs', num_epochs,
                        '--batch_size', batch_size,
                        '--batch_size_seq', batch_size_seq,
                        '--valid_batch_siz', valid_batch_siz,
                        '--lr', lr,
                        '--optimizer', 'adam',
                        '--loss', 'ce',
                        '--initializer', 'xavier',
                        '--stddev','0.002',
                        '--max_item_seq_length','200',
                        '--load_embedding_flag','0',
                        '--at_k', at_k,
                        '--knn_k', knn_k,
                        '--cosine', 'False',
                        '--embed_type', embed_type,

                        '--num_factors', num_factors,
                        '--num_negatives', num_negatives,
                        '--num_negatives_seq', num_negatives_seq,
                        '--reg_w', '0',
                        '--reg_b', '0',
                        '--reg_lambda', '0',
                        '--margin', '2.0',
                        '--keep_prob', keep_prob,

                        '--num_layers', '2',
                        '--hid_units', hid_units,
                        '--gnn_keep_prob', gnn_keep_prob,
                        '--net_keep_prob', net_keep_prob,

                        '--n_heads',n_heads,
                        '--d_k',d_k,

                        '--dataset_avg_flag_zero','0',
                        '--epoch_mod','1',
                        '--num_thread','16',
                        '--comment',comment,

                        '--store_embedding','False',
                        '--knn_graph','True',
                        '--user_adj_weights','False',
                        '--self_loop','True',

                        '--warm_start_gnn', warm_start_seq,
                    ])
    #'''
    print('ONE ITERATION FINISHED.\n\n')

if __name__ == '__main__':
    test_list           = ['1']
    lr_list             = ['0.003']
    batch_size_list     = ['2048']
    num_fact_list       = ['80'] #['16','32','64','80','128']
    #num_fact_list       = ['16','32','64','80','128']
    #num_negatives_list  = ['5','3','7' ,'6','2']
    num_negatives_list  = ['2']
    margin_list         = ['2.0']
    keep_prob_list      = ['0.5']
    gnn_keep_prob_list  = ['0.8']
    #gnn_keep_prob_list  = ['1.0', '0.8']
    net_keep_prob_list  = ['1.0']
    #n_heads_list        = ['[6,4]','[4,4]','[4,2]']
    hid_units_list      = ['[48,32]'] ##
    #hid_units_list      = ['[32,16]','[32,32]','[64,32]']
    d_k_list            = ['64']
    n_heads_list        = ['[4,2]']
    knn_k_list          = ['80']  #['5','10','15','25','30','40','50']
    #knn_k_list          = ['5','10','15','20','25','30','40','50','60','80']
    num_negatives_seq_list = ['2']

    for test in test_list:
        for lr in lr_list: #1
            for batch_size in batch_size_list: #2
                for num_factors in num_fact_list: #3
                    for num_negatives in num_negatives_list: #4
                        for margin in margin_list: #5
                            for keep_prob in keep_prob_list: #6
                                for gnn_keep_prob in gnn_keep_prob_list: #7
                                    for net_keep_prob in net_keep_prob_list: #8
                                        for hid_units in hid_units_list: #9
                                            for d_k in d_k_list: #9
                                                for n_heads in n_heads_list: #10
                                                    for knn_k in knn_k_list: #10
                                                        for num_negatives_seq in num_negatives_seq_list: #4
                                                            run_python(test=test,
                                                               lr=lr,
                                                               batch_size=batch_size,
                                                               num_factors=num_factors,
                                                               num_negatives=num_negatives,
                                                               margin=margin,
                                                               keep_prob=keep_prob,
                                                               gnn_keep_prob=gnn_keep_prob,
                                                               net_keep_prob=net_keep_prob,
                                                               hid_units=hid_units,
                                                               d_k=d_k,
                                                               n_heads=n_heads,
                                                               knn_k=knn_k,
                                                               num_negatives_seq=num_negatives_seq)
