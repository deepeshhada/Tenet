## command to run =========
# python main.py --path /home/vijai/tenet/data_tenet/required/aotm_small_small/sasrec/ --dataset aotm --train_dir default --at_k 5
# python main.py --path /home/vijai/tenet/data_tenet/required/zhihu_small_small/sasrec/ --dataset zhihu --train_dir default --at_k 5
# python main.py --path /home/vijai/tenet/data_tenet/aotm_small_small/sasrec/ --dataset aotm --train_dir default --at_k 5
# python main.py --path /home/vijai/tenet/data_tenet/sss_aotm/sasrec/ --dataset aotm --train_dir default --at_k 5
# a = sess.run([model.mask],{model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,model.is_training: True})
# a = np.array(sess.run([model.mask],{model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,model.is_training: True}))
# ========================

import os
import time
import argparse,pdb
import tensorflow as tf, pdb
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

## mine
sys.path.append('/home/vijai/tenet/Tenet/model_code/Utilities/.')
import utils

## mine - path


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=20, type=int) #50
parser.add_argument('--at_k', default=5, type=int)
parser.add_argument('--hidden_units', default=6, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()


dataset = data_partition(args.path + args.dataset)
dataset_path = args.path + args.dataset #'/home/vijai/tenet/data_tenet/aotm_small_small/sasrec/aotm'
[user_train, user_valid, user_test, usernum, itemnum] = dataset
## mine ==========
_, _, valid_negatives_dct, test_negatives_dct = utils.load_pickle(dataset_path+'.list_items.train_valid_test.pkl')
##_, valid_negatives_dct, test_negatives_dct = None,None,None #utils.load_pickle(dataset_path+'.list_items.train_valid_test.pkl')

#print(valid_negatives_dct,test_negatives_dct)
#pdb.set_trace()
#=================

num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print 'average sequence length: %.2f' % (cc / len(user_train))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

##try:
if True:
    for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            #pdb.set_trace()

            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})
            #pdb.set_trace()

        #if False: #epoch % 20 == 0 and epoch > 100: #%20
        if epoch % 10 == 0 and epoch > 1: #%20
            t1 = time.time() - t0
            T += t1
            print 'Evaluating',
            ## mine
            #pdb.set_trace()
            t_test  = evaluate(model, dataset, args, sess, test_negatives_dct,args.at_k)
            t_valid = evaluate_valid(model, dataset, args, sess, valid_negatives_dct,args.at_k)
            ## mine
            #t_valid = 0,0,0

            print ''
            print("AT_K: ", args.at_k)
            print 'epoch:%d, time: %f(s), valid (NDCG: %.4f, HR: %.4f, MAP: %.4f), test (NDCG: %.4f, HR: %.4f, MAP: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1], t_test[2])

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
##except:
else:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
