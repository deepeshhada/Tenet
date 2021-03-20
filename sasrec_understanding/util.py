import sys,pdb
import copy
import random
import numpy as np
from collections import defaultdict

## map should be corrected
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname + '.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess, negatives_dct,at_k): ##mine negatives_dct added
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    MAP = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1 or (u,test[u][0]) not in negatives_dct: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        #pdb.set_trace()
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]

        '''
        for _ in range(100): ## 100 negatives
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        '''

        ## mine
        item_idx   += negatives_dct[(u,test[u][0])]

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        #if rank < 10: ##at_k
        if rank < at_k: ##at_k
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            MAP += 1.0/ (rank + 1)
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, MAP / valid_user


def evaluate_valid(model, dataset, args, sess, negatives_dct,at_k): ## negatives_dct
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    MAP = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1 or (u,valid[u][0]) not in negatives_dct: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        '''
        for _ in range(100): ## 100 negatives
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        '''

        ## mine
        item_idx   += negatives_dct[(u,valid[u][0])]

        predictions = -model.predict(sess, [u], [seq], item_idx) ##same idx should be used
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < at_k: #10: ##at_k
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            MAP += 1.0/ (rank + 1)
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, MAP / valid_user
