from modules import *
import pdb

class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        ## mine 1
        self.mask = mask

        # 1 complete ===========
        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.item_emb_table = item_emb_table ##mine
            ## mine 2
            self.seq1 = self.seq

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            ## mine
            self.t = t
            self.seq2 = self.seq
            self.pos_emb_table = pos_emb_table

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            ## mine 3
            self.before_mask = self.seq
            #pdb.set_trace()
            self.seq *= mask
            ## mine 4
            self.after_mask = self.seq ## to mask-out the added position embeddings at the padded places

            # 2 complete ===================================

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq), ## normilizing query is extra, layer-norm
                                                   keys=self.seq, ## query --> norm(seq), key --> seq, val --> seq
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True, ##v.v.imp ## this is set true to maks-out the future positions (tokens), language model part
                                                   scope="self_attention")
                    self.seq_attn = self.seq

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training) ##after ffn no-normalization is done for seq, if multiple-blocks, only query is normalized
                    self.seq_ffn = self.seq
                    self.seq *= mask
                    self.seq_attn_ffn_masked = self.seq

            ## mine 5
            self.seq_before_norm = self.seq
            self.seq = normalize(self.seq)
            self.seq_after_norm = self.seq

        # 3 complete ===================================
        #pdb.set_trace()

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
        self.pos_reshaped = pos
        self.seq_emb = seq_emb
        self.pos_emb = pos_emb
        self.neg_emb = neg_emb
        # 4 complete ===================================
        # ======

        self.test_item = tf.placeholder(tf.int32, shape=(101)) ## note all 101 items have same seq embeddings
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_item_emb = test_item_emb
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        self.test_logits = self.test_logits[:, -1, :]
        #pdb.set_trace()

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1) ##item_embedding at the final layer with the right item (list_item could be multiplied and added with the target item)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.istarget = istarget
        # 5 complete ===================================


        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)  ## In my opinion, going to triplet loss or bpr-loss is better.
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        pdb.set_trace()
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
