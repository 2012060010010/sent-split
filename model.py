import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data_helper import batch_iter
from data_helper import id2sentence
from data_helper import pad_sequences
import copy
class BiLSTM_CRF(object):
    def __init__(self, batch_size, epoch_num, hidden_dim, embeddings,
                 dropout_keep, optimizer, lr, clip_grad,
                 tag2label, shuffle,vocab,
                 model_path, summary_path, update_embedding=False):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.dropout_keep_prob = dropout_keep
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.shuffle = shuffle
        self.vocab = vocab
        self.model_path = model_path
        self.summary_path = summary_path
        self.update_embedding = update_embedding

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.embedding_placeholder = tf.placeholder(tf.float32, [None, None], name="embedding_matrix")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("char_embed"):
            self.W = tf.Variable(tf.constant(0.0, shape=self.embeddings.shape),
                                 trainable=self.update_embedding, name="WW")
            self.embedding_init = self.W.assign(self.embedding_placeholder)

            self.word_embeddings = tf.nn.embedding_lookup(params=self.W,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        # self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi_lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("FC"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                               tag_indices=self.labels,
                                                               sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)
        tf.summary.scalar("loss", self.loss)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()
    def embed_init(self,sess):
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.embeddings})
    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=4)

        with tf.Session() as sess:
            sess.run(self.init_op)
            self.embed_init(sess)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def pred_test(self, test):
        data_split = []
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)
            for ind,test_tmp in enumerate(test):
                try:
                    label_list, seq_len_list = self.dev_one_epoch(sess, test_tmp)
                except:
                    print(ind)
                after_split = self.split_sent(label_list, seq_len_list, test_tmp)
                data_split.append(after_split)
        return data_split

    def split_sent(self, label_list,seq_len_list, data):
        sent_list = []
        for label_, (sent, tag) in zip(label_list, data):
            split_ind = []
            for i, p in enumerate(label_):
                if p:
                    split_ind.append(i)
            resent = id2sentence(sent, self.vocab).split()
            out_sent = copy.deepcopy(resent)
            for ind in range(len(resent)):
                if ind in split_ind:
                    out_sent[ind] = resent[ind] + '。'
                else:
                    out_sent[ind] = resent[ind]
            sent_list.append(''.join(out_sent))
        return ' '.join(sent_list)

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):

        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_iter(train, self.batch_size, shuffle=self.shuffle)
        step = -1
        for batch in batches:
            step+=1
            seqs, labels = zip(*batch)
            if step % 20 == 0:
                print(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)  #
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                print('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(
                    start_time, epoch + 1, step + 1,
                    loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for batch in batch_iter(dev, self.batch_size, shuffle=False):
            seqs, labels  = zip(*batch)
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)
        logits, transition_params = sess.run([self.logits, self.transition_params],
                                             feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        def get_f1(pred, truth):
            p_hit = 0
            p_not = 0
            n_hit = 0
            n_not = 0
            acc1 = 0
            acc2 = 0
            for it in zip(pred, truth):
                if it[0] == it[1] and sum(it[1]):
                    p_hit += 1
                if it[0] != it[1] and sum(it[1]):
                    p_not += 1
                if it[0] != it[1] and not sum(it[1]):
                    n_not += 1
                if it[0] == it[1] and not sum(it[1]):
                    n_hit += 1
                if it[0] == it[1]:
                    acc1 += 1
                cnt = 0
                for i in range(len(it[0])):
                    if it[0][i] == it[1][i]:
                        cnt +=1
                acc2 += (cnt/len(it[0]))
            P = p_hit / (p_hit + p_not)
            N = n_hit / (n_hit + n_not)
            F1 = P * N * 2 / (P + N +0.00001)
            ACC1 = acc1 / len(pred)
            ACC2 = acc2 / len(pred)
            return P, N, F1, ACC1, ACC2
        true_label = []
        pred_label = []
        sent_list = []
        for label_, (sent, tag) in zip(label_list, data):
            p_ind = []
            t_ind = []
            if sum(label_) == 0 and sum(tag) == 0:
                pass
            else:
                for i, p in enumerate(label_):
                    if p:
                        p_ind.append(i)
                for i, p in enumerate(tag):
                    if p:
                        t_ind.append(i)
                sent_list.append(id2sentence(sent, self.vocab)+'\n'+str(p_ind)+'VS'+str(t_ind))
            true_label.append(tag)
            pred_label.append(label_)
        import codecs
        with codecs.open('result.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(sent_list))
        print(get_f1(pred_label, true_label))

