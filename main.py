import tensorflow as tf
from model import BiLSTM_CRF
import numpy as np
import pandas as pd
import re
import os, argparse, time, random
from data_helper import read_corpus, get_fasttext, tag2l, pad_sequences, sentence2id, tag2label

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese labeling task')
parser.add_argument('--test_data', type=str, default='../BDCI2017-taiyi/train_test_ql.pkl', help='test data source')
parser.add_argument('--train_data', type=str, default='data/data4bietag_char.txt', help='train data source')
parser.add_argument('--batch_size', type=int, default=64, help='sample of each minibatch')
parser.add_argument('--epoch', type=int, default=7, help='epoch of training')
parser.add_argument('--hidden_dim', type=int, default=200, help='dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=3.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.8, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=bool, default=False, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='pretrain', help='use pretrained char embedding or init it randomly')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--test_model', type=str, default='./runs/Sun_Dec__3_12_45_07_2017/', help='model for test')
parser.add_argument('--char_vec_path', type=str, default='../w2v/fasttext_char_vec/fasttext_cbow_char.model.vec', help='file for word vec in fasttext')
args = parser.parse_args()

## get char embeddings
vocab, vocab2index, embeddings = get_fasttext(args.char_vec_path)
## read corpus and get training data

# training model
if args.mode == 'train':
    dev_percent = 0.1
    sent, tag = read_corpus(args.train_data)
    sent_ = [sentence2id(s, vocab2index) for s in sent]
    tag_ = [tag2label(l) for l in tag]
    data_ = [it for it in zip(sent_, tag_)]
    data_num = len(data_)
    dev_ind = - int(data_num * dev_percent)
    train_data = data_[:dev_ind]
    test_data = data_[dev_ind:]
    test_size = len(test_data)

    ## paths setting
    timestamp = time.asctime().replace(' ', '_').replace(':','_')
    output_path = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(output_path): os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2l, shuffle=args.shuffle,vocab=vocab,
                       model_path=ckpt_prefix, summary_path=summary_path,
                       update_embedding=args.update_embedding)
    model.build_graph()
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(len(test_data)))
    model.train(train_data, test_data)

## testing model
elif args.mode == 'test':
    import pickle
    with open(args.test_data,'rb') as f:
        train_df = pickle.load(f)
        test_df = pickle.load(f)
    print(test_df.columns)
    def make_test_data():
        data4sentseg = []
        for i in range(test_df.shape[0]):
        # for i in range(1000, 2000):
            sent_tmp = test_df.loc[i, 'sub_sents_tokenized']
            single_sent = []
            for sent in sent_tmp:
                sent_id = sentence2id(sent, vocab2index)
                label = ['o'] * len(sent)
                single_sent.append((sent_id, label))
            data4sentseg.append(single_sent)
        return data4sentseg
    data4sentseg = make_test_data()
    model_path = args.test_model+'checkpoints'
    summary_path = args.test_model+'summaries'
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)

    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=1.0, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2l, shuffle=args.shuffle,vocab=vocab,
                       model_path=ckpt_file, summary_path=summary_path,
                       update_embedding=args.update_embedding)
    model.build_graph()
    print("test data: {}".format(len(data4sentseg)))
    data_split = model.pred_test(data4sentseg)
    assert len(data_split) == test_df.shape[0]
    test_df['content_split'] = data_split
    test_df.to_csv('tt.csv', index=None, encoding='utf-8-sig')

