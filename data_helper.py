import pickle, os, random
import codecs
import numpy as np
# tag2l = {"O": 0, "S": 1}
tag2l = {'o': 0, 's': 1, 't': 2}
def read_corpus(corpus_path):
    sent_, tag_ = [], []
    with codecs.open(corpus_path, encoding='utf-8') as fr:
        for l in fr:
            item = l.strip().split('\t')
            sent_.append(item[0])
            tag_.append([it[0] for it in item[1].split()])
    return sent_, tag_
sent,tag = read_corpus('data/data4sentseg.txt')
# print(sent_[:10],tag_[:10])

def get_fasttext(vecfile):
    with codecs.open(vecfile,encoding='utf-8') as f:
        word2vec={}
        lines = f.readlines()
        vector_size = int(lines[0].strip().split()[1])
        for line in lines[1:]:
            item = line.strip().split()
            word = item[0]
            vec = np.array([float(it) for it in item[1:]])
            if len(vec) == vector_size:
                word2vec[word] = vec
    vocab = list(word2vec.keys())
    vocab.append('UNK')
    vocab2index = {}
    word_vector = []
    vector_size = len(word2vec[vocab[0]])
    for ind,w in enumerate(vocab[:-1]):
        vocab2index[w]=ind
        word_vector.append(word2vec[w])
    # print(np.array(word_vector).shape)
    vocab2index['UNK'] = len(vocab)-1
    vocab2index['PAD'] = len(vocab)
    vocab.append('PAD')
    word_vector.append(np.random.uniform(-1.0,1.0,vector_size))
    word_vector.append([0.1] * vector_size)
    print("finished loaded word2vec!!")
    return vocab, vocab2index, np.array(word_vector)

vocab,vocab2index,word_vector = get_fasttext('../w2v/fasttext_char_vec/fasttext_cbow_char.model.vec')

# with codecs.open('vocab.txt','w',encoding='utf-8') as f:
#     f.write(' '.join(vocab))

def sentence2id(sent, vocab2index):
    sentence_id = []
    for word in sent:
        try:
            sentence_id.append(vocab2index[word])
        except:
            sentence_id.append(vocab2index["UNK"])
    return sentence_id

def id2sentence(sent_id, vocab):
    sentence = []
    for id in sent_id:
        sentence.append(vocab[id])
    return ' '.join(sentence)

def pad_sequences(sequences, pad_mark=0):
    # max_len = max(map(lambda x : len(x), sequences))
    max_len = 40
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def tag2label(label):
    lb = [tag2l[l] for l in label]
    return lb

# for d in zip(sent[:10],tag[:10]):
#     sent_id = sentence2id(d[0], vocab2index)
#     print(d[0])
#     print(sent_id)
#     print(id2sentence(sent_id,vocab))

# sl=[]
# for s in sent:
#     sl.append(len(s))
# from collections import Counter
# sl_count = Counter(sl).items()
# _count = sorted(sl_count,key=lambda x:x[0], reverse=False)
# print(_count)
# import matplotlib.pyplot as plt
# x=[it[0] for it in _count]
# y=[it[1] for it in _count]
# plt.plot(x,y,'o')
# plt.show()
def batch_iter(data, batch_size, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]

