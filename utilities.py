import random
import pickle
import gzip
import logging
import numpy as np
import theano
import lasagne
from collections import Counter


def merge_list(l):
    docs = []
    for ds in l:
        docs += ds
    return docs


def build_dict(docs, max_words=500000, dict_file=None):
    """
        :param docs: a doc is a list of sentences
        :return: dictionary of words
        """

    def _dict_loader(dict_file):
        with open(dict_file) as f:
            dicts = f.readlines()
        return {d.strip(): index + 2 for (index, d) in enumerate(dicts)}

    if dict_file is not None:
        if len(dict_file) == 2:
            return _dict_loader(dict_file[0]), _dict_loader(dict_file[1])
        if len(dict_file) == 1:
            return _dict_loader(dict_file[0])

    word_count = Counter()
    char_count = Counter()
    for doc in docs:
        for sent in doc:
            for w in sent.split(' '):
                if w == "":
                    continue
                elif w.isdigit():
                    w = "num@#!123"
                word_count[w.lower()] += 1
                for c in w:
                    char_count[c] += 1

    ls = word_count.most_common(max_words)
    chars = char_count.most_common(80)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    # leave 0 to padding
    # leave 1 to UNK
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}, \
           {c[0]: index + 2 for (index, c) in enumerate(chars)}


def sent_ling_padding(vec_docs_batch, max_sent, feature_length):
    num_sample = len(vec_docs_batch)
    linguistic_x = np.zeros((num_sample, max_sent, feature_length)).astype(theano.config.floatX)
    for id_d, v_d in enumerate(vec_docs_batch):
        for id_s, v_s in enumerate(v_d[:max_sent]):
            linguistic_x[id_d, id_s, :] = [float(v) for v in v_s.split()]
    return linguistic_x


def doc_ling_padding(vec_docs_batch, feature_length):
    num_sample = len(vec_docs_batch)
    linguistic_x = np.zeros((num_sample, feature_length)).astype(theano.config.floatX)
    for id_d, v_x in enumerate(vec_docs_batch):
        linguistic_x[id_d, :] = [float(v) for v in v_x.split()]
    return linguistic_x


def vectorization(docs, word_dict, char_dict=None, max_char_length=30):
    """
    :param docs:
    :param word_dict:
    :param char_dict:
    :param max_char_length:
    :return:
    """
    vec_docs = []
    vec_doc = []
    vec_sent = []
    char_word = []
    for doc in docs:
        for sent in doc:
            word_seq = []
            for w in sent.strip().split():
                if char_dict is not None:
                    word = w[:max_char_length]
                    vec_word = [char_dict[c] if c in char_dict else 1 for c in word]
                    char_word.append(vec_word)

                w = w.lower()
                if w.isdigit():
                    word_seq.append(word_dict["num@#!123"])
                elif w in word_dict:
                    word_seq.append(word_dict[w])
                elif w != "":
                    word_seq.append(1)

            vec_sent.append(word_seq)
            vec_sent.append(char_word)
            vec_doc.append(vec_sent)
            vec_sent = []
            char_word = []
        vec_docs.append(vec_doc)
        vec_doc = []
    return vec_docs


def mask_padding(vec_docs_batch, max_sent=36, max_word=160, max_char_length=30):
    """
    :param vec_docs_batch: a batch of vectorized documents
    :param max_sent
    :param max_word
    :param max_char_length
    """
    num_sample = len(vec_docs_batch)
    # max_sent, max_word = find_max_length(vec_docs_batch)
    rnn_x = np.zeros((num_sample, max_sent, max_word)).astype('int32')
    cnn_x = np.zeros((num_sample, max_sent, max_word, max_char_length)).astype('int32')
    word_mask = np.zeros((num_sample, max_sent, max_word)).astype(theano.config.floatX)
    sent_mask = np.zeros((num_sample, max_sent)).astype(theano.config.floatX)
    for id_d, v_d in enumerate(vec_docs_batch):
        sent_mask[id_d, :len(v_d)] = 1.0
        for id_s, v_s in enumerate(v_d[:max_sent]):
            word_mask[id_d, id_s, :len(v_s[0])] = 1.0
            rnn_x[id_d, id_s, :len(v_s[0])] = v_s[0][:max_word]
            for id_w, v_w in enumerate(v_s[1][:max_word]):
                cnn_x[id_d, id_s, id_w, :len(v_w)] = v_w[:max_char_length]
    return rnn_x, sent_mask, word_mask, cnn_x


def words2embedding(word_dict, dim, in_file=None, init=lasagne.init.Uniform()):
    num_words = max(word_dict.values()) + 1
    embeddings = init((num_words, dim))
    logging.info("Embedding dimension: %d * %d" % (num_words, dim))

    if in_file is not None:
        logging.info("loading embedding file: %s" % in_file)
        pre_trained = 0
        with open(in_file, encoding='utf') as f:
            l = f.readlines()
        for line in l:
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        logging.info("pre-trained #: %d (%.2f%%)" % (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def char2embedding(char_dict, dim, init=lasagne.init.Uniform()):
    num_char = max(char_dict.values()) + 1
    embeddings = init((num_char, dim))
    logging.info("Embedding dimension: %d * %d" % (num_char, dim))
    return embeddings


def zip_shuffle(list1, list2):
    if len(list1) != len(list2):
        print(len(list1), len(list2))
        raise ValueError('the length of 2 lists not match')
    zipped = list(zip(list1, list2))
    random.shuffle(zipped)
    list1, list2 = zip(*zipped)
    return list(list1), list(list2)


def label2vec(labels):
    dim = len(set(labels))
    dim = 2 if dim == 1 else dim
    vec_labels = np.zeros((len(labels), dim))
    for i in range(len(labels)):
        vec_labels[i, labels[i]] = 1
    return vec_labels


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic


def numpy_save(network, model):
    np.savez(model, *lasagne.layers.get_all_param_values(network))


def numpy_load(network, model):
    with np.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


def feature_vectorization(feature_batch):
    docs = []
    for doc in feature_batch:
        f = []
        for s in doc:
            f.append(list(map(float, s.split())))
        docs.append(f)
    return docs


def linguistic_padding(vec_docs_batch, max_sent, feature_length):
    num_sample = len(vec_docs_batch)
    linguistic_x = np.zeros((num_sample, max_sent, feature_length)).astype(theano.config.floatX)
    for id_d, v_d in enumerate(vec_docs_batch):
        for id_s, v_s in enumerate(v_d[:max_sent]):
            linguistic_x[id_d, id_s, :] = v_s
    return linguistic_x


if __name__ == '__main__':
    docs = [["This is the test sentence, to Test tHe order of words and chars !??"]]
    w, c = build_dict(docs)
    print(w)
    print(c)
