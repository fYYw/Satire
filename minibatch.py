import numpy as np
import random
import utilities as util


def doc_minibatch(docs, minibatch_size, shuffle=True):
    examples = []
    if shuffle:
        random.shuffle(docs)
    doc_length = len(docs)
    id_list = np.arange(0, doc_length, minibatch_size)
    if shuffle:
        np.random.shuffle(id_list)
    mbs = [np.arange(id, min(id + minibatch_size, doc_length)) for id in id_list]
    for mb in mbs:
        batch_x = [docs[i] for i in mb]
        examples.append((batch_x, mb))
    return examples


def train_doc_minibatch(fake, true, args, shuffle=True, over_sample=True):
    examples = []
    if not over_sample:
        for batch in true:
            examples += batch
        return doc_minibatch(fake + examples, args.batch_size, shuffle)
    else:
        for batch in true:
            examples += doc_minibatch(fake + batch, args.batch_size, shuffle)
        return examples


def vec_minibatch(docs, word_dict, char_dict, args, shuffle=True, char=True, sent_ling=True, doc_ling=True):
    examples = []
    if shuffle:
        random.shuffle(docs)
    doc_length = len(docs)
    id_list = np.arange(0, doc_length, args.batch_size)
    if shuffle:
        np.random.shuffle(id_list)
    mbs = [np.arange(id, min(id + args.batch_size, doc_length)) for id in id_list]
    for mb in mbs:
        batch_x = [docs[i] for i in mb]
        if char and sent_ling and doc_ling:
            batch_x, batch_sent, batch_doc, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
            batch_doc = util.doc_ling_padding(list(batch_doc), args.max_ling)
            batch_y = np.array(list(batch_y))
            examples.append((batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn,
                             batch_sent, batch_doc, batch_y, mb))
        elif char and sent_ling and not doc_ling:
            batch_x, batch_sent, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
            batch_y = np.array(list(batch_y))
            examples.append((batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn,
                             batch_sent, batch_y, mb))
        elif char and not sent_ling and doc_ling:
            batch_x, batch_doc, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_doc = util.doc_ling_padding(list(batch_doc), args.max_ling)
            batch_y = np.array(list(batch_y))
            examples.append((batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn,
                             batch_doc, batch_y, mb))
        elif char and not sent_ling and not doc_ling:
            batch_x, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_y = np.array(list(batch_y))
            examples.append((batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn, batch_y, mb))
        elif not char and not sent_ling and not doc_ling:
            batch_x, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, _ = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_y = np.array(list(batch_y))
            examples.append((batch_rnn, batch_sent_mask, batch_word_mask, batch_y, mb))
        elif not char and sent_ling and doc_ling:
            batch_x, batch_sent, batch_doc, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, _ = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
            batch_doc = util.doc_ling_padding(list(batch_doc), args.max_ling)
            batch_y = np.array(list(batch_y))
            examples.append((batch_rnn, batch_sent_mask, batch_word_mask, batch_sent, batch_doc, batch_y, mb))
    return examples
