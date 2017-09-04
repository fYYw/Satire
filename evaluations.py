import numpy as np
import utilities as util


def eval_batch(fn, examples, word_dict, char_dict, args, char=True, sent_ling=True, doc_ling=True):
    a = b = c = d = 0
    for batch_x, _ in examples:
        if char and sent_ling and doc_ling:
            batch_x, batch_sent, batch_doc, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
            batch_doc = util.doc_ling_padding(list(batch_doc), args.max_ling)
            batch_y = np.array(list(batch_y))
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask, batch_sent, batch_doc)
        elif char and not sent_ling and doc_ling:
            batch_x, batch_doc, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_doc = util.doc_ling_padding(list(batch_doc), args.max_ling)
            batch_y = np.array(list(batch_y))
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask, batch_doc)
        elif char and sent_ling and not doc_ling:
            batch_x, batch_sent, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
            batch_y = np.array(list(batch_y))
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask, batch_sent)
        elif char and not sent_ling and not doc_ling:
            batch_x, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_y = np.array(list(batch_y))
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask)
        elif not char and not sent_ling and not doc_ling:
            batch_x, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, _ = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_y = np.array(list(batch_y))
            predict = fn(batch_rnn, batch_word_mask, batch_sent_mask)
        elif not char and sent_ling and doc_ling:
            batch_x, batch_sent, batch_doc, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, _ = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
            batch_doc = util.doc_ling_padding(list(batch_doc), args.max_ling)
            batch_y = np.array(list(batch_y))
            predict = fn(batch_rnn, batch_word_mask, batch_sent_mask, batch_sent, batch_doc)

        matrix = confusion_matrix(predict, batch_y)
        a += matrix[0]
        b += matrix[1]
        c += matrix[2]
        d += matrix[3]
    acc = 100.0 * (a + d) / (a + b + c + d)
    pre = 100.0 * a / (a + c)
    rec = 100.0 * a / (a + b)
    fsc = 2 * pre * rec / (pre + rec)
    return acc, pre, rec, fsc


def eval_vec_batch(fn, examples, char=True, sent_ling=True, doc_ling=True):
    a = b = c = d = 0
    if char and sent_ling and doc_ling:
        for batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn, batch_sent, batch_doc, batch_y, _ in examples:
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask, batch_sent, batch_doc)
            matrix = confusion_matrix(predict, batch_y)
            a += matrix[0]
            b += matrix[1]
            c += matrix[2]
            d += matrix[3]
    elif char and sent_ling and not doc_ling:
        for batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn, batch_sent, batch_y, _ in examples:
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask, batch_sent)
            matrix = confusion_matrix(predict, batch_y)
            a += matrix[0]
            b += matrix[1]
            c += matrix[2]
            d += matrix[3]
    elif char and not sent_ling and doc_ling:
        for batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn, batch_doc, batch_y, _ in examples:
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask, batch_doc)
            matrix = confusion_matrix(predict, batch_y)
            a += matrix[0]
            b += matrix[1]
            c += matrix[2]
            d += matrix[3]
    elif char and not sent_ling and not doc_ling:
        for batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn, batch_y, _ in examples:
            predict = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask)
            matrix = confusion_matrix(predict, batch_y)
            a += matrix[0]
            b += matrix[1]
            c += matrix[2]
            d += matrix[3]
    elif not char and not sent_ling and not doc_ling:
        for batch_rnn, batch_sent_mask, batch_word_mask, batch_y, _ in examples:
            predict = fn(batch_rnn, batch_word_mask, batch_sent_mask)
            matrix = confusion_matrix(predict, batch_y)
            a += matrix[0]
            b += matrix[1]
            c += matrix[2]
            d += matrix[3]
    elif not char and sent_ling and doc_ling:
        for batch_rnn, batch_sent_mask, batch_word_mask, batch_sent, batch_doc, batch_y, _ in examples:
            predict = fn(batch_rnn, batch_word_mask, batch_sent_mask, batch_sent, batch_doc)
            matrix = confusion_matrix(predict, batch_y)
            a += matrix[0]
            b += matrix[1]
            c += matrix[2]
            d += matrix[3]

    acc = 100.0 * (a + d) / (a + b + c + d)
    pre = 100.0 * a / (a + c)
    rec = 100.0 * a / (a + b)
    fsc = 2 * pre * rec / (pre + rec)
    return acc, pre, rec, fsc


def confusion_matrix(predict, y, mb=None):
    predict = predict > 0.5
    predict = predict.reshape(y.shape)
    a = np.sum(np.logical_and(y == 0, predict == 0))
    b = np.sum(np.logical_and(y == 0, predict == 1))
    c = np.sum(np.logical_and(y == 1, predict == 0))
    d = np.sum(np.logical_and(y == 1, predict == 1))
    if mb is not None:
        print(mb[predict == 1])
    return a, b, c, d


def get_attentions(fn, all_examples, word_dict, char_dict, args, examples_size):
    att_matrix = np.zeros((examples_size, args.max_sent))
    for batch_x, mb_idx in all_examples:
        batch_x, batch_sent, batch_doc, batch_y = zip(*batch_x)
        batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
        batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
            util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
        batch_sent = util.sent_ling_padding(list(batch_sent), args.max_sent, args.max_ling)
        att = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask, batch_sent)
        att_matrix[mb_idx, :] = att
    return att_matrix.tolist()


def get_sentences(fn, all_examples, word_dict, char_dict, args, examples_size):
    sentence_matrix = np.zeros((examples_size, args.max_sent, 160))
    for batch_x, mb_idx in all_examples:
        batch_x, batch_sent, batch_doc, batch_y = zip(*batch_x)
        batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
        batch_rnn, batch_sent_mask, batch_word_mask, batch_cnn = \
            util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
        sentence_array = fn(batch_rnn, batch_cnn, batch_word_mask, batch_sent_mask)
        sentence_matrix[mb_idx, :, :] = sentence_array
    return sentence_matrix
