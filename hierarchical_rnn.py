import evaluations as evals
import data_utilities as du
import utilities as util
import minibatch as mb
import time
import sys
import numpy as np
import lasagne
import theano.tensor as T
import theano
import logging
import arg_parser as ap
import multi_channle_fn as fn


def main(args):
    logging.info("loading data...")
    fake_train, fake_dev, fake_test = du.load_fake(doc_ling=False, sent_ling=False)
    true_train, true_dev, true_test = du.load_true(doc_ling=False, sent_ling=False)
    if args.debug:
        true_train = true_train[0][:100]
        fake_train = fake_train[:10]
        true_dev = true_dev[:100]
        fake_dev = fake_dev[:10]
        true_test = true_test[:100]
        fake_test = fake_test[:10]
    if args.rnn_type == 'gru':
        args.rnn = lasagne.layers.GRULayer
    elif args.rnn_type == 'lstm':
        args.rnn = lasagne.layers.LSTMLayer
    else:
        args.rnn = lasagne.layers.RecurrentLayer

    logging.info("building dictionary...")
    word_dict, char_dict = util.build_dict(None, max_words=0, dict_file=["word_dict", "char_dict"])
    logging.info("creating embedding matrix...")
    word_embed = util.words2embedding(word_dict, 100, args.embedding_file)
    char_embed = util.char2embedding(char_dict, 30)
    (args.word_vocab_size, args.word_embed_size) = word_embed.shape
    (args.char_vocab_size, args.char_embed_size) = char_embed.shape
    logging.info("compiling Theano function...")
    att_fn, eval_fn, train_fn, params = create_theano_function(word_embed, char_embed, values=None)
    logging.info("batching examples...")
    dev_examples = mb.vec_minibatch(fake_dev + true_dev, word_dict, char_dict, args, False,
                                    char=False, sent_ling=False, doc_ling=False)
    test_examples = mb.vec_minibatch(fake_test + true_test, word_dict, char_dict, args, False,
                                     char=False, sent_ling=False, doc_ling=False)

    temp = []
    for true_batch in true_train:
        temp += true_batch
    true_train = temp
    del temp
    train_examples = mb.doc_minibatch(fake_train + true_train, args.batch_size)

    # train_examples = mb.train_doc_minibatch(fake_train, true_train, args)
    logging.info("checking network...")
    dev_acc = evals.eval_vec_batch(eval_fn, dev_examples, char=False, sent_ling=False, doc_ling=False)
    print('Dev A: %.2f P:%.2f R:%.2f F:%.2f' % dev_acc)
    test_acc = evals.eval_vec_batch(eval_fn, test_examples, char=False, sent_ling=False, doc_ling=False)
    print('Performance on Test set: A: %.2f P:%.2f R:%.2f F:%.2f' % test_acc)
    prev_fsc = 0
    stop_count = 0
    best_fsc = 0
    best_acc = 0
    logging.info("training %d examples" % len(train_examples))
    start_time = time.time()
    n_updates = 0
    for epoch in range(args.epoches):
        np.random.shuffle(train_examples)
        # if epoch > 3:
        #     logging.info("compiling Theano function again...")
        #     args.learning_rate *= 0.9
        #     att_fn, eval_fn, train_fn, params = create_theano_function(
        #         word_embed, char_embed, values=[x.get_value() for x in params])
        for batch_x, _ in train_examples:
            batch_x, batch_y = zip(*batch_x)
            batch_x = util.vectorization(list(batch_x), word_dict, char_dict, max_char_length=args.max_char)
            batch_rnn, batch_sent_mask, batch_word_mask, _ = \
                util.mask_padding(batch_x, args.max_sent, args.max_word, args.max_char)
            batch_y = np.array(list(batch_y))
            train_loss = train_fn(batch_rnn, batch_word_mask, batch_sent_mask, batch_y)
            n_updates += 1
            if n_updates % 100 == 0 and epoch > 7:
                logging.info('Epoch = %d, loss = %.2f, elapsed time = %.2f (s)' %
                             (epoch, train_loss, time.time() - start_time))
                # dev_acc = evals.eval_batch(eval_fn, dev_examples, word_dict, char_dict, args)
                dev_acc = evals.eval_vec_batch(eval_fn, dev_examples, char=False, sent_ling=False, doc_ling=False)
                logging.info('Dev A: %.2f P:%.2f R:%.2f F:%.2f' % dev_acc)
                if dev_acc[3] > best_fsc and dev_acc[0] > best_acc:
                    best_fsc = dev_acc[3]
                    best_acc = dev_acc[0]
                    logging.info('Best dev f1: epoch = %d, n_udpates = %d, f1 = %.2f %%'
                                 % (epoch, n_updates, dev_acc[3]))
                    record = 'Best dev accuracy: epoch = %d, n_udpates = %d ' % \
                             (epoch, n_updates) + ' Dev A: %.2f P:%.2f R:%.2f F:%.2f' % dev_acc
                    test_acc = evals.eval_vec_batch(eval_fn, test_examples, char=False, sent_ling=False, doc_ling=False)
                    print('Performance on Test set: A: %.2f P:%.2f R:%.2f F:%.2f' % test_acc)
                    # util.save_params('char_not_params_%.2f' % test_acc[3], params,
                    #                  epoch=epoch, n_updates=n_updates)
                if prev_fsc > dev_acc[3]:
                    stop_count += 1
                else:
                    stop_count = 0
                if stop_count == 6:
                    print("stopped")
                prev_fsc = dev_acc[3]

    print(record)
    print('Performance on Test set: A: %.2f P:%.2f R:%.2f F:%.2f' % test_acc)
    return


def create_theano_function(word_embed, char_embed, values=None):
    word_x = T.itensor3('word_x')
    word_mask = T.tensor3('word_mask')
    sent_mask = T.matrix('sent_mask')
    label_y = T.ivector('label_y')
    att_out, network_output, loss = fn.build_fn(word_x=word_x, word_mask=word_mask, sent_mask=sent_mask,
                                                label_y=label_y, word_embed=word_embed, char_embed=None,
                                                args=args)
    if values is not None:
        lasagne.layers.set_all_param_values(network_output, values, trainable=True)

    params = lasagne.layers.get_all_params(network_output, trainable=True)
    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'momentum':
        updates = lasagne.updates.momentum(loss, params, args.learning_rate)
    train_fn = theano.function([word_x, word_mask, sent_mask, label_y],
                               loss, updates=updates)

    prediction = lasagne.layers.get_output(network_output, deterministic=True)
    eval_fn = theano.function([word_x, word_mask, sent_mask], prediction)
    fn_check_attention = theano.function([word_x, word_mask, sent_mask], att_out)
    return fn_check_attention, eval_fn, train_fn, params


if __name__ == '__main__':
    args = ap.get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M")
    logging.info(' '.join(sys.argv))
    # args.debug=True
    args.word_att = "dot"
    args.batch_size = 14
    # args.optimizer = "momentum"
    args.learning_rate = 0.2
    args.dropout_rate = 0.5
    main(args)
