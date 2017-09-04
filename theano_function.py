import theano
import lasagne
import networks
import logging
import theano.tensor as T


def char_hierarchical_doc_fn(args, word_embed, char_embed, values=None):
    char_x = T.itensor4('char_x')
    word_x = T.itensor3('word_x')
    word_mask = T.tensor3('word_mask')
    sent_mask = T.matrix('sent_mask')
    doc_linguistic_x = T.matrix('doc_linguistic')
    label_y = T.ivector('label_y')

    char_input_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word, args.max_char),
                                                 input_var=char_x)
    word_input_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word), input_var=word_x)
    word_mask_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word), input_var=word_mask)
    word_mask_layer = lasagne.layers.reshape(word_mask_layer, (-1, [2]))
    sent_mask_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent), input_var=sent_mask)
    doc_linguistic_layer = lasagne.layers.InputLayer(shape=(None, args.max_ling), input_var=doc_linguistic_x)

    char_cnn = networks.char_cnn(char_input_layer, args.num_filter, args.conv_window, char_embed, args)
    word_rnn = networks.word_rnn(word_input_layer, word_mask_layer, word_embed, args, char_cnn)
    if args.dropout_rate > 0:
        word_rnn = lasagne.layers.dropout(word_rnn, p=args.dropout_rate)

    if args.word_att == 'avg':
        word_output = networks.AveragePooling(word_rnn, mask=word_mask_layer)
    elif args.word_att == 'last':
        word_output = word_rnn
    elif args.word_att == 'dot':
        word_att = lasagne.layers.DenseLayer(word_rnn, num_units=2 * args.hidden_size, num_leading_axes=-1,
                                             nonlinearity=lasagne.nonlinearities.tanh)
        word_att = networks.Attention(word_att, num_units=2 * args.hidden_size, mask=word_mask_layer)
        word_output = networks.AttOutput([word_rnn, word_att])

    word_output = lasagne.layers.reshape(word_output, (-1, args.max_sent, [1]))
    sent_rnn = networks.sent_rnn(word_output, sent_mask_layer, args)
    if args.dropout_rate > 0:
        sent_rnn = lasagne.layers.dropout(sent_rnn, p=args.dropout_rate)
    sent_input = lasagne.layers.DenseLayer(sent_rnn, 2 * args.hidden_size, num_leading_axes=-1,
                                           nonlinearity=lasagne.nonlinearities.tanh)

    sent_att = networks.Attention(sent_input, num_units=2 * args.hidden_size, mask=sent_mask_layer)

    att_out = lasagne.layers.get_output(sent_att, deterministic=True)
    fn_check_attention = theano.function([char_x, word_x, word_mask, sent_mask], att_out)

    sent_output = networks.AttOutput([sent_rnn, sent_att])

    if args.doc_ling_nonlinear:
        doc_linguistic_layer = lasagne.layers.DenseLayer(doc_linguistic_layer, 60, num_leading_axes=-1,
                                                         nonlinearity=lasagne.nonlinearities.rectify)
    if args.dropout_rate > 0:
        doc_linguistic_layer = lasagne.layers.dropout(doc_linguistic_layer, p=args.dropout_rate)

    sent_output = lasagne.layers.ConcatLayer([sent_output, doc_linguistic_layer], axis=-1)
    network_output = lasagne.layers.DenseLayer(sent_output, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    regularization = lasagne.regularization.regularize_layer_params(network_output, penalty=lasagne.regularization.l2)
    train_pred = lasagne.layers.get_output(network_output)
    loss = lasagne.objectives.binary_crossentropy(train_pred, label_y).mean() + regularization * 0.0001

    if values is not None:
        lasagne.layers.set_all_param_values(network_output, values, trainable=True)

    params = lasagne.layers.get_all_params(network_output, trainable=True)

    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'momentum':
        updates = lasagne.updates.momentum(loss, params, args.learning_rate)

    train_fn = theano.function([word_x, char_x, word_mask, sent_mask, doc_linguistic_x, label_y],
                               loss, updates=updates)

    prediction = lasagne.layers.get_output(network_output, deterministic=True)
    eval_fn = theano.function([word_x, char_x, word_mask, sent_mask, doc_linguistic_x],
                              prediction)
    return fn_check_attention, eval_fn, train_fn, params


def char_hierarchical_linguistic_fn(args, word_embed, char_embed, values=None, if_cnn=True):
    char_x = T.itensor4('char_x')
    word_x = T.itensor3('word_x')
    word_mask = T.tensor3('word_mask')
    sent_mask = T.matrix('sent_mask')
    sent_linguistic_x = T.tensor3('sent_linguistic')
    doc_linguistic_x = T.matrix('doc_linguistic')
    label_y = T.ivector('label_y')

    char_input_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word, args.max_char),
                                                 input_var=char_x)
    word_input_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word), input_var=word_x)
    word_mask_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word), input_var=word_mask)
    word_mask_layer = lasagne.layers.reshape(word_mask_layer, (-1, [2]))
    sent_mask_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent), input_var=sent_mask)
    sent_linguistic_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_ling),
                                                      input_var=sent_linguistic_x)
    doc_linguistic_layer = lasagne.layers.InputLayer(shape=(None, args.max_ling), input_var=doc_linguistic_x)

    char_cnn = networks.char_cnn(char_input_layer, args.num_filter, args.conv_window, char_embed, args)
    word_rnn = networks.word_rnn(word_input_layer, word_mask_layer, word_embed, args, char_cnn)

    if args.dropout_rate > 0:
        word_rnn = lasagne.layers.dropout(word_rnn, p=args.dropout_rate)

    if args.word_att == 'avg':
        word_output = networks.AveragePooling(word_rnn, mask=word_mask_layer)
    elif args.word_att == 'last':
        word_output = word_rnn
    elif args.word_att == 'dot':
        word_att = lasagne.layers.DenseLayer(word_rnn, num_units=args.hidden_size, num_leading_axes=-1,
                                             nonlinearity=lasagne.nonlinearities.tanh)
        word_att = networks.Attention(word_att, num_units=args.hidden_size, mask=word_mask_layer)
        word_output = networks.AttOutput([word_rnn, word_att])

    word_output = lasagne.layers.reshape(word_output, (-1, args.max_sent, [1]))
    sent_rnn = networks.sent_rnn(word_output, sent_mask_layer, args)

    if args.sent_ling_nonlinear:
        # leaky_rectify = lasagne.nonlinearities.leaky_rectify(0.01)
        sent_linguistic_layer = lasagne.layers.DenseLayer(sent_linguistic_layer, 60, num_leading_axes=2,
                                                          nonlinearity=lasagne.nonlinearities.rectify)
    sent_input = lasagne.layers.ConcatLayer([sent_rnn, sent_linguistic_layer], axis=-1)
    if args.dropout_rate > 0:
        sent_input = lasagne.layers.dropout(sent_input, p=args.dropout_rate)
    sent_input = lasagne.layers.DenseLayer(sent_input, 2 * args.hidden_size, num_leading_axes=-1,
                                           nonlinearity=lasagne.nonlinearities.tanh)

    sent_att = networks.Attention(sent_input, num_units=2 * args.hidden_size, mask=sent_mask_layer)

    att_out = lasagne.layers.get_output(sent_att, deterministic=True)
    fn_check_attention = theano.function([word_x, char_x, word_mask, sent_mask, sent_linguistic_x],
                                         att_out)

    sent_output = networks.AttOutput([sent_rnn, sent_att])

    if args.doc_ling_nonlinear:
        doc_linguistic_layer = lasagne.layers.DenseLayer(doc_linguistic_layer, 60, num_leading_axes=-1,
                                                         nonlinearity=lasagne.nonlinearities.rectify)
    if args.dropout_rate > 0:
        doc_linguistic_layer = lasagne.layers.dropout(doc_linguistic_layer, p=args.dropout_rate)

    sent_output = lasagne.layers.ConcatLayer([sent_output, doc_linguistic_layer], axis=-1)
    # network_output = lasagne.layers.DenseLayer(sent_output, num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
    network_output = lasagne.layers.DenseLayer(sent_output, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    train_pred = lasagne.layers.get_output(network_output)
    weights = T.ones_like(label_y, dtype='float32')
    weights = T.set_subtensor(weights[T.eq(label_y, 0.).nonzero()], 1.8)
    loss = (weights * lasagne.objectives.binary_crossentropy(train_pred, label_y)).mean()
    # loss = lasagne.objectives.binary_crossentropy(train_pred, label_y).mean()

    # amplify = T.ones_like(label_y)
    # amplify[label_y == 0] *= 4
    # loss = (lasagne.objectives.binary_crossentropy(train_pred, label_y) * amplify).mean()

    # if args.regularization:
    #     regularization = lasagne.regularization.regularize_layer_params(network_output,
    #                                                                     penalty=lasagne.regularization.l2)
    #     loss += regularization * 0.0001

    if values is not None:
        lasagne.layers.set_all_param_values(network_output, values, trainable=True)

    params = lasagne.layers.get_all_params(network_output, trainable=True)

    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'momentum':
        updates = lasagne.updates.momentum(loss, params, args.learning_rate)

    train_fn = theano.function([word_x, char_x, word_mask, sent_mask, sent_linguistic_x, doc_linguistic_x, label_y],
                               loss, updates=updates)

    prediction = lasagne.layers.get_output(network_output, deterministic=True)
    eval_fn = theano.function([word_x, char_x, word_mask, sent_mask, sent_linguistic_x, doc_linguistic_x],
                              prediction)
    return fn_check_attention, eval_fn, train_fn, params


if __name__ == '__main__':
    print(float(1e-5))
    label_y = T.ivector('label_y')
    weight = T.ones_like(label_y, dtype='float32')
    weight = T.set_subtensor(weight[T.eq(label_y, 1).nonzero()], .1)
    fn = theano.function([label_y], weight)
    import numpy as np

    y = np.array([0, 0, 1, 1, 0])
    print(fn(y))
