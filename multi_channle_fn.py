import lasagne
import networks
import theano.tensor as T


def build_fn(word_x, word_mask, sent_mask, label_y, word_embed, args,
             char_x=None, char_embed=None, sent_ling=None, doc_ling=None):
    word_input_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word), input_var=word_x)
    word_mask_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word), input_var=word_mask)
    word_mask_layer = lasagne.layers.reshape(word_mask_layer, (-1, [2]))
    sent_mask_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent), input_var=sent_mask)

    if char_x is not None:
        char_input_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_word, args.max_char),
                                                     input_var=char_x)
        char_cnn = networks.char_cnn(char_input_layer, args.num_filter, args.conv_window, char_embed, args)
        word_rnn = networks.word_rnn(word_input_layer, word_mask_layer, word_embed, args, char_cnn)
    else:
        word_rnn = networks.word_rnn(word_input_layer, word_mask_layer, word_embed, args)

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

    if sent_ling is not None:
        sent_linguistic_layer = lasagne.layers.InputLayer(shape=(None, args.max_sent, args.max_ling),
                                                          input_var=sent_ling)
        sent_linguistic_layer = lasagne.layers.DenseLayer(sent_linguistic_layer, 60, num_leading_axes=2,
                                                          nonlinearity=lasagne.nonlinearities.rectify)
        sent_input = lasagne.layers.ConcatLayer([sent_rnn, sent_linguistic_layer], axis=-1)

    else:
        sent_input = sent_rnn

    if args.dropout_rate > 0:
        sent_input = lasagne.layers.dropout(sent_input, p=args.dropout_rate)
    sent_input = lasagne.layers.DenseLayer(sent_input, 2 * args.hidden_size, num_leading_axes=-1,
                                           nonlinearity=lasagne.nonlinearities.tanh)
    sent_att = networks.Attention(sent_input, num_units=2 * args.hidden_size, mask=sent_mask_layer)

    att_out = lasagne.layers.get_output(sent_att, deterministic=True)
    sent_output = networks.AttOutput([sent_rnn, sent_att])

    if doc_ling is not None:
        doc_linguistic_layer = lasagne.layers.InputLayer(shape=(None, args.max_ling), input_var=doc_ling)
        if args.doc_ling_nonlinear:
            doc_linguistic_layer = lasagne.layers.DenseLayer(doc_linguistic_layer, 60, num_leading_axes=-1,
                                                             nonlinearity=lasagne.nonlinearities.rectify)
        if args.dropout_rate > 0:
            doc_linguistic_layer = lasagne.layers.dropout(doc_linguistic_layer, p=args.dropout_rate)
        sent_output = lasagne.layers.ConcatLayer([sent_output, doc_linguistic_layer], axis=-1)

    network_output = lasagne.layers.DenseLayer(sent_output, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    train_pred = lasagne.layers.get_output(network_output)
    weights = T.ones_like(label_y, dtype='float32')
    weights = T.set_subtensor(weights[T.eq(label_y, 0.).nonzero()], 1.4)
    loss = (weights * lasagne.objectives.binary_crossentropy(train_pred, label_y)).mean()

    if args.regularization:
        regularization = lasagne.regularization.regularize_layer_params(network_output,
                                                                        penalty=lasagne.regularization.l2)
        loss += regularization * 0.0001
    return att_out, network_output, loss


