import lasagne
import theano.tensor as T


def char_cnn(char_input, num_filter, conv_window, char_embedding, args):
    network = lasagne.layers.reshape(char_input, (-1, [3]))
    network = lasagne.layers.EmbeddingLayer(incoming=network, input_size=args.char_vocab_size,
                                            output_size=args.char_embed_size, W=char_embedding)
    if args.dropout_rate > 0:
        network = lasagne.layers.DropoutLayer(incoming=network, p=args.dropout_rate)
    network = lasagne.layers.DimshuffleLayer(incoming=network, pattern=(0, 2, 1))
    network = lasagne.layers.Conv1DLayer(incoming=network, num_filters=num_filter, filter_size=conv_window,
                                         pad='full', nonlinearity=lasagne.nonlinearities.rectify)
    _, _, pool_size = network.output_shape
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=pool_size)
    return lasagne.layers.reshape(network, (-1, args.max_word, [1]))


def word_rnn(word_input, word_mask, word_embedding, args, char_input=None):
    network = lasagne.layers.reshape(word_input, (-1, [2]))
    network = lasagne.layers.EmbeddingLayer(incoming=network, input_size=args.word_vocab_size,
                                            output_size=args.word_embed_size, W=word_embedding)
    if args.dropout_rate > 0:
        network = lasagne.layers.DropoutLayer(incoming=network, p=args.dropout_rate)
    if char_input is not None:
        network = lasagne.layers.ConcatLayer([network, char_input], axis=-1)

    tail_mode = (args.word_att == 'last')
    forward = args.rnn(incoming=network, mask_input=word_mask, num_units=args.hidden_size,
                       grad_clipping=args.grad_clipping, only_return_final=tail_mode,
                       backwards=True, hid_init=lasagne.init.Constant(0.))
    backward = args.rnn(incoming=network, mask_input=word_mask, num_units=args.hidden_size,
                        grad_clipping=args.grad_clipping, only_return_final=tail_mode,
                        backwards=False, hid_init=lasagne.init.Constant(0.))
    return lasagne.layers.ConcatLayer([forward, backward], axis=-1)


def sent_rnn(network, sent_mask, args):
    if args.dropout_rate > 0:
        network = lasagne.layers.DropoutLayer(incoming=network, p=args.dropout_rate)
    tail_mode = (args.sent_att == 'last')
    forward = args.rnn(incoming=network, mask_input=sent_mask, num_units=args.hidden_size,
                       grad_clipping=args.grad_clipping, only_return_final=tail_mode,
                       backwards=True, hid_init=lasagne.init.Constant(0.))
    backward = args.rnn(incoming=network, mask_input=sent_mask, num_units=args.hidden_size,
                        grad_clipping=args.grad_clipping, only_return_final=tail_mode,
                        backwards=False, hid_init=lasagne.init.Constant(0.))
    return lasagne.layers.ConcatLayer([forward, backward], axis=-1)


class AveragePooling(lasagne.layers.MergeLayer):
    def __init__(self, incoming, mask=None, **kwargs):
        incomings = [incoming]
        if mask is not None:
            incomings.append(mask)
        if len(self.input_shapes[0]) != 3:
            raise ValueError('the shape of incoming must be a 3-element tuple')
        super(AveragePooling, self).__init__(incomings=incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-2] + input_shapes[0][-1:]

    def get_output_for(self, inputs, **kwargs):
        if len(inputs) == 1:
            # mask_input is None
            return T.mean(inputs[0], axis=1)
        else:
            # inputs[0]: batch x len x h
            # inputs[1] = mask_input: batch x len
            return (T.sum(inputs[0] * inputs[1].dimshuffle(0, 1, 'x'), axis=1) /
                    T.sum(inputs[1], axis=1).dimshuffle(0, 'x'))


class Attention(lasagne.layers.MergeLayer):
    def __init__(self, incoming, num_units, mask=None, init=lasagne.init.Uniform(), **kwards):
        incomings = [incoming]
        if mask is not None:
            incomings.append(mask)
        super(Attention, self).__init__(incomings, **kwards)
        self.num_units = num_units
        self.W_attention = self.add_param(spec=init, shape=(self.num_units,), name='W_attention')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        M = T.dot(inputs[0], self.W_attention)
        alpha = T.nnet.softmax(M)
        if len(inputs) == 2:
            alpha = alpha * inputs[1]
        return alpha


class AttOutput(lasagne.layers.MergeLayer):
    def __init__(self, incomings, mask=None, **kwargs):
        assert len(incomings) == 2
        if mask is not None:
            incomings.append(mask)
        super(AttOutput, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-2] + input_shapes[0][-1:]

    def get_output_for(self, inputs, **kwargs):
        return T.sum(inputs[0] * inputs[1].dimshuffle(0, 1, 'x'), axis=1)
