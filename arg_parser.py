import argparse


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--debug', type='bool', default=False,
                        help='debug mode, where a small sample is trained')
    parser.add_argument('--test_only', type='bool', default=False,
                        help='no training, run with saved parameters')
    parser.add_argument('--random_seed', type=int, default=880126,
                        help='random seed')
    parser.add_argument('--train_file', type=str, default=None,
                        help='path to training file')
    parser.add_argument('--dev_test_file', type=str, default=None,
                        help='path to develop/test file')
    parser.add_argument('--model_file', type=str, default='model.pkl.gz',
                        help='Model parameter file to save')
    parser.add_argument('--embedding_file', type=str, default="glove.6B.100d.txt",
                        help='Word embedding file')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='Default embedding size if embedding_file is not given')
    parser.add_argument('--bidir', type='bool', default=True,
                        help='bidir: whether to use a bidirectional RNN')
    parser.add_argument('--max_ling', type=int, default=110,
                        help='')
    ############################################################################
    parser.add_argument('--batch_size', type=int, default=14,
                        help='Batch size')
    parser.add_argument('--batch_type', type=str, default='std',
                        help='Minibatch type')
    parser.add_argument('--epoches', type=int, default=15,
                        help='Number of epoches')
    parser.add_argument('--eval_iter', type=int, default=100,
                        help='Evaluation on dev set after K updates')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Optimizer: sgd (default) or adam or rmsprop or momentum')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                        help='Learning rate for SGD')
    parser.add_argument('--grad_clipping', type=float, default=10.0,
                        help='Gradient clipping')
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help='vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=60,
                        help='Hidden size of RNN units')
    parser.add_argument('--rnn_type', type=str, default='gru',
                        help='RNN type: lstm or gru (default)')
    parser.add_argument('--word_att', type=str, default='last',
                        help='Attention function: mlp or avg or last or dot or abs')
    parser.add_argument('--sent_att', type=str, default='dot',
                        help='Attention function: mlp or avg or last or dot or abs')
    parser.add_argument('--num_of_class', type=int, default=2,
                        help='binary class or k cluster on true (k + 1) class')
    parser.add_argument('--max_sent', type=int, default=16,
                        help='max number of sentences in a document')
    parser.add_argument('--max_word', type=int, default=128,
                        help='max number of words in a sentence')
    parser.add_argument('--max_char', type=int, default=24,
                        help='max number of characters in a word')
    parser.add_argument('--num_filter', type=int, default=30,
                        help='')
    parser.add_argument('--conv_window', type=int, default=3,
                        help='')
    parser.add_argument('--sent_ling_nonlinear', type=bool, default=True, help='')
    parser.add_argument('--doc_ling_nonlinear', type=bool, default=False, help='')
    parser.add_argument('--regularization', type=bool, default=False, help='')
    return parser.parse_args()
