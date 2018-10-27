from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--mode', default = 'test', help = 'train / test / generate / clean')
parser.add_argument('--model_typ', default = 'rnn-last', help = 'cnn / rnn-last / rnn-avg / xgboost')
parser.add_argument('--data_name', default = 'NLPCC', help = 'data name')
parser.add_argument('--model_dir', default = 'sentiment_analysis/model/', help = 'output model weight dir')
parser.add_argument('--data_dir', default = 'sentiment_analysis/data/', help = 'data dir')
parser.add_argument('--load', default = '', help = 'loaded model name')
parser.add_argument('--batch_size', default = 32, help = 'batch size')
parser.add_argument('--unit_size', default = 256, help = 'unit size')
parser.add_argument('--vocab_size', default = 50000, help = 'max vocab size')
parser.add_argument('--max_length', default = 26, help = 'sentence length')

"""
parser.add_argument('--printing_step', default = 1, help = 'printing step')
parser.add_argument('--saving_step', default = 2, help = 'saving step')
parser.add_argument('--num_step', default = 6, help = 'number of steps')
"""

parser.add_argument('--printing_step', default = 1000, help = 'printing step')
parser.add_argument('--saving_step', default = 20000, help = 'saving step')
parser.add_argument('--num_step', default = 100000, help = 'number of steps')

FLAGS = parser.parse_args()
FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'data_{}'.format(FLAGS.data_name))
FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}_{}'.format(FLAGS.model_typ, FLAGS.data_name))


