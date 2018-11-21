import tensorflow as tf
from argparse import ArgumentParser
import os

FLAGS = {}
FLAGS['mode'] = 'test'                          # train / test / generate / clean
FLAGS['model_typ'] = 'rnn-last'                 # cnn / rnn-last / rnn-avg / xgboost 
FLAGS['data_name'] = 'NLPCC_word'
FLAGS['model_dir'] = 'sentiment_analysis/model/'

FLAGS['data_dir'] = 'sentiment_analysis/data/'
FLAGS['batch_size'] = 32
FLAGS['unit_size'] = 256
FLAGS['vocab_size'] = 50000
FLAGS['max_length'] = 26

"""
FLAGS['printing_step'] = 1
FLAGS['saving_step'] = 2
FLAGS['num_step'] = 6
"""

FLAGS['printing_step'] = 1000
FLAGS['saving_step'] = 20000
FLAGS['num_step'] = 100000

parser = ArgumentParser()
parser.add_argument('--mode', default = 'test', help = 'train / test / generate / cleany')
parser.add_argument('--model_typ', default = 'rnn-last', help = 'cnn / rnn-last / rnn-avg / xgboost')
"""
parser.add_argument('--data_name', default = 'NLPCC_word', help = 'data name')
parser.add_argument('--model_dir', default = 'sentiment_analysis/model/', help = 'output model weight dir')
parser.add_argument('--data_dir', default = 'sentiment_analysis/data/', help = 'data dir')
parser.add_argument('--load', default = '', help = 'loaded model name')
parser.add_argument('--batch_size', default = 32, help = 'batch size')
parser.add_argument('--unit_size', default = 256, help = 'unit size')
parser.add_argument('--vocab_size', default = 50000, help = 'max vocab size')
parser.add_argument('--max_length', default = 26, help = 'sentence length')

parser.add_argument('--printing_step', default = 1, help = 'printing step')
parser.add_argument('--saving_step', default = 2, help = 'saving step')
parser.add_argument('--num_step', default = 6, help = 'number of steps')

parser.add_argument('--printing_step', default = 1000, help = 'printing step')
parser.add_argument('--saving_step', default = 20000, help = 'saving step')
parser.add_argument('--num_step', default = 100000, help = 'number of steps')

parser.add_argument('--r1', default = 0.0, help = 'r1 weight')
parser.add_argument('--r2', default = 1.0, help = 'r2 weight')
parser.add_argument('--r3', default = 1.0, help = 'r3 weight')

"""
FLAGS2 = parser.parse_args()

print(type(FLAGS2))

FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'data_{}'.format(FLAGS.data_name))
FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}_{}'.format(FLAGS.model_typ, FLAGS.data_name))

