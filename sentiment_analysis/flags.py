import tensorflow as tf
from argparse import ArgumentParser
import os

parser = ArgumentParser()
tf.app.flags.DEFINE_string('mode_', default = 'test', help = 'train / test / generate / clean')
tf.app.flags.DEFINE_string('model_typ', default = 'rnn-last', help = 'cnn / rnn-last / rnn-avg / xgboost')
tf.app.flags.DEFINE_string('data_name_', default = 'NLPCC_word', help = 'data name')
tf.app.flags.DEFINE_string('model_dir', default = 'sentiment_analysis/model/', help = 'output model weight dir')
tf.app.flags.DEFINE_string('data_dir_', default = 'sentiment_analysis/data/', help = 'data dir')
#tf.app.flags.DEFINE_string('load', default = '', help = 'loaded model name')
tf.app.flags.DEFINE_integer('batch_size_', default = 64, help = 'batch size')
tf.app.flags.DEFINE_integer('unit_size', default = 256, help = 'unit size')
tf.app.flags.DEFINE_integer('vocab_size', default = 50000, help = 'max vocab size')
tf.app.flags.DEFINE_integer('max_length', default = 26, help = 'sentence length')

"""
tf.app.flags.DEFINE_string('printing_step', default = 1, help = 'printing step')
tf.app.flags.DEFINE_string('saving_step', default = 2, help = 'saving step')
tf.app.flags.DEFINE_string('num_step', default = 6, help = 'number of steps')
"""

tf.app.flags.DEFINE_integer('printing_step', default = 1000, help = 'printing step')
tf.app.flags.DEFINE_integer('saving_step', default = 20000, help = 'saving step')
tf.app.flags.DEFINE_integer('num_step', default = 100000, help = 'number of steps')

#FLAGS = parser.parse_args()
FLAGS = tf.app.flags.FLAGS
FLAGS.data_dir_ = os.path.join(FLAGS.data_dir_, 'data_{}'.format(FLAGS.data_name_))
FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}_{}'.format(FLAGS.model_typ, FLAGS.data_name_))

