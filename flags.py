import tensorflow as tf
import os
import json
import hickle as hkl 

#fasttext_model = './cc.zh.300.bin'
#fasttext_hkl = '%sfasttext.hkl'%corpus_dir 

#tf.app.flags.DEFINE_string('data_name', 'NLPCC', 'directory of data')
tf.app.flags.DEFINE_string('data_name', 'CN', 'directory of data')
tf.app.flags.DEFINE_string('data_dir', 'data', 'directory of data')
tf.app.flags.DEFINE_string('model_dir', 'model', 'directory of model')
tf.app.flags.DEFINE_string('load', '', 'load model')

tf.app.flags.DEFINE_string('mode', 'MLE', 'MLE / RL / val_pre / val_rl / TEST')
tf.app.flags.DEFINE_string('data', 'train', 'directory of data')
tf.app.flags.DEFINE_string('data_test', 'test', 'directory of data')
tf.app.flags.DEFINE_string('source_data', 'x', 'directory of data')
tf.app.flags.DEFINE_string('target_data', 'y', 'directory of data')

tf.app.flags.DEFINE_integer('src_vocab_size', 50000, 'vocabulary size of the input')
tf.app.flags.DEFINE_integer('trg_vocab_size', 50000, 'vocabulary size of the input')
tf.app.flags.DEFINE_integer('hidden_size', 512, 'number of units of hidden layer')
tf.app.flags.DEFINE_integer('num_layers', 4, 'number of layers')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
# for rnn dropout
tf.app.flags.DEFINE_float('input_keep_prob', '0.8', 'step input dropout of saving model')
tf.app.flags.DEFINE_float('output_keep_prob', '1.0', 'step output dropout of saving model')
tf.app.flags.DEFINE_float('state_keep_prob', '1.0', 'step state dropout of saving model')
# output_keep_prob is the dropout added to the RNN's outputs, the dropout will have no effect on the calculation of the subsequent states.
# beam search
tf.app.flags.DEFINE_boolean('beam_search', False, 'beam search')
tf.app.flags.DEFINE_integer('beam_size', 10 , 'beam size')
tf.app.flags.DEFINE_boolean('debug', True, 'debug')
# schedule sampling
tf.app.flags.DEFINE_string('schedule_sampling', 'linear', 'schedule sampling type[linear|exp|inverse_sigmoid|False]')
tf.app.flags.DEFINE_float('sampling_decay_rate', 0.99 , 'schedule sampling decay rate')
tf.app.flags.DEFINE_integer('sampling_global_step', 10000000, 'sampling_global_step')
tf.app.flags.DEFINE_integer('sampling_decay_steps', 300, 'sampling_decay_steps')
tf.app.flags.DEFINE_boolean('reset_sampling_prob', False, 'reset_sampling_prob')
# word segmentation type
tf.app.flags.DEFINE_string('src_word_seg', 'word', 'source word segmentation type')
tf.app.flags.DEFINE_string('trg_word_seg', 'word', 'target word segmentation type')
# if load pretrain word vector
tf.app.flags.DEFINE_string('pretrain_vec', None, 'load pretrain word vector')
tf.app.flags.DEFINE_boolean('pretrain_trainable', False, 'pretrain vec trainable or not')

#reward
tf.app.flags.DEFINE_float('r1', 0.4, 'r1 weight')
tf.app.flags.DEFINE_float('r2', 0.3, 'r2 weight')
tf.app.flags.DEFINE_float('r3', 0.3, 'r3 weight')

tf.app.flags.DEFINE_string('output', 'output', 'output file')

"""
tf.app.flags.DEFINE_integer('print_step', '1', 'step interval of printing')
tf.app.flags.DEFINE_integer('check_step', '2', 'step interval of saving model')
tf.app.flags.DEFINE_integer('max_step', '6', 'max step')
"""

tf.app.flags.DEFINE_integer('print_step', '500', 'step interval of printing')
tf.app.flags.DEFINE_integer('check_step', '500', 'step interval of saving model')
tf.app.flags.DEFINE_integer('max_step', '2000', 'max step')

FLAGS = tf.app.flags.FLAGS

FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}'.format(FLAGS.data_name))

if FLAGS.mode == 'val_pre':
  FLAGS.output = os.path.join(FLAGS.output, 'output_{}_RL_pre'.format(FLAGS.data_name))
elif FLAGS.mode == 'val_rl':
  FLAGS.output = os.path.join(FLAGS.output, 'output_{}_RL_{}_{}_{}'.format(FLAGS.data_name, FLAGS.r1, FLAGS.r2, FLAGS.r3))

if FLAGS.load != '':
  FLAGS.output = '{}_{}'.format(FLAGS.output, FLAGS.load)
  FLAGS.load = os.path.join(FLAGS.model_rl_dir, 'RL.ckpt-{}'.format(FLAGS.load))

if not os.path.exists(FLAGS.model_dir):
    print('create model dir: ', FLAGS.model_dir)
    os.mkdir(FLAGS.model_dir)

FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'data_{}'.format(FLAGS.data_name))
FLAGS.data = os.path.join(FLAGS.data_dir, FLAGS.data)
FLAGS.data_test = os.path.join(FLAGS.data_dir, FLAGS.data_test)
FLAGS.source_data = os.path.join(FLAGS.data_dir, FLAGS.source_data)
FLAGS.target_data = os.path.join(FLAGS.data_dir, FLAGS.target_data)

if 'val' in FLAGS.mode:
  FLAGS.batch_size = 1

# for data etl
SEED = 112
#buckets = [(5, 5), (10, 10), (15, 15)]
buckets = [(5, 5), (10, 10)]
#split_ratio = 0.99

# for reset schedule sampling probability
reset_prob = 1.0

# apply same word segment strategy to both source and target or not
word_seg_strategy = 'diff'

