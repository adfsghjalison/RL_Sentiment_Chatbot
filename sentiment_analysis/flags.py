import tensorflow as tf

tf.app.flags.DEFINE_string('mode', 'train', 'train / test / generate / clean')
tf.app.flags.DEFINE_string('model_typ', 'rnn_last', 'cnn / rnn_last / rnn_avg / xgboost')
tf.app.flags.DEFINE_string('model_dir', 'model/model_rnn-last_NLPCC', 'output model weight dir')
tf.app.flags.DEFINE_string('data_dir', 'data/data_NLPCC', 'data dir')
tf.app.flags.DEFINE_string('load', '', 'loaded model name')
tf.app.flags.DEFINE_integer('batch_size', 2048, 'batch size')
tf.app.flags.DEFINE_integer('unit_size', 256, '')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'max vocab size')
tf.app.flags.DEFINE_integer('max_length', 26, 'sentence length')
tf.app.flags.DEFINE_integer('printing_step', 1000, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 20000, 'saving step')
tf.app.flags.DEFINE_integer('num_step', 100000, 'number of steps')

FLAGS = tf.app.flags.FLAGS

