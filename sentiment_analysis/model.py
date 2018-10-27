from . import utils
import os
import numpy as np
import random
import tensorflow as tf

class discriminator():

  def __init__(self, args):
    self.vocab_size = args.vocab_size
    self.unit_size = args.unit_size
    self.batch_size = args.batch_size
    self.max_length = args.max_length
    self.mode = args.mode
    self.model_dir = args.model_dir
    self.data_dir = args.data_dir
    self.model_type = {
      'cnn': self.build_model_cnn,
      'rnn-last': self.build_model_rnn_last,
      'rnn-ave': self.build_model_rnn_ave,
      'xgboost': self.build_model_xgboost,
    }
    self.dropout_keep_prob = 1.0 
    self.seq_length = tf.placeholder(tf.int32, [None])
    self.initializer = tf.random_normal_initializer(seed=1995)
    #self.initializer = tf.glorot_uniform_initializer()
    #self.initializer = tf.glorot_normal_initializer()

    print('\n\n----------------')
    print(utils)

    # for cnn
    self.filter_sizes = [3,4]
    self.num_filters = 128 

    self.build_model(args.model_typ)

    if not os.path.exists(self.model_dir):
      os.system('mkdir {}'.format(self.model_dir))
    self.saver = tf.train.Saver(max_to_keep = 5)


  def build_model(self,typ='cnn'):
    self.model_type[typ]()

  def build_model_cnn(self):
    print('==== model type: cnn ====')
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size],initializer=self.initializer)
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    embedding_expanded = tf.expand_dims(embedding,-1)
    pooled_outputs = []
    for i, filter_size in enumerate(self.filter_sizes):
      with tf.name_scope('conv-maxpool-%s' % filter_size):
        filter_shape = [filter_size, self.unit_size, 1, self.num_filters]
        W = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1), name='cnn_w')
        b = tf.Variable(
            tf.constant(0.1, shape=[self.num_filters]), name='cnn_b')
        conv = tf.nn.conv2d(
            embedding_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, self.max_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='pool')
        pooled_outputs.append(pooled)

    num_filters_total = self.num_filters * len(self.filter_sizes)
    with tf.name_scope('concat'):
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
    with tf.name_scope('dropout'):
        self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                    self.dropout_keep_prob)
    self.top_layer(self.h_drop)

  def build_model_xgboost(self):
    print('==== model type: xgboost ====')
    pass

  def build_model_rnn_ave(self):
    print('==== model type: rnn ave ====')
    cell = tf.contrib.rnn.GRUCell(self.unit_size)
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size],initializer=self.initializer)
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    
    outputs, hidden_state = tf.nn.dynamic_rnn(cell, embedding, sequence_length = self.seq_length, dtype = tf.float32) 
    outputs = tf.reduce_mean(outputs,axis=1)
    self.top_layer(outputs)

  def build_model_rnn_last(self):
    print('==== model type: rnn last ====')
    cell = tf.contrib.rnn.GRUCell(self.unit_size)
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size],initializer=self.initializer)
    print('params: ',params,tf.shape(params))
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    print('encoder_input: ',self.encoder_input,tf.shape(self.encoder_input))
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    print('embedding: ',embedding, tf.shape(embedding))
    
    _, hidden_state = tf.nn.dynamic_rnn(cell, embedding, sequence_length = self.seq_length, dtype = tf.float32) 
    self.top_layer(hidden_state)

  def top_layer(self,outputs):
    w = tf.get_variable('w', [self.unit_size, 1])
    b = tf.get_variable('b', [1])
    output = tf.matmul(outputs, w) + b

    self.logit = tf.nn.sigmoid(output)
    self.pred = tf.to_int32(self.logit > 0.5)

    if self.mode != 'test':
      self.target = tf.placeholder(tf.float32, [None, 1])
      self.target_int = tf.placeholder(tf.int32, [None, 1])
      self.loss = tf.reduce_mean(tf.square(self.target - self.logit))
      self.acc = tf.reduce_mean(tf.contrib.metrics.accuracy(self.pred, self.target_int))
      self.opt = tf.train.AdamOptimizer().minimize(self.loss)
    else:
      self.vocab_map, _ = utils.read_map(os.path.join(self.data_dir, 'dict'))

  def step(self, session, encoder_inputs, seq_length, target = None):
    input_feed = {}
    input_feed[self.encoder_input] = encoder_inputs
    input_feed[self.seq_length] = seq_length

    if self.mode == 'train':
      input_feed[self.target] = target
      input_feed[self.target_int] = target
      
      output_feed = [self.loss, self.acc, self.opt]
      outputs = session.run(output_feed, input_feed)
      return outputs
    elif self.mode == 'valid':
      input_feed[self.target] = target
      input_feed[self.target_int] = target

      output_feed = [self.loss, self.acc]
      outputs = session.run(output_feed, input_feed)
      return outputs
    elif self.mode == 'test':
      output_feed = [self.logit]
      outputs = session.run(output_feed, input_feed)
      return outputs

  def get_batch(self, data, shuffle=True, xy=None):
    encoder_inputs = []
    encoder_length = []
    target = []
    sen = []

    num = min(self.batch_size, len(data))

    for i in range(num):
      if shuffle:
          pair = random.choice(data)
      else:
          pair = data[i]

      length = len(pair[1])
      target.append([pair[0]])
      sen.append(pair[2])
      if length > self.max_length:
        s = pair[1][:self.max_length]
        s[-1] = utils.EOS_id
        encoder_inputs.append(s)
        encoder_length.append(self.max_length)
      else:
        encoder_pad = [utils.EOS_id] * (self.max_length - length)
        encoder_inputs.append(pair[1] + encoder_pad)
        encoder_length.append(length)

    batch_input = np.array(encoder_inputs, dtype = np.int32)
    batch_length = np.array(encoder_length, dtype = np.int32)
    batch_target = np.array(target, dtype = np.float32) if xy == None else target

    return batch_input, batch_length, batch_target, sen

if __name__ == '__main__':
  test = discriminator(1000, 100, 32, 1, 50)

