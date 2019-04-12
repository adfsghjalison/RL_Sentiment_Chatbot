import tensorflow as tf
from tensorflow.python.platform import gfile
import random
import os
import sys
import numpy as np
sys.path.append('../sentiment_analysis/')
from . import utils
from . import model
from .flags import FLAGS
#from . import model

#FLAGS = flags.Parse()
print(FLAGS)
model_dir = FLAGS.model_dir
data_dir = FLAGS.data_dir
batch_size = FLAGS.batch_size_

def sentence_cutter(sentence):
    sentence = [s for s in sentence]
    return (' ').join(sentence)

def create_model(session):
  m = model.discriminator(FLAGS)
  ckpt = tf.train.get_checkpoint_state(model_dir)
  print('ckpt: ',ckpt)

  if ckpt:
    print("Reading model from %s" % ckpt.model_checkpoint_path)
    m.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters")
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

  return m

def log(step, loss, acc, loss_v, acc_v):
  _file = os.path.join(FLAGS.model_dir, 'log')
  f = open(_file, 'a')
  print("\nStep %d " % step)
  print("Train Loss: %s , Train Acc: %s" % (loss, acc))
  print("Valid Loss: %s , Valid Acc: %s" % (loss_v, acc_v))
  f.write("\nStep %d\n" % step)
  f.write("Train Loss: %s , Train Acc: %s\n" % (loss, acc))
  f.write("Valid Loss: %s , Valid Acc: %s\n" % (loss_v, acc_v))
  

def train():
  if gfile.Exists(os.path.join(data_dir, 'dict')) and gfile.Exists(os.path.join(data_dir, 'source_train.token')):
    print('Files have already been formed!')
  else:
    vocab_map, _ = utils.read_map(os.path.join(data_dir, 'dict'))
    utils.file_to_token(os.path.join(data_dir, 'source_train'), vocab_map)
    utils.file_to_token(os.path.join(data_dir, 'source_test'), vocab_map)

  train_set = utils.read_data(os.path.join(data_dir, 'source_train'))
  valid_set = utils.read_data(os.path.join(data_dir, 'source_test'))

  if not os.path.exists(model_dir):
    os.system('mkdir '+model_dir)


  sess = tf.Session()

  Model = create_model(sess)
  #Model = create_model(sess, 'valid')
  step = 0
  loss, acc = 0, 0

  while step < FLAGS.num_step:
    step += 1
    encoder_input, encoder_length, target, _ = Model.get_batch(train_set)
    '''
    print(encoder_input)
    print(encoder_length)
    print(target)
    exit()
    '''
    loss_train, acc_train, _ = Model.step(sess, encoder_input, encoder_length, target)
    loss += loss_train/FLAGS.printing_step
    acc += acc_train/FLAGS.printing_step

    if step % FLAGS.printing_step == 0 or step % FLAGS.saving_step == 0:
      Model.mode = 'valid'
      temp_loss, temp_acc = 0, 0
      for _ in range(100):
        encoder_input, encoder_length, target, _ = Model.get_batch(valid_set)
        loss_valid, acc_valid = Model.step(sess, encoder_input, encoder_length, target)
        temp_loss += loss_valid/100.
        temp_acc += acc_valid/100.
      log(step, loss, acc, temp_loss, temp_acc)

      if step % FLAGS.saving_step == 0:
        checkpoint_path = os.path.join(model_dir, 'dis.ckpt')
        Model.saver.save(sess, checkpoint_path, global_step = step)
        print("\nStep %s : Model Saved!" % step)

      loss, acc = 0, 0
      Model.mode = 'train'

def test():
  print('\n\n****************')
  print(utils)
  vocab_map, _ = utils.read_map(os.path.join(data_dir, 'dict'))
  sess = tf.Session()
  Model = create_model(sess)
  Model.batch_size = 1
  
  sys.stdout.write('>')
  sys.stdout.flush()
  sentence = sys.stdin.readline()
  sentence = sentence_cutter(sentence)

  while(sentence):
    token_ids = utils.convert_to_token(sentence, vocab_map)
    print('toekn_ids: ',token_ids)
    encoder_input, encoder_length, _, _ = Model.get_batch([(0, token_ids, sentence)]) 
    print('encoder_input: ',encoder_input, encoder_input.shape)
    print('encoder_length: ',encoder_length)
    score = Model.step(sess, encoder_input, encoder_length)
    print('Score: ' + str(score[0][0]))
    print ('>')
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    sentence = sentence_cutter(sentence)

def generate_data_d_x(Model):
  vocab_map, _ = utils.read_map(os.path.join(data_dir, 'dict'))
  data = utils.read_data(os.path.join(data_dir, 'sentiment'), vocab_map, skip=False)

  f = open(os.path.join(data_dir, 'int_x'), 'w')
  H = 0.8
  L = 0.2

  num = len(data)
  start = 0
  cnt_h, cnt_l = 0, 0

  while start < num:
    if start + batch_size < num:
      encoder_input, encoder_length, S, X = Model.get_batch(data[start : start + batch_size], shuffle=False)
    else:
      encoder_input, encoder_length, S, X = Model.get_batch(data[start:], shuffle=False)

    score = Model.step(sess, encoder_input, encoder_length)
    for s, l, x in zip(score[0], S, X):
      l = l[0]
      if s > H and (l == -1 or l == 1):
        f.write("{} +++$+++ {}\n".format(1, x))
        cnt_h += 1
      elif s < L and (l == -1 or l == 0):
        f.write("{} +++$+++ {}\n".format(0, x))
        cnt_l += 1

    if start % (batch_size*200) == 0:
      print ('\n\n-------------------{}--------------------\n\n'.format(start))
    start += batch_size

  f.close()

  print ('H : {}\nL : {}'.format(cnt_h, cnt_l))


def generate_data_f_x_y(Model, xy=True):
  vocab_map, _ = utils.read_map(os.path.join(data_dir, 'dict'))
  data = utils.read_data(os.path.join(data_dir, 'chatbot'), vocab_map, xy=xy, skip=False)

  f = open(os.path.join(data_dir, 'float_x_y'), 'w')

  num = len(data)
  start = 0

  while start < num:
    if start + batch_size < num:
      encoder_input, encoder_length, X, Y = Model.get_batch(data[start : start + batch_size], shuffle=False, xy=xy)
    else:
      encoder_input, encoder_length, X, Y = Model.get_batch(data[start:], shuffle=False, xy=xy)

    score = Model.step(sess, encoder_input, encoder_length)
    for s, x, y in zip(score[0], X, Y):
      f.write("{} +++$+++ {} +++$+++ {}\n".format(round(s[0], 2), x[0], y))

    if start % (batch_size*200) == 0:
      print ('\n\n-------------------{}--------------------\n\n'.format(start))
    start += batch_size

  f.close()

def clean():
  #os.system('/bin/rm '+os.path.join(FLAGS.data_dir, 'source_train.token'))
  os.system('/bin/rm '+os.path.join(FLAGS.model_dir, '*'))

if __name__ == '__main__':
  if FLAGS.mode == 'train':
    train()
  elif FLAGS.mode == 'test':
    test()
  elif FLAGS.mode == 'generate':
    sess = tf.Session()
    Model = create_model(sess)
    Model.mode='test'
    #generate_data_d_x(Model)
    generate_data_f_x_y(Model)
  elif FLAGS.mode == 'clean':
    clean()

