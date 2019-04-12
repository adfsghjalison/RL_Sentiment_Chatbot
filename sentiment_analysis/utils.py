import os, re, json
from .flags import FLAGS

data_dir = FLAGS.data_dir

BOS_id = 0
EOS_id = 1
UNK_id = 2

def read_map(dict_file):
  #if os.path.exists(dict_file):
  try:
      fp = open(dict_file,'r')
      word_id_dict = json.load(fp)
      print('word number:',len(word_id_dict))

      id_word_dict = [[]]*len(word_id_dict)
      for word in word_id_dict:
          id_word_dict[word_id_dict[word]] = word
      return word_id_dict, id_word_dict
  except:
  #else:
      print('where is dictionary file QQ?')
      raise

def convert_to_token(sen, vocab_map):
  #w = w.decode('utf8')
  return [vocab_map.get(w, UNK_id) for w in sen.split()]

def file_to_token(file_path, vocab_map, skip=True):
  output_path = file_path + '.token'
  with open(file_path, 'r') as input_file:
    with open(output_path, 'w') as output_file:
      counter = 0
      for l in input_file:
        counter += 1
        l = l.strip().split(' +++$+++ ')
        if counter % 100000 == 0:
          print('  Tokenizing line %s' % counter)

        # skip unknown label
        if skip and int(l[0]) == -1:
          continue
          
        token_ids = convert_to_token(l[1], vocab_map)
        output_file.write('{} +++$+++ {} +++$+++ {}\n'.format(l[0], l[1], " ".join([str(tok) for tok in token_ids])))

def read_data(path, vocab_map=None, xy=None, skip=True):
  data = []
  if not os.path.exists(path + '.token'):
    file_to_token(path, vocab_map, skip=skip)
  f = open(path+'.token', 'r')
  counter = 0
  for l in f:
    counter += 1
    if counter % 100000 == 0:
      print("  Reading data line %s" % counter)
    l = l.strip().split(' +++$+++ ')
    if len(l)<3:
      print (l)
    data_ids = [int(x) for x in l[2].split()]
    if xy:
      data.append((l[0], data_ids, l[1]))   # x, id_y, y
    else:
      data.append((int(l[0]), data_ids, l[1]))  #label, id, x
  return data

if __name__ == '__main__':
  #form_vocab_mapping(50000)
  #vocab_map, _ = read_map('corpus/mapping')
  #file_to_token('corpus/SAD.csv', vocab_map)
  #d = read_data('corpus/SAD.csv.token')
  #print(d[0])]
  pass

