import os, re, json
from flags import FLAGS

data_dir = FLAGS.data_dir

WORD_SPLIT = re.compile("([.,!?\"':;)(])")
DIGIT_RE = re.compile(r"\d")
DUMMY = re.compile('\.|\,|\@')

BOS_id = 0
EOS_id = 1
UNK_id = 2

def tokenizer(sentence):
  words = []
  sentence = DUMMY.sub('', sentence)
  for split_sen in sentence.split():
    words.extend(WORD_SPLIT.split(split_sen))
  return [word for word in words if word]

def read_map(dict_file):
  if os.path.exists(dict_file):
      fp = open(dict_file,'r')
      word_id_dict = json.load(fp)
      print('word number:',len(word_id_dict))

      id_word_dict = [[]]*len(word_id_dict)
      for word in word_id_dict:
          id_word_dict[word_id_dict[word]] = word
  else:
      print('where is dictionary file QQ?')

  return word_id_dict, id_word_dict

def convert_to_token(sentence, vocab_map):
  words = tokenizer(sentence)
  return [vocab_map.get(w.decode('utf8'), UNK_id) for w in words]

def file_to_token(file_path, vocab_map):
  output_path = file_path + '.token'
  with open(file_path, 'r') as input_file:
    with open(output_path, 'w') as output_file:
      counter = 0
      for line in input_file:
        counter += 1
        line = line.strip().split(' +++$+++ ')
        if counter % 100000 == 0:
          print('  Tokenizing line %s' % counter)
        token_ids = convert_to_token(line[1], vocab_map)
        output_file.write(line[0] + ' +++$+++ ' + " ".join([str(tok) for tok in token_ids]) + '\n')

def read_data(path, vocab_map=None, xy=None):
  data = []
  f1 = open(path, 'r')
  if not os.path.exists(path + '.token'):
    file_to_token(path, vocab_map)
  f2 = open(path+'.token', 'r')
  counter = 0
  for l1, l2 in zip(f1, f2):
    counter += 1
    if counter % 100000 == 0:
      print("  Reading data line %s" % counter)
    data_ids = [int(x) for x in l2.split(' +++$+++ ')[1].split()]
    if xy:
      data.append((l2.split(' +++$+++ ')[0], data_ids, l1.strip().split(' +++$+++ ')[1]))
    else:
      data.append((int(l2.split(' +++$+++ ')[0]), data_ids, l1.strip().split(' +++$+++ ')[1]))
  return data

if __name__ == '__main__':
  #form_vocab_mapping(50000)
  #vocab_map, _ = read_map('corpus/mapping')
  #file_to_token('corpus/SAD.csv', vocab_map)
  #d = read_data('corpus/SAD.csv.token')
  #print(d[0])]
  pass

