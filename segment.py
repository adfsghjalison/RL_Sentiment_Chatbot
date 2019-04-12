# coding=utf-8
import jieba
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="input_file", help="input file")
parser.add_argument("--out", dest="output_file", help="output file")
parser.add_argument("--mode", choices=['char', 'word'], dest="mode", help="char or word")
args = parser.parse_args()

mode = args.mode.lower()
# load dictionary
if mode == 'word':
  jieba.load_userdict('dict.txt')

f = open(args.output_file, 'w', encoding='utf8')

for l in open(args.input_file):
  # remove all whitespace characters
  l = ''.join(l.split())
  if mode == 'char':
    f.write(' '.join(list(l)) + '\n')
  else: 
    seg = jieba.cut(l, cut_all=False)
    f.write(' '.join(seg) + '\n')

f.close()

