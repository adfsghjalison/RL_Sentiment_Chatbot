from flags import * 
import data_utils
import numpy as np
import argparse

parser = argparse.ArgumentParser()
# for word segmentation
parser.add_argument("--source_input", default='corpus/source_input', dest="src_inp", help="source input file for word segmentation")
parser.add_argument("--target_input", default='corpus/target_input', dest="trg_inp", help="target input file for word segmentation")
parser.add_argument("--source_mode", default='word', choices=['char', 'word'], dest="src_mode", help="char or word")
parser.add_argument("--target_mode", default='char', choices=['char', 'word'], dest="trg_mode", help="char or word")
# for dataset splitting (train & val)
parser.add_argument('--source', default=FLAGS.source_data, dest='src', help='source file')
parser.add_argument('--target', default=FLAGS.target_data, dest='trg', help='target file')
args = parser.parse_args()

#step 0. apply word segmentation 
## source
data_utils.word_seg(args.src_inp,args.src,args.src_mode.lower())
## target
data_utils.word_seg(args.trg_inp,args.trg,args.trg_mode.lower())

#step 1. get mapping of whole data
data_utils.prepare_whole_data(FLAGS.source_data, FLAGS.target_data, FLAGS.src_vocab_size, FLAGS.trg_vocab_size, skip_to_token=True)

#step 2. split data into train & val
data_utils.split_train_val(args.src,args.trg)

#step 3. generate tokens of train & val
data_utils.prepare_whole_data(FLAGS.source_data, FLAGS.target_data, FLAGS.src_vocab_size, FLAGS.trg_vocab_size, mode=word_seg_strategy)

#step 4. pretrain fasttext
data_utils.train_fasttext(fasttext_model,source_mapping,fasttext_hkl)
