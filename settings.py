import json
# for data etl
SEED = 112
buckets = [(10, 10), (15, 15), (25, 25), (50, 50)]
split_ratio = 0.9

# for inference filter dirty words
with open('replace_words.json','r') as f:
    replace_words = json.load(f)

# for reset schedule sampling probability
reset_prob = 1.0

# word size upper bound
WORD_DIM = 60000 

# apply same word segment strategy to both source and target or not
word_seg_strategy = 'diff'
