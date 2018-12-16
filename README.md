# Seq2seq Chatbot With Deep Reinforcement Learning

Train the conventional seq2seq model using deep reinforcement learning.
This project is aimed to make Chinese chatbot responses more positive.

- Reward Function:
	- Sentiment Analysis Score: Trying make chatbot's response positive.
	- Coherence Score: To make response suitable for the users' input.

## Prerequisites
1. Python packages:
	- Python 3.5
	- Tensorflow r1.8 or higher
	- Numpy

2. Clone this repository:
```shell=
git clone https://github.com/adfsghjalison/RL_Sentiment_Chatbot.git
```

## Usage

Before training the seq2seq model with reinforcement learning, you need to pre-train the seq2seq model and sentiment analysis model.

### Sentiment Analysis Model:

1. First go to `./sentiment_analysis`

2. Create data directory.  
`mkdir data`  
`mkdir data/data_[database_name]`  

3. Put training data `source_train` and testing data `source_test` to `data/data_[database_name]`.  
format : one data a line  
`[label] +++$+++ [sentence]`

4. Edit flags.py to set names and parameters.  

5. Run:  
`python main.py`

### Pre-train seq2seq model

1. Go to `./`

2. Create data directory  
`mkdir data`  
`mkdir data/data_[database_name]`  

3. Put the training data `chatbot` and testing data `source_test` in `data/data_[database_name]`.  
format : one data pair a line  
`[input sentence] +++$+++ [output sentence]`

4. Pre-train the seq2seq model as the coherence reward function and also as the initialization for the reinforcement learning.  

5. Run:  
`python main.py --mode MLE`  

### Reinforcement Learning

After training sentiment analysis model and pre-training seq2seq model:

Run:
`python main.py --mode RL`

Program will automatically load the pre-trained models and start training seq2seq model using reinforcement learning.

### Test Model

Simply run:
`python run.py --mode TEST`

### Important Hyperparameters of the flags.py
`--vocab_size`: the vocabulary size of the input  
`--hidden_size`: number of units of hidden layer  
`--num_layers`: numbers of the layer  
`--batch_size`: batch size  
`--mode`: mode of the seq2seq model (MLE, RL, TEST)  
`--source_data`: the path of the source file  
`--target_data`: the path of the target file  
`--model_pre_dir`: directory of the pre-trained seq2seq model  
`--model_rl_dir`: direcory of the reinforcement learning seq2seq model  
`--check_step`: step interval of saving model  

`--r1`: weight of reward 1  
`--r2`: weight of reward 2  
`--r3`: weight of reward 3  

## File in this project

### Folders:  
`data/`: store the training data.  
`model/`: store the pre-trained seq2seq model.  
`model_RL/`: store the reinforcement learning seq2seq model.  
`sentiment_analysis/`: the code of sentiement analysis.  

### Files:  
`flags.py`: all settings.  
`data_utils.py`: Data preprocessing (Tokenizer, load data ...etc).  
`seq2seq_model.py`: the core function of the reinforcment learning model.  
`seq2seq.py`: some functions modified from tensorflow source code in order to fit the reinforcement learning algorithm. (only this function is from open source)  
`main.py`: the load, train, and test function of the whole project.  

