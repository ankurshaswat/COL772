#!/usr/bin/env python
# coding: utf-8

import pickle
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

from model import Net,window_size
from nltk import word_tokenize, pos_tag

eval_data = sys.argv[1]
eval_data_td = sys.argv[2]
output = 'output.txt'

dbfile = open('wordIndexes.pkl', 'rb')      
db = pickle.load(dbfile)

word2idx = db['word2idx'] 
idx2word = db['idx2word']
vocab_size = db['vocab_size']
embedding_dimension = db['embedding_dimension']

dbfile.close()

model = Net(vocab_size,embedding_dimension)
model.load_state_dict(torch.load('model.pt'))
model.eval()

def get_sentence(fp):
    window_tokens = []
    for line in open(fp,'r'):
        tokens = []
        line = line.strip('\n')
        sentence = line.split('::::')[0]
        sentences = sentence.split('<<target>>')
        pre_tokens = word_tokenize(sentences[0])
        post_tokens = word_tokenize(sentences[1])
        for i in range(-1*window_size-1,0):
            tokens.append(word2idx[pre_tokens[i]])
        for i in range(0,window_size):
            tokens.append(word2idx[post_tokens[i]])
        # for token in pre_tokens:
        window_tokens.append(tokens)
        # targets.append(target)

    return window_tokens

def get_options(fp):
    td = []
    for line in open(fp,'r'):
        line = line.strip('\n')
        td.append(line.split())
    return td

def write_results(fp,ranks):
    with open(fp,'w') as file:
        for row in ranks:
            file.write(' '.join(row))
            file.write('\n')

def rank(inp):
    return sorted(range(len(inp)), key=inp.__getitem__)

tokens = get_sentence(eval_data)
word_options = get_options(eval_data_td)

ranks = []

for i in range(len(tokens)):
    row = tokens[i]
    input_ = torch.as_tensor(row)
    outputs = model(input_)
    summation = torch.sum(outputs, dim=0) # size = [1, ncol]
    log_softmax = F.log_softmax(summation,dim=1)
    # all_ranks = rank(log_softmax)
    word_ranks = []
    for word in word_options[i]:
        idx = word2idx[word]
        word_ranks.append(log_softmax[idx])
    word_ranks = rank(word_ranks)
    ranks.append(word_ranks)

write_results(output,ranks)
