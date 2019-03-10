#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import gensim
import pickle
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Net,window_size
from os import listdir
from nltk import word_tokenize, pos_tag
from tqdm import tqdm as tqdm
from torch.autograd import Variable

dataset_path = sys.argv[1]
embeddings_path = 'GoogleNews-vectors-negative300.bin'
embedding_dimension = 300
vocab_size = 0
debug_iters = 100
epochs = 2
batch_size = 256


# In[2]:


def load():
    print('Loading Google Model')

    model = gensim.models.KeyedVectors.load_word2vec_format(
        embeddings_path, binary=True)

    print('Generating Vocab')
    word2idx = {}
    idx2word = {}

    idx = 0
    # for word in model.vocab:
    #     word2idx[word] = idx
    #     idx2word[idx] = word
    #     idx += 1

    # trained_words = idx

    old_vocab = model.vocab

    print('Reading new domain files')

    dataset_files = listdir(dataset_path)

    data_tokenized = []

    for file_path in dataset_files:
        with open(dataset_path+'/'+file_path) as file:
            data_tokenized.append(word_tokenize(file.read()))

    print("Replacing proper nouns")
    for i in tqdm(range(len(data_tokenized))):
        token_set = data_tokenized[i]
        datum_pos_tagged = pos_tag(token_set)
        for j in range(len(datum_pos_tagged)):
            tag = datum_pos_tagged[j][1]
            if(tag == 'NNP' or tag == 'NNPS'):
                data_tokenized[i][j] = '-pro-'

    print('Adding new domain tokens')
    for tokens in data_tokenized:
        for token in tokens:
            if (token not in word2idx):
                word2idx[token] = idx
                idx2word[idx] = token
                idx += 1

    vocab_size = idx

    print('Copying old embeddings')
    # embedding_dimension = model[idx2word[0]].shape[0]

    initial_embeds = torch.randn(vocab_size, embedding_dimension)
    for i in range(vocab_size):
        if idx2word[i] in old_vocab:
            initial_embeds[i, :] = torch.as_tensor(model[idx2word[i]])
    # initial_embeds[:trained_words,:] = torch.as_tensor(model[model.vocab])

    print("Creating Training Examples")

    train_examples = []
    target_words = []
    for i in tqdm(range(len(data_tokenized))):
        for j in range(len(data_tokenized[i])):
            for k in range(j-window_size, j+window_size+1):
                if(k < 0 or j == k or k >= len(data_tokenized[i])):
                    continue
                train_examples.append(word2idx[data_tokenized[i][k]])
                target_words.append(word2idx[data_tokenized[i][j]])

    train_examples = torch.as_tensor(train_examples)
    target_words = torch.as_tensor(target_words)

    return word2idx, idx2word, vocab_size, embedding_dimension, initial_embeds, train_examples, target_words


word2idx, idx2word, vocab_size, embedding_dimension, initial_embeds, train_examples, target_words = load()


# In[9]:


print('Creating Model')

model = Net(vocab_size, embedding_dimension)
model.set_weights(initial_embeds)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


db = {}

db['word2idx'] = word2idx
db['idx2word'] = idx2word
db['embedding_dimension'] = embedding_dimension
db['vocab_size'] = vocab_size

dbfile = open('wordIndexes.pkl', 'ab')

pickle.dump(db, dbfile)
dbfile.close()


print('Training')

for epoch in range(epochs):
    total_loss = 0.0
    iter_num = 0
    for i in range(0, len(train_examples), batch_size):
        print(iter_num, end='\r')

        if(i + batch_size > len(train_examples)):
            context_words = train_examples[i:]
            center_word = target_words[i:]
        else:
            context_words = train_examples[i:i+batch_size]
            center_word = target_words[i:i+batch_size]

        input_ = context_words
        output_ = Variable(center_word)

        optimizer.zero_grad()

        # Forward
        outputs = model(input_)

        # Backward
        loss = criterion(outputs, output_)
        loss.backward()

        # Optimize
        optimizer.step()

        total_loss += loss.item()
        if iter_num % debug_iters == debug_iters-1:
            print(total_loss)
            # out = model(torch.tensor(word2idx['the']))
            # log_softmax = F.log_softmax(out,dim=1)
            # _, indices = log_softmax.max(0)
            # print(idx2word[int(indices.numpy())], _, indices)
            total_loss = 0.0
        iter_num += 1

torch.save(model.state_dict(), 'model.pt')
