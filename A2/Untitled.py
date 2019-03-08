#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import gensim

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from os import listdir
from nltk import word_tokenize,pos_tag
from tqdm import tqdm as tqdm
from torch.autograd import Variable


# In[2]:


dataset_path = 'dataset'
embeddings_path = 'GoogleNews-vectors-negative300.bin'

def load():
    # In[3]:
    print('Loading Google Model')

    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)  


    # In[4]:

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

    # In[5]:
    print('Reading new domain files')

    dataset_files = listdir(dataset_path)

    data_tokenized = []

    for file_path in dataset_files:
        with open(dataset_path+'/'+file_path) as file:
            data_tokenized.append(word_tokenize(file.read()))


    # In[ ]:


    print("Replacing proper nouns")
    for i in tqdm(range(len(data_tokenized))):
            token_set = data_tokenized[i]
            datum_pos_tagged = pos_tag(token_set)
            for j in range(len(datum_pos_tagged)):
                tag = datum_pos_tagged[j][1]
                if(tag == 'NNP' or tag == 'NNPS'):
                    data_tokenized[i][j] = '-pro-'


    # In[21]:

    print('Adding new domain tokens')
    for tokens in data_tokenized:
        for token in tokens:
            if (token not in word2idx):
                # print(token)
                word2idx[token] = idx
                idx2word[idx] = token
                idx += 1

    vocab_size = idx


    # In[24]:

    print('Copying old embeddings')
    # embedding_dimension = model[idx2word[0]].shape[0]
    embedding_dimension = 300

    initial_embeds = torch.randn(vocab_size,embedding_dimension)
    for i in range(vocab_size):
        if idx2word[i] in old_vocab:
            initial_embeds[i,:] = torch.as_tensor(model[idx2word[i]])
    # initial_embeds[:trained_words,:] = torch.as_tensor(model[model.vocab])
    # context, target
    window_size = 2


    # In[ ]:


    print("Creating Training Examples")

    train_examples = []
    target_words = []
    for i in tqdm(range(len(data_tokenized))):
        for j in range(len(data_tokenized[i])):
            for k in range(j-window_size,j+window_size+1):
                if(k<0 or j==k or k>=len(data_tokenized[i])):
                    continue
                train_examples.append(word2idx[data_tokenized[i][k]])
                target_words.append(word2idx[data_tokenized[i][j]])

    return word2idx,idx2word,vocab_size,embedding_dimension,initial_embeds,train_examples,target_words

word2idx,idx2word,vocab_size,embedding_dimension,initial_embeds,train_examples,target_words = load()

print(vocab_size)
print(initial_embeds.shape)
print(len(train_examples))
print(len(target_words))
# In[ ]:

print('Creating Model')
class Net(nn.Module):
    def __init__(self,initial_embeds):
        super(Net, self).__init__()
        self.embed1 = nn.Embedding(vocab_size,embedding_dimension)
        self.embed1.weight =  nn.Parameter(initial_embeds)
        self.embed1.weight.requires_grad = False
        
        self.linear1 = nn.Linear(vocab_size,embedding_dimension,bias=False)
        self.linear1.weight =  nn.Parameter(initial_embeds)
        self.linear1.weight.requires_grad = True
        
    def forward(self, x):
        x = self.embed1(x)
        x = self.linear1(x)
        return x

net = Net(initial_embeds)



# import pickle
# db = {} 
# db['model'] = net
# db['word2idx'] = word2idx
# db['idx2word'] = idx2word
# db['train'] = train_examples
# db['target'] =  target_words
    
# dbfile = open('pickLLe', 'ab')
    
# pickle.dump(db, dbfile)             
# dbfile.close()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# In[ ]:

print('Training')
# optimizer.zero_grad()
debug_iters = 2000
epochs = 2
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(train_examples)):
        print(i,end='\r')
        context_words = train_examples[i]
        center_word = target_words[i]
        
        input_ = torch.tensor([context_words])
        output_ = Variable(torch.from_numpy(np.array([center_word])).long())

        optimizer.zero_grad()
        
        # Forward
        outputs = net(torch.tensor([input_]))
        # log_softmax = F.log_softmax(outputs)
        
        # Backward
        loss = criterion(outputs,output_)
        # loss = F.nll_loss(log_softmax,output_)
        loss.backward()
        
        # Optimize
        optimizer.step()

        total_loss += loss.item()
        if i % debug_iters == debug_iters-1:
            print(total_loss)
            total_loss = 0.0


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




