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
import pickle


dbfile = open('pickLLe', 'rb')      
db = pickle.load(dbfile)

net = db['model'] 
word2idx = db['word2idx'] 
idx2word = db['idx2word'] 
train_examples = db['train'] 
target_words = db['target'] 

dbfile.close()


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[ ]:

print('Training')
optimizer.zero_grad()
debug_iters = 2000
epochs = 2
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(train_examples)):
        context_words = train_examples[i]
        center_word = target_words[i]
        
        input_ = torch.tensor([context_words])
        output_ = Variable(torch.from_numpy(np.array([center_word])).long())

        
        # Forward
        outputs = net(torch.tensor([input_]))
        log_softmax = F.log_softmax(outputs)
        
        # Backward
        loss = F.nll_loss(log_softmax,output_)
        loss.backward()
        
        # Optimize
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if i % debug_iters == debug_iters-1:
            print(total_loss)
            total_loss = 0.0
