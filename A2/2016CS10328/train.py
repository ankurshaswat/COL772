import torch
import gensim
import pickle
import sys
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Net, window_size
from os import listdir
from nltk import word_tokenize, pos_tag
from tqdm import tqdm as tqdm
from torch.autograd import Variable

DEBUG = True
DEBUG_ITERS = 100
EPOCHS = 6
BATCH_SIZE = 256

cuda_available = torch.cuda.is_available()
cuda_device = -1

if cuda_available:
    cuda_device = torch.cuda.current_device()
    print("CUDA Available. Using device number ", cuda_device)

dataset_path = sys.argv[1]
model_path = sys.argv[2]
embeddings_path = sys.argv[3]
evaluation_file_path = sys.argv[4]

vocab_size = 0
counts = {}


def log(s):
    if DEBUG:
        print(s)


def load():
    log('Loading Google Model')

    model = gensim.models.KeyedVectors.load_word2vec_format(
        embeddings_path, binary=True)

    embedding_dimension = model['run'].shape[0]

    log('Generating Vocab')
    word2idx = {}
    idx2word = {}

    idx = 0

    old_vocab = model.vocab

    log('Reading new domain files')

    dataset_files = listdir(dataset_path)

    data_tokenized = []

    for file_path in dataset_files:
        with open(dataset_path+'/'+file_path) as file:
            data_tokenized.append(word_tokenize(file.read()))

    log("Replacing proper nouns")
    for i in tqdm(range(len(data_tokenized))):
        token_set = data_tokenized[i]
        datum_pos_tagged = pos_tag(token_set)
        for j in range(len(datum_pos_tagged)):
            tag = datum_pos_tagged[j][1]
            if(tag == 'NNP' or tag == 'NNPS'):
                data_tokenized[i][j] = '-pro-'

    log('Adding new domain tokens')
    for tokens in data_tokenized:
        for token in tokens:
            if (token not in word2idx):
                word2idx[token] = idx
                idx2word[idx] = token
                counts[idx] = 1
                idx += 1
            else:
                counts[word2idx[token]] += 1

    # log('Adding google vocab tokens')
    # for token in model.vocab:
    #     if (token not in word2idx):
    #         word2idx[token] = idx
    #         idx2word[idx] = token
    #         counts[idx] = 0
    #         idx += 1

    for token in ['-PADDING-', '-UNK-']:
        word2idx[token] = idx
        idx2word[idx] = token
        counts[idx] = 1
        idx += 1

    vocab_size = idx

    log('Copying old embeddings')
    initial_embeds = torch.zeros(vocab_size, embedding_dimension)
    for i in range(vocab_size):
        if idx2word[i] in old_vocab:
            initial_embeds[i, :] = torch.as_tensor(model[idx2word[i]])

    log("Creating Training Examples")
    train_examples = []
    target_words = []
    for i in tqdm(range(len(data_tokenized))):
        for j in range(len(data_tokenized[i])):
            context = []
            target_word = word2idx[data_tokenized[i][j]]
            for k in range(j-window_size, j+window_size+1):

                if(k < 0 or j == k or k >= len(data_tokenized[i])):
                    continue

                context.append(word2idx[data_tokenized[i][k]])

            while(len(context) < 2*window_size):
                context.append(word2idx['-PADDING-'])

            train_examples.append(context)
            target_words.append(target_word)

    train_examples = torch.as_tensor(train_examples)
    target_words = torch.as_tensor(target_words)

    return word2idx, idx2word, vocab_size, embedding_dimension, initial_embeds, train_examples, target_words


word2idx, idx2word, vocab_size, embedding_dimension, initial_embeds, train_examples, target_words = load()

log('Creating Model')
model = Net(vocab_size, embedding_dimension)
model.set_weights(initial_embeds)

if cuda_available:
    train_examples = train_examples.to(cuda_device)
    target_words = target_words.to(cuda_device)
    model = model.to(cuda_device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

log("Saving torch model")

log(str(len(train_examples)) + ' training examples in total')

log('Starting Training')

start_epoch = time.time()

for epoch in range(EPOCHS):
    total_loss = 0.0
    start = time.time()
    for i in range(0, len(target_words), BATCH_SIZE):

        if(i + BATCH_SIZE > len(train_examples)):
            context_words = train_examples[i:, :]
            center_word = target_words[i:]
        else:
            context_words = train_examples[i:i+BATCH_SIZE, :]
            center_word = target_words[i:i+BATCH_SIZE]

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
    end = time.time()

    log('Epoch Took ' + str((end-start)/3600) + ". Loss = "+str(total_loss))
    torch.save({'model': model.state_dict(), 'word2idx': word2idx, 'vocab_size': vocab_size,
                'embedding_dimension': embedding_dimension, 'idx2word': idx2word}, model_path)

log('All Took ' + str((time.time()-start_epoch)/3600))
