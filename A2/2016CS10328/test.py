import pickle
import sys
import torch
import gensim

import torch.nn as nn
import torch.nn.functional as F

from model import Net, window_size
from nltk import word_tokenize, pos_tag
from scipy.stats import rankdata

BATCH_SIZE = 1

# cuda_available = torch.cuda.is_available()
# cuda_device = -1

# if cuda_available:
#     print("Cuda availalbe")
#     cuda_device = torch.cuda.current_device()
#     print("CUDA device = ", cuda_device)

eval_data = sys.argv[1]
eval_data_td = sys.argv[2]
output_path = 'output.txt'

dbfile = open(sys.argv[3] + '/wordIndexes.pkl', 'rb')
db = pickle.load(dbfile)

google_path = sys.argv[4]

word2idx = db['word2idx']
idx2word = db['idx2word']
vocab_size = db['vocab_size']
embedding_dimension = db['embedding_dimension']

dbfile.close()

model = Net(vocab_size, embedding_dimension)
model.load_state_dict(torch.load(sys.argv[3] + '/model.pt'))
model.eval()


def load_google_model(fp):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        fp, binary=True)
    return model

# google_model = load_google_model(google_path)


def get_sentence(fp):
    window_tokens = []
    for line in open(fp, 'r'):
        tokens = []
        line = line.strip('\n')
        sentence = line.split('::::')[0]
        sentences = sentence.split('<<target>>')
        pre_tokens = word_tokenize(sentences[0])
        post_tokens = word_tokenize(sentences[1])
        for i in range(-1*window_size, 0):
            if(-1*i > len(pre_tokens)):
                idx = word2idx['-PADDING-']
            else:
                if pre_tokens[i] in word2idx:
                    idx = word2idx[pre_tokens[i]]
                else:
                    idx = word2idx['-UNK-']
            tokens.append(idx)

        for i in range(0, window_size):
            if(i >= len(post_tokens)):
                idx = word2idx['-PADDING-']
            else:
                if post_tokens[i] in word2idx:
                    idx = word2idx[post_tokens[i]]
                else:
                    idx = word2idx['-UNK-']
            tokens.append(idx)
        window_tokens.append(tokens)
    return window_tokens


def get_sentence_with_google(fp):
    window_tokens = []
    in_model = []
    for line in open(fp, 'r'):
        tokens = []
        in_model_row = []
        line = line.strip('\n')
        sentence = line.split('::::')[0]
        sentences = sentence.split('<<target>>')
        pre_tokens = word_tokenize(sentences[0])
        post_tokens = word_tokenize(sentences[1])
        for i in range(-1*window_size, 0):
            if(-1*i > len(pre_tokens)):
                idx = word2idx['-PADDING-']
                in_model_row.append(True)
                tokens.append(idx)
            else:
                if pre_tokens[i] in word2idx:
                    idx = word2idx[pre_tokens[i]]
                    in_model_row.append(True)
                    tokens.append(idx)
                elif pre_tokens[i] in google_model.vocab:
                    tokens.append(google_model[pre_tokens[i]])
                    in_model_row.append(False)
                else:
                    idx = word2idx['-UNK-']
                    in_model_row.append(True)
                    tokens.append(idx)

        for i in range(0, window_size):
            if(i >= len(post_tokens)):
                idx = word2idx['-PADDING-']
                in_model_row.append(True)
                tokens.append(idx)
            else:
                if post_tokens[i] in word2idx:
                    idx = word2idx[post_tokens[i]]
                    in_model_row.append(True)
                    tokens.append(idx)
                elif post_tokens[i] in google_model.vocab:
                    tokens.append(google_model[post_tokens[i]])
                    in_model_row.append(False)
                else:
                    idx = word2idx['-UNK-']
                    in_model_row.append(True)
                    tokens.append(idx)
        tokens = list(map(torch.as_tensor, tokens))
        window_tokens.append(tokens)
        in_model.append(in_model_row)

    return in_model, window_tokens


def get_options(fp):
    td = []
    for line in open(fp, 'r'):
        line = line.strip('\n')
        td.append(line.split())
    return td


def write_results(fp, ranks):
    with open(fp, 'w') as file:
        for row in ranks:
            file.write(' '.join(str(int(v)) for v in row))
            file.write('\n')


tokens = get_sentence(eval_data)
tokens = torch.as_tensor(tokens)
# in_model, tokens = get_sentence_with_google(eval_data)
word_options = get_options(eval_data_td)

# if cuda_available:
    # tokens = tokens.to(cuda_device)
    # model = model.to(cuda_device)

ranks = []

data_size = len(tokens)

for i in range(0, data_size):

    batch = [tokens[i]]

    # if(i + BATCH_SIZE > data_size):
    #     batch = tokens[i:, :]
    #     limit = data_size
    # else:
    #     batch = tokens[i:i+BATCH_SIZE, :]
    #     limit = i+BATCH_SIZE
    input_ = torch.as_tensor(batch)
    output_ = model(input_)
    # output_ = model.cust_forward(in_model[i], tokens[i])
    log_softmax = F.log_softmax(output_, dim=0)
    # for j in range(i, limit):
    print(i, ' out of ', data_size, end='\r')
    word_probs = []
        # loc_word_options = word_options[j]
        # probs = log_softmax[j % BATCH_SIZE]
    for word in word_options[i]:
        if word in word2idx:
            idx = word2idx[word]
        else:
            idx = word2idx['-UNK-']
        # word_probs.append(-1*probs[idx])
        word_probs.append(-1*log_softmax[idx])
    ranks.append(rankdata(word_probs, method='ordinal'))
write_results(output_path, ranks)
