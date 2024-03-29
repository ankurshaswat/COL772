import pickle
import sys
import torch
import gensim

import torch.nn as nn
import torch.nn.functional as F

from model import Net, window_size
from nltk import word_tokenize, pos_tag
from scipy.stats import rankdata

eval_data = sys.argv[1]
eval_data_td = sys.argv[2]
model_path = sys.argv[3]
google_path = sys.argv[4]
output_path = 'output.txt'

restored_dict = torch.load(model_path)

vocab_size = restored_dict['vocab_size']
embedding_dimension = restored_dict['embedding_dimension']
word2idx = restored_dict['word2idx']
idx2word = restored_dict['idx2word']
model_params = restored_dict['model']

model = Net(vocab_size,embedding_dimension)
model.load_state_dict(model_params)
model.eval()


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
# tokens = torch.as_tensor(tokens)
word_options = get_options(eval_data_td)

ranks = []

data_size = len(tokens)

for i in range(0, data_size):
    # print(i, ' out of ', data_size, end='\r')

    input_ = torch.as_tensor([tokens[i]])

    output_ = model(input_)[0]
    # output_ = output_.view(-1)

    # log_softmax = F.log_softmax(output_, dim=0)

    word_probs = []

    for word in word_options[i]:

        if word in word2idx:
            idx = word2idx[word]
        else:
            idx = word2idx['-UNK-']

        # word_probs.append(-1*log_softmax[idx])
        word_probs.append(-1*output_[idx])

    ranks.append(rankdata(word_probs, method='ordinal'))

write_results(output_path, ranks)
