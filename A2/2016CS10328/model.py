import torch
import torch.nn as nn

window_size = 3


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dimension):
        super(Net, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension

        self.embed1 = nn.Embedding(vocab_size, embedding_dimension)
        self.embed1.weight.requires_grad = True

        self.linear1 = nn.Linear(embedding_dimension, vocab_size, bias=False)
        self.linear1.weight.requires_grad = True

    def set_weights(self, initial_embeds):
        # self.embed1.weight = nn.Parameter(initial_embeds)
        self.embed1.weight.data.copy_(initial_embeds)
        # self.linear1.weight = nn.Parameter(initial_embeds)
        self.linear1.weight.data.copy_(initial_embeds)

    def forward(self, x):
        x = self.embed1(x)
        x = torch.sum(x, dim=1)
        x = self.linear1(x)
        return x

    def cust_forward(self, in_model, embeds):
        sum_ = torch.zeros(self.embedding_dimension)
        for i in range(len(in_model)):
            if(in_model[i]):
                sum_ += self.embed1(embeds[i])
            else:
                sum_ += embeds[i]
        x = self.linear1(sum_)
        return x
