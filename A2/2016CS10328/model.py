import torch.nn as nn

window_size = 2

class Net(nn.Module):
    def __init__(self,vocab_size,embedding_dimension):
        super(Net, self).__init__()
        self.embed1 = nn.Embedding(vocab_size,embedding_dimension)
        self.embed1.weight.requires_grad = True

        self.linear1 = nn.Linear(embedding_dimension,vocab_size,bias=False)
        self.linear1.weight.requires_grad = True

    def set_weights(self,initial_embeds):
        self.embed1.weight =  nn.Parameter(initial_embeds)
        self.linear1.weight =  nn.Parameter(initial_embeds)

    def forward(self, x):
        x = self.embed1(x)
        x = self.linear1(x)
        return x
