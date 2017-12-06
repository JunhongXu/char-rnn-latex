import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class CharRNN(nn.Module):
    def __init__(self, num_chars, hidden_size, n_layers=1, dropout=None):
        super(CharRNN, self).__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # word embedding, given a tensor with N, num_characters, outputs a N, num_characters, hidden_size
        self.encoder = nn.Embedding(num_chars, self.hidden_size)
        # we use LSTM here
        if dropout is None:
            self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                               dropout=dropout)
        self.decoder = nn.Linear(hidden_size, num_chars)

    def forward(self, x, hidden):
        """
            x is a LongTensor with shape batch_size * num_characters
            hidden is a tuple of two FloatTensors with shape batch size * hidden_size
            output is FloatTensor with shape seq_len * batch_size, num_char
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        # batch_size, num_characters, hidden_size
        encode = self.encoder(x)
        output, hidden = self.rnn(encode, hidden)
        output = output.contiguous()
        output = output.view(seq_len * batch_size, -1)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden states using zeros"""
        return (
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda()
        )


if __name__ == '__main__':
    rnn = CharRNN(7, 10)
    rnn.cuda()
    word = Variable(torch.LongTensor([[0, 1, 2], [4, 5, 6]])).cuda()
    hidden = rnn.init_hidden(2)

    output, hidden = rnn(word, hidden)
    print(output.size())
    print(output)