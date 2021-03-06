import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class RNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_length = 140

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sig = nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.08
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.lstm.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)

        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        # self.fc.bias.data.zero_()
        self.fc.bias.data.fill_(0)
        # self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.normal_(0.0, (1.0 / np.sqrt(self.fc.in_features)))


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.dropout(self.embedding(nn_input))
        # lstm_out, hidden = self.lstm(embeds.view(len(nn_input), self.window_length, -1), hidden)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        sig_out = self.fc(lstm_out)
        # out = self.fc1(out)
        # print(sig_out.shape)

        sig_out = sig_out.view(batch_size, -1, self.output_size)
        sig_out = sig_out[:, -1]

        # sig_out = F.log_softmax(sig_out, dim=1)

        # return one batch of output word scores and the hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''

        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data

        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden







class Vanilla(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(Vanilla, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_length = 140

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sig = nn.Softmax(dim=1)


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.dropout(self.embedding(nn_input))
        # lstm_out, hidden = self.lstm(embeds.view(len(nn_input), self.window_length, -1), hidden)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        sig_out = self.fc(lstm_out)
        # out = self.fc1(out)
        # print(sig_out.shape)

        sig_out = sig_out.view(batch_size, -1, self.output_size)
        sig_out = sig_out[:, -1]

        # sig_out = F.log_softmax(sig_out, dim=1)

        # return one batch of output word scores and the hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        if (torch.cuda.is_available()):
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())
        else:
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))

        return hidden




class GRU(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(GRU, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_length = 140

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sig = nn.Softmax(dim=1)


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.dropout(self.embedding(nn_input))
        # lstm_out, hidden = self.lstm(embeds.view(len(nn_input), self.window_length, -1), hidden)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        sig_out = self.fc(lstm_out)
        # out = self.fc1(out)
        # print(sig_out.shape)

        sig_out = sig_out.view(batch_size, -1, self.output_size)
        sig_out = sig_out[:, -1]

        # sig_out = F.log_softmax(sig_out, dim=1)

        # return one batch of output word scores and the hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        if (torch.cuda.is_available()):
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())
        else:
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))

        return hidden
class Word2VecModel(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, syn0, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(Word2VecModel, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_length = 140
        self.syn0 = syn0

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sig = nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.08
        self.embedding.weight.data.copy_(torch.from_numpy(self.syn0))
        self.embedding.weight.requires_grad = False
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.lstm.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)

        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        # self.fc.bias.data.zero_()
        self.fc.bias.data.fill_(0)
        # self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.normal_(0.0, (1.0 / np.sqrt(self.fc.in_features)))


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.dropout(self.embedding(nn_input))
        # lstm_out, hidden = self.lstm(embeds.view(len(nn_input), self.window_length, -1), hidden)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        sig_out = self.fc(lstm_out)
        # out = self.fc1(out)
        # print(sig_out.shape)

        sig_out = sig_out.view(batch_size, -1, self.output_size)
        sig_out = sig_out[:, -1]

        # sig_out = F.log_softmax(sig_out, dim=1)

        # return one batch of output word scores and the hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''

        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data

        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
from torch.autograd import Variable
class W2VVanilla(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, syn0, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(W2VVanilla, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_length = 140
        self.syn0 = syn0

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(self.syn0))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sig = nn.Softmax(dim=1)


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.dropout(self.embedding(nn_input))
        # lstm_out, hidden = self.lstm(embeds.view(len(nn_input), self.window_length, -1), hidden)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        sig_out = self.fc(lstm_out)
        # out = self.fc1(out)
        # print(sig_out.shape)

        sig_out = sig_out.view(batch_size, -1, self.output_size)
        sig_out = sig_out[:, -1]

        # sig_out = F.log_softmax(sig_out, dim=1)

        # return one batch of output word scores and the hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        if (torch.cuda.is_available()):
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())
        else:
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))

        return hidden

class W2VGRU(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, syn0, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(W2VGRU, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_length = 140
        self.syn0 = syn0

        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(self.syn0))
        self.lstm = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sig = nn.Softmax(dim=1)


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        embeds = self.dropout(self.embedding(nn_input))
        # lstm_out, hidden = self.lstm(embeds.view(len(nn_input), self.window_length, -1), hidden)

        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        sig_out = self.fc(lstm_out)
        # out = self.fc1(out)
        # print(sig_out.shape)

        sig_out = sig_out.view(batch_size, -1, self.output_size)
        sig_out = sig_out[:, -1]

        # sig_out = F.log_softmax(sig_out, dim=1)

        # return one batch of output word scores and the hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        if (torch.cuda.is_available()):
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())
        else:
            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))

        return hidden
