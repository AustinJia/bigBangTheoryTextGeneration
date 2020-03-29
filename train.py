import helper
import numpy as np
import torch
import torch.nn as nn
from torch_utils import batch_data, train_rnn
from models import RNN, Vanilla, GRU
import matplotlib.pyplot as plt

# hyperparameters
sequence_length = 6
batch_size = 128
num_epochs = 10
learning_rate = 0.002
embedding_dim = 256
hidden_dim = 256
n_layers = 2
show_every_n_batches = 500


# load data
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
train_loader = batch_data(int_text, sequence_length, batch_size)
vocab_size = len(vocab_to_int)
output_size = len(vocab_to_int)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')

# create model and move to gpu if available
# rnn = GRU(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.25)
rnn = Vanilla(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# set decay, optimizer, and loss
decay_rate = learning_rate / num_epochs
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay_rate)
criterion = nn.CrossEntropyLoss()

# train the model
saved_model_name = 'trained_rnn'
trained_rnn, loss_history = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, train_loader, show_every_n_batches, saved_model_name)

plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()
