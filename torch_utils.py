import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import helper

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    words = np.array(words, dtype=np.int)
    n_batches = len(words)//(batch_size*sequence_length)
    words = words[:batch_size*sequence_length*n_batches]
    x = np.array([words[n:n+sequence_length] for n in range(0, len(words)) if (n+sequence_length) <= len(words) ])
    y = np.array([words[n] for n in range(sequence_length, len(words))], dtype=np.int)
    y = np.concatenate((y, [words[0]]), axis=0)
    data = TensorDataset(torch.Tensor(x).long(), torch.Tensor(y).long())
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader



def forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden_dim, clip=9):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    if(torch.cuda.is_available()):
        inputs, labels = inputs.cuda(), labels.cuda()

    #hidden = repackage_hidden(hidden)
    hidden = tuple([each.data for each in hidden_dim])

    rnn.zero_grad()
    optimizer.zero_grad()

    try:
        # get the output from the model
        output, hidden = rnn(inputs, hidden)
    except RuntimeError:
        raise
    # print(labels)
    loss = criterion(output, labels)
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(rnn.parameters(),  clip)
    # for p in rnn.parameters():
        # p.data.add_(-learning_rate, p.grad.data)

    optimizer.step()
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, train_loader, show_every_n_batches=100):
    batch_losses = []

    rnn.train()

    previousLoss = np.Inf
    minLoss = np.Inf

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        print("epoch ",epoch_i)
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            batch_last = batch_i
            n_batches = len(train_loader.dataset) // batch_size
            if(batch_i > n_batches):
                break

            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden, clip=5)
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                average_loss = np.average(batch_losses)
                print('Epoch: {:>4}/{:<4}  Loss: {} Decrease Rate: {} \n'.format(epoch_i, n_epochs, average_loss, (previousLoss - average_loss)))
                if average_loss <= previousLoss:
                    previousLoss = average_loss
                if average_loss <= minLoss:
                    minLoss = average_loss
                    helper.save_model('./save/trained_rnn_new', rnn)
                    print('Model Trained and Saved')
                batch_losses = []

    # returns a trained rnn
    return rnn
