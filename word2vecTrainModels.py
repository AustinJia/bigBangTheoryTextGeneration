from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
clean_texts = helper.preprocess_words2vector()
path = get_tmpfile("word2vec.model")
# print(common_texts)
modelWV = Word2Vec(clean_texts, size=256, window=5, min_count=1, workers=4)
modelWV.save("word2vec.model")

w2vg = W2VGRU(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, modelWV.wv.syn0)
if train_on_gpu:
    w2vg.cuda()
# set decay, optimizer, and loss
decay_rate = learning_rate / num_epochs
# print(lstmwv.parameters())
optimizer = torch.optim.Adam(w2vg.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay_rate)
criterion = nn.CrossEntropyLoss()

# train the model
trained_rnn = train_rnn(w2vg, batch_size, optimizer, criterion, num_epochs, train_loader, show_every_n_batches, "trained_w2v_gru")
