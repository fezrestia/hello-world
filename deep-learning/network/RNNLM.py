#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plot
import pickle

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

from layer.TimeEmbedding import TimeEmbedding
from layer.TimeRNN import TimeRNN
from layer.TimeAffine import TimeAffine
from layer.TimeSoftmaxWithLoss import TimeSoftmaxWithLoss

from optimizer.StochasticGradientDecent import StochasticGradientDecent

from trainer.RNNLMTrainer import RNNLMTrainer

class RNNLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        rn = np.random.randn

        # weights
        embed_W = (rn(V, D) / 100.0).astype("f")
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype("f")  # Xavier init
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype("f")  # Xavier init
        rnn_b = np.zeros(H).astype("f")
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")  # Xavier init
        affine_b = np.zeros(V).astype("f")

        # layers
        self.layers = [
                TimeEmbedding(embed_W),
                TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful = True),
                TimeAffine(affine_W, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # params/grads
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)

        loss = self.loss_layer.forward(xs, ts)

        return loss

    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()



# RUN
import dataset.ptb as ptb

# hyper params
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5  # for truncated backward propagation time through
learning_rate = 0.1
max_epoch = 100

# data
corpus, word_vs_id, id_vs_word = ptb.load_data("train")

corpus_size = 1000
corpus = corpus[:corpus_size]

vocab_size = int(max(corpus) + 1)  # corpus = index of word id, max = word count

# learning data / grand truth
# data : [1, 2, 3, 4, 5]
#   xs : [1, 2, 3, 4   ]
#   ts : [   2, 3, 4, 5]
# 1->2, 2->3, ...
xs = corpus[:-1]
ts = corpus[1:]
print(f"corpus size = {corpus_size}, vocab size = {vocab_size}")


model = RNNLM(vocab_size, wordvec_size, hidden_size)

optimizer = StochasticGradientDecent(learning_rate)

trainer = RNNLMTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)

trainer.plot()

