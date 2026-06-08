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
from layer.TimeLSTM import TimeLSTM
from layer.TimeAffine import TimeAffine
from layer.TimeSoftmaxWithLoss import TimeSoftmaxWithLoss

from optimizer.StochasticGradientDecent import StochasticGradientDecent

from trainer.RNNLMTrainer import RNNLMTrainer

class LSTMLM:
    def __init__(self, vocab_size = 10000, wordvec_size = 100, hidden_size = 100):
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        rn = np.random.randn

        embed_W = (rn(V, D) / 100.0).astype("f")
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.layers = [
                TimeEmbedding(embed_W),
                TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True),
                TimeAffine(affine_W, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout = 1.0):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

    def save_params(self, file_name = "LSTMLM.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name = "LSTMLM.pkl"):
        with open(file_name, "rb") as f:
            self.params = pickle.load(f)



# RUN
import dataset.ptb as ptb

# hyper params
batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35  # for truncated backward propagation time through
learning_rate = 20.0
max_epoch = 4
max_grad = 0.25

# data
corpus, word_vs_id, id_vs_word = ptb.load_data("train")
corpus_test, _, _ = ptb.load_data("test")

vocab_size = len(word_vs_id)

# learning data / grand truth
# data : [1, 2, 3, 4, 5]
#   xs : [1, 2, 3, 4   ]
#   ts : [   2, 3, 4, 5]
# 1->2, 2->3, ...
xs = corpus[:-1]
ts = corpus[1:]
print(f"corpus size = {corpus.shape}, vocab size = {vocab_size}")


model = LSTMLM(vocab_size, wordvec_size, hidden_size)

optimizer = StochasticGradientDecent(learning_rate)

trainer = RNNLMTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval = 20)

trainer.plot(ylim = (0, 500))



# TEST
model.reset_state()
perplexity_test = resource.eval_perplexity(model, corpus_test)
print(f"test perplexity = {perplexity_test}")

script_dir = Path(__file__).resolve().parent
data_dir = str(script_dir) + "/../dataset/ptb"
pkl_file = f"{data_dir}/lstmlm_params.pkl"
model.save_params(pkl_file)

