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
from layer.TimeDropout import TimeDropout

from optimizer.StochasticGradientDecent import StochasticGradientDecent

from trainer.RNNLMTrainer import RNNLMTrainer

class DeepLSTMLM:
    def __init__(
            self,
            vocab_size = 10000,
            wordvec_size = 650,
            hidden_size = 650,
            dropout_ratio = 0.5):
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        rn = np.random.randn

        embed_W = (rn(V, D) / 100.0).astype("f")
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b1 = np.zeros(4 * H).astype("f")
        lstm_Wx2 = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b2 = np.zeros(4 * H).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.layers = [
                TimeEmbedding(embed_W),
                TimeDropout(dropout_ratio),
                TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful = True),
                TimeDropout(dropout_ratio),
                TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful = True),
                TimeDropout(dropout_ratio),
                # input : V->D, output : D->V
                TimeAffine(embed_W.T, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [
                self.layers[2],
                self.layers[4],
        ]
        self.drop_layers = [
                self.layers[1],
                self.layers[3],
                self.layers[5],
        ]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, is_training = False):
        for layer in self.drop_layers:
            layer.is_training = is_training

        for layer in self.layers:
            xs = layer.forward(xs)

        return xs

    def forward(self, xs, ts, is_training = True):
        score = self.predict(xs, is_training)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout = 1.0):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

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
wordvec_size = 650
hidden_size = 650
time_size = 35  # for truncated backward propagation time through
learning_rate = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

# data
corpus, word_vs_id, id_vs_word = ptb.load_data("train")
corpus_test, _, _ = ptb.load_data("test")
corpus_val, _, _ = ptb.load_data("valid")

vocab_size = len(word_vs_id)

# learning data / grand truth
# data : [1, 2, 3, 4, 5]
#   xs : [1, 2, 3, 4   ]
#   ts : [   2, 3, 4, 5]
# 1->2, 2->3, ...
xs = corpus[:-1]
ts = corpus[1:]
print(f"corpus size = {corpus.shape}, vocab size = {vocab_size}")


model = DeepLSTMLM(vocab_size, wordvec_size, hidden_size, dropout)

optimizer = StochasticGradientDecent(learning_rate)

trainer = RNNLMTrainer(model, optimizer)


script_dir = Path(__file__).resolve().parent
data_dir = str(script_dir) + "/../dataset/ptb"
pkl_file = f"{data_dir}/lstmlm_params.pkl"

best_perplexity = float("inf")
for epoch in range(max_epoch):
    trainer.fit(
            xs,
            ts,
            max_epoch = 1,
            batch_size = batch_size,
            time_size = time_size,
            max_grad = max_grad,
    )

    model.reset_state()
    perplexity_val = resource.eval_perplexity(model, corpus_val)
    print(f"valid perplexity = {perplexity_val}")

    if best_perplexity > perplexity_val:
        best_perplexity = perplexity_val
        model.save_params
    else:
        learning_rate /= 4.0
        optimizer.learning_rate = learning_rate

    model.reset_state()
    print(f"----------------------------------------------------------------")



# TEST
model.reset_state()
perplexity_test = resource.eval_perplexity(model, corpus_test)
print(f"test perplexity = {perplexity_test}")

#script_dir = Path(__file__).resolve().parent
#data_dir = str(script_dir) + "/../dataset/ptb"
#pkl_file = f"{data_dir}/lstmlm_params.pkl"
#model.save_params(pkl_file)

trainer.plot(ylim = (0, 500))

