#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plot
import pickle

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

from layer.MatMul import MatMul
from layer.SoftmaxWithLoss import SoftmaxWithLoss
from layer.Embedding import Embedding
from layer.NegativeSamplingLoss import NegativeSamplingLoss

from optimizer.Adam import Adam

from trainer.Trainer import Trainer

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V = vocab_size
        H = hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # layers
        self.in_layers = []
        for i in range(2 * window_size):  # +/- direction
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power = 0.75, sample_size = 5)

        layers = self.in_layers + [self.ns_loss]
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vs_vec = W_in

    def forward(self, contexts, target):
        h = 0.0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1.0 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout = 1):
        dout = self.ns_loss.backward(dout)
        dout *= 1.0 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None



# RUN
import dataset.ptb as ptb

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

corpus, word_vs_id, id_vs_word = ptb.load_data("train")

vocab_size = len(word_vs_id)

contexts, target = resource.create_contexts_vs_target(corpus, window_size)

model = CBOW(vocab_size, hidden_size, window_size, corpus)

optimizer = Adam()

trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)

trainer.plot()

word_vs_vec = model.word_vs_vec

params = {}
params["word_vs_vec"] = word_vs_vec.astype(np.float16)
params["word_vs_id"] = word_vs_id
params["id_vs_word"] = id_vs_word

script_dir = Path(__file__).resolve().parent
data_dir = str(script_dir) + "/../dataset/ptb"
pkl_file = f"{data_dir}/cbow_params.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)

