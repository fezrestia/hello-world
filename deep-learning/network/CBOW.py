#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plot

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

from layer.MatMul import MatMul
from layer.SoftmaxWithLoss import SoftmaxWithLoss

from optimizer.Adam import Adam

from trainer.Trainer import Trainer

class CBOW:
    def __init__(self, vocab_size, hidden_size):
        V = vocab_size
        H = hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [
            self.in_layer0,
            self.in_layer1,
            self.out_layer,
        ]

        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vs_vec = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) / 2.0
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout = 1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None



# RUN

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "You say goodbye and I say hello."

corpus, word_vs_id, id_vs_word = resource.preprocess(text)

vocab_size = len(word_vs_id)
contexts, target = resource.create_contexts_vs_target(corpus, window_size)
contexts = resource.convert_idx_to_onehot(contexts, vocab_size)
target = resource.convert_idx_to_onehot(target, vocab_size)

model = CBOW(vocab_size, hidden_size)

optimizer = Adam()

trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)

trainer.plot()

word_vs_vec = model.word_vs_vec
for word_id, word in id_vs_word.items():
    print(word, word_vs_vec[word_id])

