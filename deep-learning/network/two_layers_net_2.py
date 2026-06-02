#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plot
from collections import OrderedDict

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

from layer.Affine import Affine
from layer.Sigmoid import Sigmoid
from layer.SoftmaxWithLoss import SoftmaxWithLoss

from optimizer.StochasticGradientDecent import StochasticGradientDecent

from trainer.Trainer import Trainer

from dataset import spiral

class TwoLayersNet2:
    def __init__(self, input_size, hidden_size, output_size):
        I = input_size
        H = hidden_size
        O = output_size

        # Init weight
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # Layers
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout



# RUN

# hyper params
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()

model = TwoLayersNet2(
        input_size = 2,
        hidden_size = hidden_size,
        output_size = 3,
)

optimizer = StochasticGradientDecent(lr = learning_rate)

trainer = Trainer(model, optimizer)

trainer.fit(x, t, max_epoch, batch_size, eval_interval = 10)

trainer.plot()

