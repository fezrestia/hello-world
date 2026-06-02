import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import resource

class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []

        self.y = None  # output
        self.t = None  # grand truth (one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = resource.softmax_func(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis = 1)

        loss = resource.cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

