import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import resource

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # loss
        self.y = None  # output
        self.t = None  # grand truth (one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = resource.softmax_func(x)
        self.loss = resource.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:  # grand truth is one-hot-vector
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

