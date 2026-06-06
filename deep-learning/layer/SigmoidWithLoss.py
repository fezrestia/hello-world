import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import resource

class SigmoidWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []

        self.loss = None

        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        # P(OK)     = [a, b, c]
        # P(1 - OK) = [x, y, z]
        # cross_entropy_error input is below.
        #       [ [OK, 1-OK],
        #         [OK, 1-OK],
        #         ...
        # [P(OK), P(1-OK)] = [ [a, b, c], [x, y, z] ]
        # np.c_[P(OK), P(1-OK)] = [ [a, x], [b, y], [c, z] ]
        self.loss = resource.cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

