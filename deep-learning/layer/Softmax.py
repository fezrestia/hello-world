import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import resource

class Softmax:
    def __init__(self):
        self.params = []
        self.grads = []

        self.out = None

    def forward(self, x):
        self.out = resource.softmax_func(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis = 1, keepdims = True)
        dx -= self.out * sumdx
        return dx

