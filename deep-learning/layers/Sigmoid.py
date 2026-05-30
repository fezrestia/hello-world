import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import resource

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = resource.sigmoid_func(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx

