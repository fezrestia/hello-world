#!/usr/bin/env python3

import importlib
import sys, os
import numpy as np
import resource

def reload():
    importlib.reload(sys.modules[__name__])

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # initial weight

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = resource.softmax_func(z)
        loss = resource.cross_entropy_error(y, t)
        return loss

