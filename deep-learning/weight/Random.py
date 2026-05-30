import numpy as np

class Random:
    def __init__(self, scale):
        self.scale = scale

    def gen(self, in_size, out_size):
        return np.random.randn(in_size, out_size) * self.scale

