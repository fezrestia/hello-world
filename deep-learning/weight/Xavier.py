import numpy as np

class Xavier:
    def __init__(self):
        pass

    def gen(self, in_size, out_size):
        return np.random.randn(in_size, out_size) / np.sqrt(in_size)

