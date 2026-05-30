import numpy as np

class He:
    def __init__(self):
        pass

    def gen(self, in_size, out_size):
        return np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)

