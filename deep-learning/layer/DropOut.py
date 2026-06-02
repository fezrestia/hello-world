import numpy as np

class DropOut:
    def __init__(self, drop_out_ratio = 0.001):
        self.params = []
        self.grads = []

        self.drop_out_ratio = drop_out_ratio
        self.mask = None

    def forward(self, x, is_training = True):
        if is_training:
            self.mask = np.random.rand(*x.shape) > self.drop_out_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.drop_out_ratio)

    def backward(self, dout):
        return dout * self.mask

