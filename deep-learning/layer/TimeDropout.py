import numpy as np

class TimeDropout:
    def __init__(self, dropout_ratio = 0.5):
        self.params = []
        self.grads = []

        self.dropout_ratio = dropout_ratio

        self.mask = None
        self.is_training = True

    def forward(self, xs):
        if self.is_training:
            # rand : [0.0 - 1.0], True/False mask.
            is_dropped_mask = np.random.rand(*xs.shape) > self.dropout_ratio

            scale = 1 / (1.0 - self.dropout_ratio)

            self.mask = is_dropped_mask.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask

