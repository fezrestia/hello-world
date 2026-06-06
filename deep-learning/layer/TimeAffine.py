import numpy as np

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [
                np.zeros_like(W),
                np.zeros_like(b),
        ]

        self.x = None

    def forward(self, x):
        self.x = x
        W, b = self.params

        N, T, D = x.shape  # N:batch, T:timeline num, D:x dim

        rx = x.reshape(N * T, -1)

        out = np.dot(rx, W) + b
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        W, b = self.params

        N, T, D = x.shape

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis = 0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

