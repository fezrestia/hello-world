import numpy as np

from layer.Embedding import Embedding

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [
                np.zeros_like(W),
        ]

        self.layers = None

    # xs : N x word idx
    def forward(self, xs):
        W, = self.params

        N, T = xs.shape  # N:batch, T:timeline num
        V, D = W.shape  # V:vocab count, D:x dim

        out = np.empty((N, T, D), dtype = "f")

        self.layers = []

        for t in range(T):
            layer = Embedding(W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad

        return None

