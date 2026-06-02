import numpy as np

class MatMul:
    def __init__(self, W):
        self.W = W
        self.params = [self.W]
        self.dW = np.zeros_like(W)
        self.grads = [self.dW]
        self.x = None

    def forward(self, x):
        out = np.dot(x, self.W)
        self.x = x
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x.T, dout)
        self.dW[...] = dW  # deep copy
        return dx

