import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import resource

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []

        self.cache = None

        self.ignore_label = -1

    # xs: (N, T, V)
    # ts: (N, label) or (N, V, one-hot)
    def forward(self, xs, ts):
        N, T, V = xs.shape  # N:batch, T:timeline num, V:vocab size

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis = 2)

        mask = (ts != self.ignore_label)

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = resource.softmax_func(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # set loss = 0 for ignore label
        loss = -1.0 * np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout = 1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # (N, label) -> (N, 1, label), timeline = 1

        dx = dx.reshape((N, T, V))

        return dx

