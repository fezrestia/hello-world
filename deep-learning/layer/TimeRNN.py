import numpy as np

from layer.RNN import RNN

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful = False):
        self.params = [Wx, Wh, b]
        self.grads = [
                np.zeros_like(Wx),
                np.zeros_like(Wh),
                np.zeros_like(b),
        ]

        self.layers = None

        self.h = None
        self.dh = None

        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    # dx : input
    def forward(self, xs):
        Wx, Wh, b = self.params

        N, T, D = xs.shape  # N:batch, T:timeline num, D:x dim
        D, H = Wx.shape  # D:x dim, H:hidden dim

        self.layers = []

        # hs : output
        hs = np.empty((N, T, H), dtype = "f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype = "f")

        for t in range(T):
            layer = RNN(*self.params)

            # xs[:, t, :] : all batch, same timeline sequence num, all x dim.
            self.h = layer.forward(xs[:, t, :], self.h)

            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    # dhs : d output of forward
    def backward(self, dhs):
        Wx, Wh, b = self.params

        N, T, H = dhs.shape  # N:batch, T:timeline num, H:hidden dim
        D, H = Wx.shape  # D:x dim, H:hidden dim

        dxs = np.empty((N, T, D), dtype = "f")

        dh = 0.0
        grads = [0.0, 0.0, 0.0]
        for t in reversed(range(T)):
            layer = self.layers[t]

            # input of backward is dh_next + dh
            dx, dh = layer.backward(dhs[:, t, :] + dh)

            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh

        return dxs

