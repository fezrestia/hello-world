import numpy as np

from layer.Softmax import Softmax

class AttentionWeight:
    def __init__(self):
        self.params = []
        self.grads = []

        self.softmax = Softmax()

        self.cache = None

    # hs : (N, T, H)
    # h  : (N, H)
    def forward(self, hs, h):
        N, T, H = hs.shape

        # h            : (N, H)
        #   -> reshape : (N, 1, H)
        #   -> repeat  : (N, T, H)
        # hr : (N, T, H)
        hr = h.reshape(N, 1, H).repeat(T, axis = 1)

        # t : (N, T, H)
        t = hs * hr

        # s : (N, T)
        s = np.sum(t, axis = 2)

        # a : (N, T)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)

        return a

    # da : (N, T)
    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        # ds : (N, T)
        ds = self.softmax.backward(da)

        # ds.reshape : (N, T, 1)
        #   -> repeat : (N, T, H)
        # dt : (N, T, H)
        dt = ds.reshape(N, T, 1).repeat(H, axis = 2)

        # (N, T, H) o (N, T, H)
        dhs = dt * hr
        dhr = dt * hs

        # dhr : (N, T, H)
        # dh  : (N, H)
        dh = np.sum(dhr, axis = 1)

        return dhs, dh

