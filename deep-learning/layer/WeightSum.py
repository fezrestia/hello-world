import numpy as np

class WeightSum:
    def __init__(self):
        self.params = []
        self.grads = []

        self.cache = None

    # hs : (N, T, H)
    # a  : (N, T)
    def forward(self, hs, a):
        N, T, H = hs.shape

        # ar : (N, T, H)
        ar = a.reshape(N, T, 1).repeat(H, axis = 2)

        # t : (N, T, H)
        t = hs * ar

        # c : (N, H)
        c = np.sum(t, axis = 1)

        self.cache = (hs, ar)
        return c

    # dc : (N, H)
    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        # dc.reshape : (N, 1, H)
        #   -> repeat : (N, T, H)
        # dt : (N, T, H)
        dt = dc.reshape(N, 1, H).repeat(T, axis = 1)

        # (N, T, H) o (N, T, H) = (N, T, H)
        dar = dt * hs
        dhs = dt * ar

        # dar : (N, T, H)
        # da  : (N, T)
        da = np.sum(dar, axis = 2)

        return dhs, da

