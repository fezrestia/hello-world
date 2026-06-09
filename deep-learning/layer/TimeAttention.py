import numpy as np

from layer.Attention import Attention

class TimeAttention:
    def __init__(self):
        self.params = []
        self.grads = []

        self.layers = None

        self.attention_weights = None

    # hs_enc/dec : (N, T, H)
    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape

        out = np.empty_like(hs_dec)

        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()

            # hs_enc : all encoder hidden vector
            # hs_dec[:, t, :] : each decoder hidden vector
            # out : (N, H)
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])

            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        # out : (N, T, H)
        return out

    def backward(self, dout):
        N, T, H = dout.shape

        dhs_enc = 0.0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec

