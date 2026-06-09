from layer.AttentionWeight import AttentionWeight
from layer.WeightSum import WeightSum

class Attention:
    def __init__(self):
        self.params = []
        self.grads = []

        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()

        self.attention_weight = None

    # hs : (N, T, H)
    # h  : (N, H)
    def forward(self, hs, h):
        # a : (N, T)
        a = self.attention_weight_layer.forward(hs, h)
        # out : (N, H)
        out = self.weight_sum_layer.forward(hs, a)

        self.attention_weight = a

        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

