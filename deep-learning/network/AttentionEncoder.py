from network.SequenceEncoder import SequenceEncoder

class AttentionEncoder(SequenceEncoder):
    # xs : (N, T)
    def forward(self, xs):
        # xs : N, T, D
        xs = self.embed.forward(xs)
        # hs : N, T, H
        hs = self.lstm.forward(xs)

        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

