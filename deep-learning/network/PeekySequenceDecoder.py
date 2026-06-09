import numpy as np

from layer.TimeEmbedding import TimeEmbedding
from layer.TimeLSTM import TimeLSTM
from layer.TimeAffine import TimeAffine

class PeekySequenceDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype("f")

        # x  : (N, D)
        # Wx : (D, H) -> gated (D, 4H)
        #   -> x x Wx = (N, 4H)
        #
        # h_peek : (N, H) last time sequence hidden vector
        #
        # concat
        #   input x  : [h_peek, x] = (N, H) + (N, D) = (N, H + D)
        #   output h : (N, H + D) x (H + D, 4 x H) = (N, 4 x H) -> slice -> (N, H)
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")

        # input  : output of LSTM, (N, H)
        # h_peek : (N, H) last time sequence hidden vector
        #
        # concat
        #   input  : [h_peak, h] = (N, H) + (N, H) = (N, 2 x H)
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params = []
        self.grads = []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        self.cache = H

        self.lstm.set_state(h)

        # xs : (N, T)
        # embed_W : (V, D)
        #   embed = T (word_id) -> D (word_vec) : V = T -> D
        # out : (N, T, D)
        out = self.embed.forward(xs)
        # h  : (N, H)
        # hs : (N, [H, H, H, ...]) -> (N, T, H)
        hs = np.repeat(h, T, axis = 0).reshape(N, T, H)
        # out : (N, T, H) + (N, T, D) = (N, T, H + D)
        out = np.concatenate((hs, out), axis = 2)

        # input out  : (N, T, H + D)
        # lstm_Wx    : (H + D, 4 x H) -> slice
        # output out : (N, T, H)
        out = self.lstm.forward(out)
        # out : hs (N, T, H) + out (N, T, H) = (N, T, 2 x H)
        out = np.concatenate((hs, out), axis = 2)

        # affine_W : (N, 2 x H, V)
        # score : (N, T, V)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout = dout[:, :, H:]
        dhs0 = dout[:, :, :H]

        dout = self.lstm.backward(dout)
        dembed = dout[:, :, H:]
        dhs1 = dout[:, :, :H]

        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis = 1)

        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))  # 1 batch, 1 char (1, 1)

            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis = 2)
            out = self.lstm.forward(out)

            out = np.concatenate((peeky_h, out), axis = 2)
            score = self.affine.forward(out)  # 1 batch, V char (1, V)

            sample_id = np.argmax(score.flatten())  # V char (V, ) -> to scalar by argmax
            sampled.append(int(sample_id))

        return sampled

