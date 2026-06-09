#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plot

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

from layer.TimeSoftmaxWithLoss import TimeSoftmaxWithLoss
from network.SequenceEncoder import SequenceEncoder
from network.SequenceDecoder import SequenceDecoder
from network.PeekySequenceDecoder import PeekySequenceDecoder
from network.AttentionEncoder import AttentionEncoder
from network.AttentionDecoder import AttentionDecoder

from optimizer.Adam import Adam

from trainer.Trainer import Trainer

class Seq2Seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V = vocab_size
        D = wordvec_size
        H = hidden_size

        # self.encoder = SequenceEncoder(V, D, H)
        self.encoder = AttentionEncoder(V, D, H)

        # self.decoder = SequenceDecoder(V, D, H)
        # self.decoder = PeekySequenceDecoder(V, D, H)
        self.decoder = AttentionDecoder(V, D, H)

        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs = ts[:, :-1]  # (N, x) _1234
        decoder_ts = ts[:, 1:]  # (N, x) 1234

        h = self.encoder.forward(xs)

        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)

        return loss

    def backward(self, dout = 1.0):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled



# RUN
if __name__ == "__main__":
    import dataset.sequence as sequence

    # data
    (x_train, t_train), (x_test, t_test) = sequence.load_data("date.txt")
    char_vs_id, id_vs_char = sequence.get_vocab()

    # reverse data
    x_train = x_train[:, ::-1]  # all batch, from start to end by step = -1
    x_test = x_test[:, ::-1]

    # hyper params
    vocab_size = len(char_vs_id)
    wordvec_size = 16
    hidden_size = 256
    batch_size = 128
    max_epoch = 10
    max_grad = 5.0

    model = Seq2Seq(vocab_size, wordvec_size, hidden_size)

    optimizer = Adam()

    trainer = Trainer(model, optimizer)


    accurate_list = []
    for epoch in range(max_epoch):
        trainer.fit(
                x_train,
                t_train,
                max_epoch = 1,
                batch_size = batch_size,
                max_grad = max_grad,
        )

        correct_count = 0
        for i in range(len(x_test)):
            question = x_test[[i]]
            correct = t_test[[i]]

            verbose = i < 10

            correct_count += resource.eval_seq2seq(model, question, correct, id_vs_char, verbose)

        accurate = float(correct_count) / len(x_test)
        accurate_list.append(accurate)

        print(f"val accurate = {accurate}")

