import numpy as np

from sampler.UnigramSampler import UnigramSampler
from layer.SigmoidWithLoss import SigmoidWithLoss
from layer.EmbeddingDot import EmbeddingDot

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power = 0.75, sample_size = 5):
        self.sample_size = sample_size

        self.sampler = UnigramSampler(corpus, power, sample_size)

        # for negative sample + 1 (target = index 0)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params = []
        self.grads = []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]

        # for target
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype = np.int32)  # all 1 = OK
        loss = self.loss_layers[0].forward(score, correct_label)

        # for negative samples
        negative_sample = self.sampler.get_negative_sample(target)
        negative_label = np.zeros(batch_size, dtype = np.int32)  # all 0 = NG
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]  # axis0 = batch
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout = 1):
        dh = 0
        for loss_layer, embed_dot_layer in zip(self.loss_layers, self.embed_dot_layers):
            dscore = loss_layer.backward(dout)
            dh += embed_dot_layer.backward(dscore)

        return dh

