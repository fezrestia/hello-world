import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

class RNNLMTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype = "i")
        batch_t = np.empty((batch_size, time_size), dtype = "i")

        data_size = len(x)

        # read start pos for each batch
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1

        return batch_x, batch_t

    def fit(
            self,
            xs,
            ts,
            max_epoch = 10,
            batch_size = 20,
            time_size = 35,
            max_grad = None,
            eval_interval = 20,
    ):
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval

        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)

        model = self.model
        optimizer = self.optimizer

        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                loss = model.forward(batch_x, batch_t)
                model.backward()

                params, grads = resource.remove_duplicate(model.params, model.grads)

                if max_grad is not None:
                    resource.clip_grads(grads, max_grad)

                optimizer.update(dict(enumerate(params)), dict(enumerate(grads)))

                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    perplexity = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(f"epoch = {epoch}, iter = {iter}, time = {elapsed_time}, perplexity = {perplexity}")

                    self.ppl_list.append(float(perplexity))

                    total_loss = 0.0
                    loss_count = 0

            self.current_epoch += 1

    def plot(self, ylim = None):
        x = np.arange(len(self.ppl_list))

        if ylim is not None:
            plt.ylim(*ylim)

        plt.plot(x, self.ppl_list, label = "train")
        plt.xlabel(f"iteration {str(self.eval_interval)}")
        plt.ylabel("perplexity")

        plt.show()

