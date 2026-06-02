import sys
import numpy as np
import matplotlib.pyplot as plt
import time

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.eval_interval = None
        self.current_epoch = 1

        self.loss_list = []

    def fit(self, x, t, max_epoch = 10, batch_size = 32, max_grad = None, eval_interval = 20):
        data_size = len(x)
        max_iters = data_size // batch_size

        self.eval_interval = eval_interval

        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # shuffle data
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters * batch_size:(iters + 1) * batch_size]
                batch_t = t[iters * batch_size:(iters + 1) * batch_size]

                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                params, grads = remove_duplicate(self.model.params, self.model.grads)

                if max_grad is not None:
                    clip_grads(grads, max_grad)

                self.optimizer.update(dict(enumerate(params)), dict(enumerate(grads)))

                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time

                    print(f"epoch = {self.current_epoch}, iter = {iters + 1}/{max_iters}, elapsed_time = {elapsed_time}, loss = {avg_loss}")

                    self.loss_list.append(float(avg_loss))

                    total_loss = 0.0
                    loss_count = 0

            self.current_epoch += 1

    def plot(self, ylim = None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label = 'train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()



# integrate grads for shared params.
def remove_duplicate(params, grads):
    params = params[:]  # shallow copy
    grads = grads[:]  # shallow copy

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # shared weight
                if params[i] is params[j]:
                    grads[i] += grads[j]  # integrate grads
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # shared weight with transpose (weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T  # integrate grads
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

# clip grads range.
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1.0:
        for grad in grads:
            grad *= rate

