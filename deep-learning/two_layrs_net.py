#!/usr/bin/env python3

import importlib
import sys, os
import numpy as np
import resource

def reload():
    importlib.reload(sys.modules[__name__])

class TwoLayersNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}

        # Init weight
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = resource.sigmoid_func(a1)

        a2 = np.dot(z1, W2) + b2
        y = resource.softmax_func(a2)

        return y

    # x: input
    # t: grand truth
    def loss(self, x, t):
        y = self.predict(x)
        return resource.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    # x: input
    # t: grand truth
    def numerical_gradient(self, x, t):
        # loss func
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        # calculate grad of loss function with current weight params
        grads["W1"] = resource.numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = resource.numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = resource.numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = resource.numerical_gradient(loss_W, self.params["b2"])

        return grads



# RUN

# dummy run
#
# net = TwoLayersNet(input_size = 784, hidden_size = 100, output_size = 10)
#
# print(net.params["W1"].shape)
# print(net.params["b1"].shape)
# print(net.params["W2"].shape)
# print(net.params["b2"].shape)
#
# x = np.random.rand(100, 784)  # dummy img
# t = np.random.rand(100, 10)  # dummy truth
# grads = net.numerical_gradient(x, t)  # dummy grads
# y = net.predict(x)  # dummy result
# print(f"grads[W1].shape = {grads["W1"].shape}")
# print(f"grads[b1].shape = {grads["b1"].shape}")
# print(f"grads[W2].shape = {grads["W2"].shape}")
# print(f"grads[b2].shape = {grads["b2"].shape}")
# print(f"dummy result y.shape = {y.shape}")
# print(f"dummy result y = {y}")


# data
(img_train, label_train), (img_test, label_test) = \
    resource.load_mnist("./mnist", normalize = True, flatten = True, one_hot_label = True)

# hyper params
iters_num = 1000  #10000
train_size = img_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# loop count for each epoch
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayersNet(input_size = 784, hidden_size = 50, output_size = 10)

for i in range(iters_num):
    # get mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    img_batch = img_train[batch_mask]
    label_batch = label_train[batch_mask]

    # calc grad
    grad = network.numerical_gradient(img_batch, label_batch)

    # update params
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # record
    loss = network.loss(img_batch, label_batch)
    train_loss_list.append(loss)

    # evaluate
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(img_train, label_train)
        test_acc = network.accuracy(img_test, label_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"train_acc = {train_acc}, test_acc = {test_acc}")



print(f"train_loss_list = {train_loss_list}")
















