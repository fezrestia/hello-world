#!/usr/bin/env python3

import importlib
import sys, os
import numpy as np
import matplotlib.pyplot as plot
from collections import OrderedDict
import re

import resource

from layers.Affine import Affine
from layers.Relu import Relu
from layers.SoftmaxWithLoss import SoftmaxWithLoss
from layers.BatchNormalization import BatchNormalization
from layers.DropOut import DropOut

from optimizer.StochasticGradientDecent import StochasticGradientDecent
from optimizer.Momentum import Momentum
from optimizer.AdaGrad import AdaGrad
from optimizer.Adam import Adam

from weight.Random import Random
from weight.Xavier import Xavier
from weight.He import He

class DeepLayerNet:
    def __init__(
            self,
            input_size,
            hidden_size_list,
            output_size,
            weight = Random(0.01),
            weight_decay_lambda = 0.0,
            drop_out_ratio = 0.0,
    ):
        connection_size_list = np.concatenate([[input_size], hidden_size_list, [output_size]])
        self.layer_num = len(connection_size_list) - 1  # input->hidden1, hidden1->hidden2, ..., hiddenx->output

        # Params
        self.params = OrderedDict()
        for i in range(self.layer_num):
            self.params[f"W{i}"] = weight.gen(connection_size_list[i], connection_size_list[i + 1])
            self.params[f"b{i}"] = np.zeros(connection_size_list[i + 1])

            self.params[f"gamma{i}"] = np.ones(connection_size_list[i + 1])
            self.params[f"beta{i}"] = np.zeros(connection_size_list[i + 1])

        # Layers
        self.layers = OrderedDict()

        # input/hidden
        for i in range(self.layer_num):
            layer = Affine(self.params[f"W{i}"], self.params[f"b{i}"])
            self.layers[f"Affine{i}"] = layer

            self.layers[f"BatchNorm{i}"] = BatchNormalization(self.params[f"gamma{i}"], self.params[f"beta{i}"])

            if i < (self.layer_num - 1):
                # not output layer
                self.layers[f"Relu{i}"] = Relu()

                self.layers[f"DropOut{i}"] = DropOut(drop_out_ratio)

        # output
        self.output_layer = SoftmaxWithLoss()

        self.weight_decay_lambda = weight_decay_lambda

    def predict(self, x, is_training = False):
        for key in self.layers:
            layer = self.layers[key]
            if isinstance(layer, BatchNormalization) or isinstance(layer, DropOut):
                x = layer.forward(x, is_training)
            else:
                x = layer.forward(x)
        return x

    # x: input
    # t: grand truth
    def loss(self, x, t, is_training = False):
        y = self.predict(x, is_training)

        weight_decay = 0
        for key in self.params:
            if re.search(r"^W\d+$", key):
                W = self.params[key]
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.output_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, is_training = False)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    # x: input
    # t: grand truth
    def numerical_gradient(self, x, t):
        # loss func
        loss_W = lambda W: self.loss(x, t, is_training = True)

        grads = {}

        for i in range(self.layer_num):
            grads[f"W{i}"] = resource.numerical_gradient(loss_W, self.params[f"W{i}"])
            grads[f"b{i}"] = resource.numerical_gradient(loss_W, self.params[f"b{i}"])
            grads[f"gamma{i}"] = resource.numerical_gradient(loss_W, self.params[f"gamma{i}"])
            grads[f"beta{i}"] = resource.numerical_gradient(loss_W, self.params[f"beta{i}"])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t, is_training = True)

        # backward
        dout = 1
        dout = self.output_layer.backward(dout)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        # output
        grads = {}
        for key in self.layers:
            layer = self.layers[key]
            idx = int(re.search(r"\d+$", key).group())

            if isinstance(layer, Affine):
                grads[f"W{idx}"] = layer.dW + self.weight_decay_lambda * self.params[f"W{idx}"]
                grads[f"b{idx}"] = layer.db

            if isinstance(layer, BatchNormalization):
                grads[f"gamma{idx}"] = layer.dgamma
                grads[f"beta{idx}"] = layer.dbeta

        return grads



# RUN

ONLY_CHECK_GRAD = False
SHOW_PLOT = False

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

# for overfitting test
#img_train = img_train[:300]
#label_train = label_train[:300]


# network def
network = DeepLayerNet(
        input_size = 784,
        hidden_size_list = [392, 196, 98, 49, 7],
        output_size = 10,
        weight = He(),
        weight_decay_lambda = 0.000001,
        drop_out_ratio = 0.001,
)


# check grad
if ONLY_CHECK_GRAD:
    img_batch = img_train[:3]
    label_batch = label_train[:3]

    grad_numerical = network.numerical_gradient(img_batch, label_batch)
    grad_backprop = network.gradient(img_batch, label_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))

    sys.exit()


# hyper params
iters_num = 10000
train_size = img_train.shape[0]
batch_size = 100
learning_rate = 0.001

train_loss_list = []
train_acc_list = []
test_acc_list = []

# loop count for each epoch
iter_per_epoch = max(train_size / batch_size, 1)

sgd = StochasticGradientDecent(lr = learning_rate)
momentum = Momentum(lr = learning_rate, momentum = 0.9)
ada_grad = AdaGrad(lr = learning_rate)
adam = Adam(lr = learning_rate, beta1 = 0.9, beta2 = 0.999)

for i in range(iters_num):
    # get mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    img_batch = img_train[batch_mask]
    label_batch = label_train[batch_mask]

    # calc grad
    grad = network.gradient(img_batch, label_batch)

    # update params
    #sgd.update(network.params, grad)
    #momentum.update(network.params, grad)
    #ada_grad.update(network.params, grad)
    adam.update(network.params, grad)

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



for idx, loss in enumerate(train_loss_list):
    if idx % batch_size == 0:
        print(f"train_loss_list[{idx}] = {loss}")






if SHOW_PLOT:
    # loss graph
    plot.figure()
    x = np.arange(iters_num)
    plot.plot(x, train_loss_list, label = "loss", linestyle = "-")
    plot.xlabel("train")
    plot.ylabel("loss")

    # accuracy graph
    plot.figure()
    x = np.arange(len(train_acc_list))
    plot.plot(x, train_acc_list, label = "train acc")
    plot.plot(x, test_acc_list, label = "test acc", linestyle='--')
    plot.xlabel("epochs")
    plot.ylabel("accuracy")
    plot.ylim(0, 1.0)
    plot.legend(loc='lower right')
    plot.show()


    plot.show()

