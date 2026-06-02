#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plot
from collections import OrderedDict

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

from layer.Affine import Affine
from layer.Relu import Relu
from layer.SoftmaxWithLoss import SoftmaxWithLoss
from layer.Convolution import Convolution
from layer.Pooling import Pooling

from optimizer.Adam import Adam

class ConvLayerNet:
    def __init__(
            self,
            input_chw = {1, 28, 28},
            conv_param = {
                    "filter_num": 30,
                    "filter_size": 5,
                    "stride": 1,
                    "padding": 0,
            },
            hidden_size = 100,
            output_size = 10,
            weight_init_std = 0.01,
    ):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_stride = conv_param["stride"]
        filter_padding = conv_param["padding"]

        input_ch = input_chw[0]
        input_size = input_chw[1]
        conv_output_size = (input_size - filter_size + 2 * filter_padding) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))


        self.params = {}

        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_ch, filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)

        self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)

        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)


        self.layers = OrderedDict()

        self.layers["Conv1"] = Convolution(
                self.params["W1"],
                self.params["b1"],
                filter_stride,
                filter_padding,
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h = 2, pool_w = 2, stride = 2)

        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()

        self.layers["Affine3"] = Affine(self.params["W3"], self.params["b3"])

        self.loss_layer = SoftmaxWithLoss()

    # x: input
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x: input
    # t: grand truth
    def loss(self, x, t):
        y = self.predict(x)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.loss_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # output
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        grads["W3"] = self.layers["Affine3"].dW
        grads["b3"] = self.layers["Affine3"].db

        return grads



# RUN

SHOW_PLOT = True

# data
(img_train, label_train), (img_test, label_test) = \
    resource.load_mnist(root_dir + "/mnist", normalize = True, flatten = False, one_hot_label = True)


# hyper params
iters_num = 10000
train_size = img_train.shape[0]
batch_size = 100
learning_rate = 0.001


# network def
network = ConvLayerNet(
        input_chw = (1,28,28),
        conv_param = {
                "filter_num": 30,
                "filter_size": 5,
                "stride": 1,
                "padding": 0,
        },
        hidden_size = 100,
        output_size = 10,
        weight_init_std = 0.01
)


train_loss_list = []
train_acc_list = []
test_acc_list = []


# loop count for each epoch
iter_per_epoch = max(train_size / batch_size, 1)

adam = Adam(lr = learning_rate, beta1 = 0.9, beta2 = 0.999)

for i in range(iters_num):
    # get mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    img_batch = img_train[batch_mask]
    label_batch = label_train[batch_mask]

    # calc grad
    grad = network.gradient(img_batch, label_batch)

    # update params
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

