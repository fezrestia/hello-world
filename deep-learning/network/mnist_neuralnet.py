#!/usr/bin/env python3

import sys
import pickle
import numpy as np

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)

import resource

def get_data():
    (img_train, label_train), (img_test, label_test) = \
            resource.load_mnist(root_dir + "/mnist", normalize = True, flatten = True, one_hot_label = False)
    return img_test, label_test

def init_network():
    with open(root_dir + "/mnist/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(networi, img):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(img, W1) + b1
    z1 = resource.sigmoid_func(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = resource.sigmoid_func(a2)

    a3 = np.dot(z2, W3) + b3
    result = resource.softmax_func(a3)

    return result


# RUN

img, label = get_data()
network = init_network()

batch_size = 100

accuracy_cnt = 0

for i in range(0, len(img), batch_size):
    img_batch = img[i:i + batch_size]
    result_batch = predict(network, img_batch)
    p = np.argmax(result_batch, axis = 1)
    accuracy_cnt += np.sum(p == label[i:i + batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(img)))

