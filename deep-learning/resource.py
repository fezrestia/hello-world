import numpy as np
import matplotlib.pyplot as plot
import importlib
import sys
import os
import gzip
import pickle
from PIL import Image

def reload():
    importlib.reload(sys.modules[__name__])



# MNIST

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _load_label(file_path):
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def _load_img(file_path):
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    return data

def _change_to_one_hot_label(X):
    T = np.zeros(X.size, 10)
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_mnist(dir_path, normalize = True, flatten = True, one_hot_label = False):
    pkl_path = dir_path + "/" + "mnist.pkl"

    if not os.path.exists(pkl_path):
        train_img_file = dir_path + "/" + "train-images-idx3-ubyte.gz"
        train_label_file = dir_path + "/" + "train-labels-idx1-ubyte.gz"
        test_img_file = dir_path + "/" + "t10k-images-idx3-ubyte.gz"
        test_label_file = dir_path + "/" + "t10k-labels-idx1-ubyte.gz"

        dataset = {}
        dataset["train_img"] = _load_img(train_img_file)
        dataset["train_label"] = _load_label(train_label_file)
        dataset["test_img"] = _load_img(test_img_file)
        dataset["test_label"] = _load_label(test_label_file)

        with open(pkl_path, "wb") as f:
            pickle.dump(dataset, f, -1)

    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset["train_label"] = _change_to_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_to_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])

def mnist_flatten_img_show(flatten_img):
    img = flatten_img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()



# Logic Func

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.2

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    result = AND(s1, s2)
    return result



# Func

def identity_func(x):
    return x

def step_func(x):
    y = x > 0.0
    return y.astype(int)

def sigmoid_func(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu_func(x):
    return np.maximum(0.0, x)

def softmax_func(x):
    c = np.max(x)
    exp_x = np.exp(x - c)  # for anti overflow
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y



# Graph

# elms : [ {x, y, linestyle, label}, ... ]
def plot_graph(elms):
    for elm in elms:
        plot.plot(elm["x"], elm["y"], linestyle=elm["linestyle"], label=elm["label"])
    plot.xlabel("x")
    plot.ylabel("y")
    plot.title("graph")
    plot.legend()
    plot.show()









