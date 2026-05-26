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
    T = np.zeros((X.size, 10))
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
    c = np.max(x, axis = -1, keepdims = True)  # for last dim
    exp_x = np.exp(x - c)  # for anti overflow
    sum_exp_x = np.sum(exp_x, axis = -1, keepdims = True)
    y = exp_x / sum_exp_x
    return y



# Numerical Func

def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

def _numerical_gradient(f, x):
    h = 1e-4  # 0.0001

    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # recover

    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient(f, x)

        return grad



# Learning Method

# lr : learning late
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)



# Evaluate Func

def sum_squared_error(act, exp):
    return 0.5 * np.sum((act - exp) ** 2)

def cross_entropy_error(act, exp):
    if act.ndim == 1:
        act = act.reshape(1, act.size)
        exp = exp.reshape(1, exp.size)

    batch_size = act.shape[0]

    delta = 1e-7
    return -1.0 * np.sum(exp * np.log(act + delta)) / batch_size



# Graph

# elms : [ {x, y, linestyle, label}, ... ]
def plot_graph(elms):
    plt.figure()
    for elm in elms:
        plot.plot(elm["x"], elm["y"], linestyle=elm["linestyle"], label=elm["label"])
    plot.xlabel("x")
    plot.ylabel("y")
    plot.title("graph")
    plot.legend()
    plot.show()

def plot_2d(pos_x, pos_y, xlim, ylim, x_label, y_label):
    plot.figure()
    plot.plot([xlim[0] - 1.0, xlim[1] + 1.0], [0, 0], "--b")
    plot.plot([0, 0], [ylim[0] - 1.0, ylim[1] + 1.0], "--b")
    plot.plot(pos_x, pos_y, 'o')
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.show()



# pos_x/y, vec_x/y : array
# xlim, ylim : array [2], min, max
def plot_2d_vector_field(pos_x, pos_y, vec_x, vec_y, xlim, ylim, x_label, y_label):
    plot.figure()
    plot.quiver(pos_x, pos_y, vec_x, vec_y, angles="xy")
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.grid()
    plot.draw()
    plot.show()







