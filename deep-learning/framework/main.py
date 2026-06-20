#!/usr/bin/env python3

# mypy: ignore-errors

import os
import sys
from pathlib import Path
from typing import override
import numpy as np
import math
import matplotlib.pyplot as plt

if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from deepzero import Variable
from deepzero import Parameter
from deepzero import Function
from deepzero import use_config, no_grad
from deepzero import Visualize
from deepzero.Function import add, mul, sub, rsub, div, rdiv, neg, pow, sin, cos, tanh, sum, reshape, transpose, matmul, linear, sigmoid, mean_squared_error, as_variable, get_item, softmax, softmax_cross_entropy, accuracy, relu
from deepzero import Layer
from deepzero.Layer import Linear
from deepzero import Model
from deepzero.Model import MultiLayerPerceptron
from deepzero import Optimizer
from deepzero.Optimizer import StochasticGradientDecent, MomentumSGD, Adam
from deepzero import DataSet
from deepzero.DataSet import Spiral, MNIST
from deepzero import DataLoader
from deepzero.cuda import gpu_enabled



#x = Variable(np.array(10))
#f = Function()
#try:
#    f(x)
#except NotImplementedError:
#    print("not implemented error")


#x = Variable(np.array(10))
#f = Square()
#y = f(x)
#print(type(y))
#print(y.data)


#A: Function = Square()
#B: Function = Exp()
#C: Function = Square()
#x = Variable(np.array(0.5))
#a = A(x)
#b = B(a)
#y = C(b)
#print(y.data)

#f = Square()
#x = Variable(np.array(2.0))
#dy = numerical_diff(f, x)
#print(dy)

#def ff(x):
#    A = Square()
#    B = Exp()
#    C = Square()
#    return C(B(A(x)))
#x = Variable(np.array(0.5))
#dy = numerical_diff(ff, x)
#print(dy)

#A = Square()
#B = Exp()
#C = Square()
#x = Variable(np.array(0.5))
#a = A(x)
#b = B(a)
#y = C(b)
#y.grad = np.array(1.0)
#b.grad = C.backward(y.grad)
#a.grad = B.backward(b.grad)
#x.grad = A.backward(a.grad)
#print(x.grad)
#assert y.creator == C
#assert y.creator.input == b
#assert y.creator.input.creator == B
#assert y.creator.input.creator.input == a
#assert y.creator.input.creator.input.creator == A
#assert y.creator.input.creator.input.creator.input == x

#y.grad = np.array(1.0)
#C = y.creator
#b = C.input
#b.grad = C.backward(y.grad)
#B = b.creator if b is not None and b.creator is not None else Function()
#a = B.input
#a.grad = B.backward(b.grad)
#A = a.creator if a is not None and a.creator is not None else Function()
#x = A.input
#x.grad = A.backward(a.grad)
#print(x.grad)

#x.grad = np.array(0.0)
#y.grad = np.array(1.0)
#y.backward()
#print(x.grad)

#x = Variable(np.array(0.5))
#a = square(x)
#b = exp(a)
#y = square(b)
#y.grad = np.array(1.0)
#y.backward()
#print(x.grad)

#x = Variable(np.array(0.5))
#y = square(exp(square(x)))
#y.backward()
#print(x.grad)

#xs = [Variable(np.array(2)), Variable(np.array(3))]
#f = Add()
#ys = f(*xs)
#y = ys[0]
#print(y.data)

#x0 = Variable(np.array(2))
#x1 = Variable(np.array(3))
#y, = add(x0, x1)
#print(y.data)
#z = square([x0, x1])
#print(z)
#print(z[0].data)

#x = Variable(np.array(0.5))
#y = square(exp(square(x)))
#y.backward()
#print(x.grad)

#x = Variable(np.array(2.0))
#y = Variable(np.array(3.0))
#z = add(square(x), square(y))
#z.backward()
#print(z.data)
#print(x.grad)
#print(y.grad)

#x = Variable(np.array(3.0))
#y = add(x, x)
#print(f"x = {x.data}, y = {y.data}")
#y.backward()
#print(f"x.grad = {x.grad}, y.grad = {y.grad}")

##x = Variable(np.array(3.0))
#x.clear_grad()
#y = add(add(x, x), x)
#y.backward()
#print(f"x={x.data}, y={y.data}")
#print(f"x.grad={x.grad}, y.grad={y.grad}")

#x = Variable(np.array(2.0))
#a = square(x)
#b = square(a)
#c = square(a)
#y = add(b, c)
#y.backward()
#print(f"y.ddata = {y.data}")
#print(f"x.grad = {x.grad}")
#print(f"a.grad = {a.grad}")
#print(f"b.grad = {b.grad}")
#print(f"c.grad = {c.grad}")

#Config.enable_backprop = True
#x = Variable(np.ones((100, 100, 100)))
#y = square(square(square(x)))
#y.backward()
#Config.enable_backprop = False
#x = Variable(np.ones((100, 100, 100)))
#y = square(square(square(x)))

#with use_config("enable_backprop", True):
#    x = Variable(np.array(2.0))
#    y = square(x)
#    y.backward()
#    print(f"x={x.data}, x.grad={x.grad}")
#with no_grad():
#    x = Variable(np.array(2.0))
#    y = square(x)
#    y.backward()
#    print(f"x={x.data}, x.grad={x.grad}")

#a = Variable(np.array(3.0))
#b = Variable(np.array(2.0))
#c = Variable(np.array(1.0))
#y = add(mul(a, b), c)
#y.backward()
#print(f"y = {y.data}")
#print(f"a.grad = {a.grad}")
#print(f"b.grad = {b.grad}")

#x = Variable(np.array([[1,2,3],[4,5,6]]))
#print(x.shape)
#print(x.ndim)
#print(x.size)
#print(x.dtype)
#print(len(x))
#print(x)

#a = Variable(np.array(3.0))
#b = Variable(np.array(2.0))
#y = a * b
#print(y)
#c = Variable(np.array(1.0))
#y = a * b + c
#y.backward()
#print(y)
#print(a.grad)
#print(b.grad)
#print(c.grad)

#x = Variable(np.array(2.0))
#y = x + np.array(3.0)
#print(y)

#x = Variable(np.array(2.0))
#y = x + 3.0
#print(y)
#print(1.0 + x)
#print(2.0 * x)

#x = Variable(np.array(2.0))
#y = -x
#print(y)
#y1 = 2.0 - x
#y2 = x - 2.0
#print(y1)
#print(y2)
#d1 = 2.0 / x
#d2 = x / 2.0
#print(d1)
#print(d2)
#p = x ** 3
#print(p)

#x = Variable(np.array(1.0))
#y = (x + 1) + x + (1 - x) - x + (2 * x) + (x * 2) + (1 / x) + (x / 1) + x ** 2
#print(f"y = {y}")

#def sphere(x, y):
#    z = x ** 2 + y ** 2
#    return z
#x = Variable(np.array(1.0))
#y = Variable(np.array(1.0))
#z = sphere(x, y)
#z.backward()
#print(f"x.grad={x.grad}, y.grad={y.grad}")

#def matyas(x, y):
#    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
#    return z
#x = Variable(np.array(1.0))
#y = Variable(np.array(1.0))
#z = matyas(x, y)
#z.backward()
#print(f"x.grad={x.grad}, y.grad={y.grad}")

#def goldstein(x, y):
#    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) \
#            * (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
#    return z
#x = Variable(np.array(1.0))
#y = Variable(np.array(1.0))
#z = goldstein(x, y)
#z.backward()
#print(f"x.grad={x.grad}, y.grad={y.grad}")
#x.name = "x"
#y.name = "y"
#z.name = "z"
#Visualize.plot_dot_graph(z, verbose = False, to_file = "~/.deepzero/goldstein_graph.png")

#x = Variable(np.random.randn(2, 3))
#x.name = "x"
#print(Visualize._dot_var(x))
#print(Visualize._dot_var(x, verbose = True))
#x0 = Variable(np.array(1.0))
#x0.name = "x0"
#x1 = Variable(np.array(1.0))
#x1.name = "x1"
#y = x0 + x1
#y.name = "y"
#txt = Visualize._dot_func(y.creator)
#print(txt)
#txt = Visualize.get_dot_graph(y, verbose = True)
#print(txt)
#txt = Visualize.plot_dot_graph(y, verbose = False, to_file = "~/.deepzero/graph.png")
#print(txt)

#x = Variable(np.array(np.pi / 4))
#y = sin(x)
#y.backward()
#print(y.data)
#print(x.grad)

#def my_sin(x, threshold = 1e-150):
#    y = 0
#    for i in range(100000):
#        c = (-1) ** i / math.factorial(2 * i + 1)
#        t = c * x ** (2 * i + 1)
#        y = y + t
#        if abs(t.data) < threshold:
#            break
#    return y
#x = Variable(np.array(np.pi / 4))
#y = my_sin(x)
#y.backward()
#print(y.data)
#print(x.grad)
#Visualize.plot_dot_graph(y, verbose = False, to_file = "~/.deepzero/graph.png")

#def rosenbrock(x0, x1):
#    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
#    return y
#x0 = Variable(np.array(0.0))
#x1 = Variable(np.array(2.0))
#y = rosenbrock(x0, x1)
#y.backward()
#print(x0.grad, x1.grad)
#lr = 0.001
#iters = 1000
#for i in range(iters):
#    print(x0, x1)
#    y = rosenbrock(x0, x1)
#    x0.clear_grad()
#    x1.clear_grad()
#    y.backward()
#    x0.data -= lr * x0.grad
#    x1.data -= lr * x1.grad

#def f(x):
#    y = x ** 4 - 2 * x ** 2
#    return y
#def d2f(x):
#    return 12 * x ** 2 - 4
#x = Variable(np.array(2.0))
#iters = 10
#for i in range(iters):
#    print(i, x)
#    y = f(x)
#    x.clear_grad()
#    y.backward()
#    if x.grad is not None:
#        x.data -= x.grad.data / d2f(x.data)

#def f(x):
#    y = x ** 4 - 2 * x ** 2
#    return y
#x = Variable(np.array(2.0))
#y = f(x)
#y.backward(create_graph = True)
#print(x.grad)
#gx = x.grad
#x.clear_grad()
#gx.backward()
#print(x.grad)

#def f(x):
#    y = x ** 4 - 2 * x ** 2
#    return y
#x = Variable(np.array(2.0))
#iters = 10
#for i in range(iters):
#    print(i, x)
#    y = f(x)
#    x.clear_grad()
#    y.backward(create_graph = True)
#    gx = x.grad
#    x.clear_grad()
#    if gx is not None:
#        gx.backward()
#    gx2 = x.grad
#    if gx is not None and gx2 is not None:
#        x.data -= gx.data / gx2.data

#x = Variable(np.array(1.0))
#y = sin(x)
#y.backward(create_graph = True)
#for i in range(3):
#    gx = x.grad
#    x.clear_grad()
#    gx.backward(create_graph = True)
#    print(x.grad)

#x = Variable(np.linspace(-7, 7, 200))
#y = sin(x)
#y.backward(create_graph = True)
#logs = [y.data.flatten()]
#for i in range(3):
#    logs.append(x.grad.data.flatten())
#    gx = x.grad
#    x.clear_grad()
#    gx.backward(create_graph = True)
#labels = ["y=sin(x)", "y'", "y''", "y'''"]
#for i, v in enumerate(logs):
#    plt.plot(x.data, logs[i], label=labels[i])
#plt.legend(loc="lower right")
#plt.show()

#x = Variable(np.array(1.0))
#y = tanh(x)
#x.name = "x"
#y.name = "y"
#y.backward(create_graph = True)
#iters = 8
#for i in range(iters):
#    gx = x.grad
#    x.clear_grad()
#    gx.backward(create_graph = True)
#gx = x.grad
#gx.name = "gx" + str(iters + 1)
#Visualize.plot_dot_graph(gx, verbose = False, to_file = "~/.deepzero/tanh.png")

#x = Variable(np.array([[1,2,3],[4,5,6]]))
#y = sin(x)
#print(y)
#c = Variable(np.array([[10,20,30],[40,50,60]]))
#t = x + c
#y = sum(t)
#print(y)
#y.backward(keep_grad = True)
#print(y.grad)
#print(t.grad)
#print(x.grad)
#print(c.grad)

#x = Variable(np.array([[1,2,3],[4,5,6]]))
#y = x.reshape((3,2))
#print(y)
#y = transpose(x)
#print(y)
#y.backward(keep_grad = True)
#print(x.grad)
#print(y.grad)
#x = Variable(np.random.rand(2,3))
#y1 = x.transpose()
#y2 = x.T
#print(y1)
#print(y2)

#x = Variable(np.array([[[1,11],[2,22],[3,33]],[[4,44],[5,55],[6,66]],[[7,77],[8,88],[9,99]]]))
#y = x.T
#print(y)
#yy = x.transpose((0,2,1))
#print(yy)

#x = Variable(np.array([[1,2,3],[4,5,6]]))
#y = sum(x)
#y.backward()
#print(y)
#print(x.grad)

#x = Variable(np.array([[1,2,3],[4,5,6]]))
#y = sum(x, axis = 0)
#y.backward()
#print(y)
#print(x.grad)
#x = Variable(np.random.randn(2,3,4,5))
#y = x.sum(keepdims = True)
#print(y.shape)

#x0 = Variable(np.array([1,2,3]))
#x1 = Variable(np.array([10]))
#y = x0 + x1
#print(y)

#x0 = Variable(np.array([1,2,3]))
#x1 = Variable(np.array([10]))
#y = x0 + x1
#print(y)
#y.backward()
#print(x1)
#print(x1.grad)

#x = Variable(np.random.randn(2, 3))
#W = Variable(np.random.randn(3, 4))
#y = matmul(x, W)
#y.backward()
#print(f"y.shape={y.shape}")
#print(f"x.shape={x.shape}, x.grad.shape={x.grad.shape}")
#print(f"W.shape={W.shape}, W.grad.shape={W.grad.shape}")

#np.random.seed(0)
#x = np.random.rand(100, 1)  # (100,1)
#y = 5 + 2 * x + np.random.rand(100, 1)  # (100,1)
#plt.scatter(x, y, s = 10)
#plt.xlabel("x")
#plt.ylabel("y")
##plt.show()
#x = Variable(x)
#y = Variable(y)
#W = Variable(np.zeros((1,1)))
#b = Variable(np.zeros(1))
#def predict(x):
#    y = matmul(x, W) + b
#    return y
#def mean_squared_error(x0, x1):
#    diff = x0 - x1
#    return sum(diff ** 2) / len(diff)
#lr = 0.1
#iter = 100
#for i in range(iter):
#    y_pred = predict(x)
#    loss = mean_squared_error(y, y_pred)
#    W.clear_grad()
#    b.clear_grad()
#    loss.backward()
#    W.data -= lr * W.grad.data
#    b.data -= lr * b.grad.data
#    print(W, b, loss)
#plt.plot(x.data, y_pred.data, color = "r")
#plt.show()

#x = Variable(np.random.randn(2, 3))
#W = Variable(np.random.randn(3, 4))
#b = Variable(np.zeros(1))
#y = linear(x, W, b)
#y.backward()
#print(x.grad)

#np.random.seed(0)
#x = np.random.rand(100, 1)
#y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#plt.scatter(x, y, s=10)
#plt.xlabel("x")
#plt.ylabel("y")
#I, H, O = 1, 10, 1
#W1 = Variable(0.01 * np.random.randn(I, H))
#b1 = Variable(np.zeros(H))
#W2 = Variable(0.01 * np.random.randn(H, O))
#b2 = Variable(np.zeros(O))
#def predict(x):
#    y = linear(x, W1, b1)
#    y = sigmoid(y)
#    y = linear(y, W2, b2)
#    return y
#lr = 0.2
#iters = 10000
#for i in range(iters):
#    y_pred = predict(x)
#    loss = mean_squared_error(y, y_pred)
#    W1.clear_grad()
#    b1.clear_grad()
#    W2.clear_grad()
#    b2.clear_grad()
#    loss.backward()
#    W1.data -= lr * W1.grad.data
#    b1.data -= lr * b1.grad.data
#    W2.data -= lr * W2.grad.data
#    b2.data -= lr * b2.grad.data
#    if i % 1000 == 0:
#        print(loss)
#print(f"W1.shape={W1.shape}")
#print(W1)
#print(f"b1.shape={b1.shape}")
#print(b1)
#print(f"W2.shape={W2.shape}")
#print(W2)
#print(f"b2.shape={b2.shape}")
#print(b2)
#t = np.arange(0, 1, .01)[:, np.newaxis]
#y_pred = predict(t)
#plt.plot(t, y_pred.data, color = "r")
#plt.show()

#x = Variable(np.array(1.0))
#p = Parameter(np.array(1.0))
#y = x + p
#print(isinstance(p, Parameter))
#print(isinstance(x, Parameter))
#print(isinstance(y, Parameter))

#np.random.seed(0)
#x = np.random.rand(100, 1)
#y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#l1 = Linear(10)
#l2 = Linear(1)
#def predict(x):
#    (y,) = l1(x)
#    y = sigmoid(y)
#    (y,) = l2(y)
#    return y
#lr = 0.2
#iters = 10000
#for i in range(iters):
#    y_pred = predict(x)
#    loss = mean_squared_error(Variable(y), y_pred)
#    l1.clear_grads()
#    l2.clear_grads()
#    loss.backward()
#    for l in [l1, l2]:
#        for p in l.params():
#            p.data -= lr * p.grad.data
#    if i % 1000 == 0:
#        print(loss)

#np.random.seed(0)
#x = np.random.rand(100, 1)
#y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#lr = 0.2
#iters = 10000
#model = Layer()
#model.l1 = Linear(5)
#model.l2 = Linear(3)
#def predict(model, x):
#    (y,) = model.l1(x)
#    y = sigmoid(y)
#    (y,) = model.l2(y)
#    return y
#for i in range(iters):
#    y_pred = predict(model, x)
#    loss = mean_squared_error(Variable(y), y_pred)
#    model.clear_grads()
#    loss.backward()
#    for p in model.params():
#        p.data -= lr * p.grad.data
#    if i % 1000 == 0:
#        print(loss)
#for p in model.params():
#    print(p)

#class TwoLayerNet(Model):
#    def __init__(self, hidden_size: int, out_size: int):
#        super().__init__()
#        self.l1 = Linear(hidden_size)
#        self.l2 = Linear(out_size)
#    @override
#    def forward(self, *x: Variable) -> tuple[Variable, ...]:
#        (y,) = self.l1(*x)
#        y = sigmoid(y)
#        (y,) = self.l2(y)
#        return (y,)
#x = Variable(np.random.randn(5, 10), name = "x")
#model: Model = TwoLayerNet(100, 10)
#model.plot(x, to_file = "~/.deepzero/model.png")

#np.random.seed(0)
#x = np.random.rand(100, 1)
#y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#lr = 0.2
#max_iters = 10000
#hidden_size = 10
#class TwoLayerNet(Model):
#    def __init__(self, hidden_size: int, out_size: int):
#        super().__init__()
#        self.l1 = Linear(hidden_size)
#        self.l2 = Linear(out_size)
#    @override
#    def forward(self, *x: Variable) -> tuple[Variable, ...]:
#        (y,) = self.l1(*x)
#        y = sigmoid(y)
#        (y,) = self.l2(y)
#        return (y,)
#model: Model = TwoLayerNet(hidden_size, 1)
#for i in range(max_iters):
#    (y_pred,) = model(Variable(x))
#    loss = mean_squared_error(Variable(y), y_pred)
#    model.clear_grads()
#    loss.backward()
#    for p in model.params():
#        p.data -= lr * p.grad.data
#    if i % 1000 == 0:
#        print(loss)

#np.random.seed(0)
#x = np.random.rand(100, 1)
#y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#lr = 0.2
#max_iters = 10000
#hidden_size = 10
#model: Model = MultiLayerPerceptron((hidden_size, 1))
##optimizer: Optimizer = StochasticGradientDecent(lr).setup(model)
#optimizer: Optimizer = MomentumSGD(lr).setup(model)
#for i in range(max_iters):
#    (y_pred,) = model(as_variable(x))
#    loss = mean_squared_error(as_variable(y), y_pred)
#    model.clear_grads()
#    loss.backward()
#    optimizer.update()
#    if i % 1000 == 0:
#        print(loss)

#x = Variable(np.array([[1,2,3],[4,5,6]]))
#indices = np.array([0, 0, 1])
#y = get_item(x, indices)
#print(y)
#print(x[1])
#print(x[:,2])

#np.random.seed(0)
#x = np.array([[0.2,-0.4],[0.3,0.5],[1.3,-3.2],[2.1,0.3]])
#t = np.array([2,0,1,0])
#model: Model = MultiLayerPerceptron((10, 3))
#(y,) = model(x)
#print(y)
#loss = softmax_cross_entropy(y, Variable(t))
#loss.backward()
#print(loss)

#N = 100
#CLS_NUM = 3
#x, t = spiral.load_train_data()
#print(x.shape)
#print(t.shape)
#print(x[10], t[10])
#print(x[110], t[110])
## train
#max_epoch = 300
#batch_size = 30
#hidden_size = 10
#lr = 1.0
#model = MultiLayerPerceptron((hidden_size, CLS_NUM))
#optimizer = StochasticGradientDecent(lr).setup(model)
#data_size = len(x)
#max_iter = math.ceil(data_size / batch_size)
#for epoch in range(max_epoch):
#    index = np.random.permutation(data_size)
#    sum_loss = 0.0
#    for i in range(max_iter):
#        batch_index = index[i * batch_size:(i + 1) * batch_size]
#        batch_x = x[batch_index]
#        batch_t = t[batch_index]
#        (y,) = model(batch_x)
#        loss = softmax_cross_entropy(y, batch_t)
#        model.clear_grads()
#        loss.backward()
#        optimizer.update()
#        sum_loss += float(loss.data) * len(batch_t)  # loss is average for N
#    avg_loss = sum_loss / data_size
#    print(f"epoch = {epoch + 1}, loss = {avg_loss}")
## boundary plot
#h = 0.001
#x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
#y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#X = np.c_[xx.ravel(), yy.ravel()]
#with no_grad():
#    (score,) = model(X)
#predict_cls = np.argmax(score.data, axis = 1)
#Z = predict_cls.reshape(xx.shape)
#plt.contourf(xx, yy, Z)
## point plot
#markers = ["o", "x", "^"]
#colors = ["orange", "blue", "green"]
#for i in range(len(x)):
#    c = t[i]
#    plt.scatter(x[i][0], x[i][1], s = 40, marker = markers[c], c = colors[c])
#plt.show()

#train_set = Spiral(train = True)
#print(train_set[0])
#print(len(train_set))
#batch_index = [0, 1, 2]
#batch = [train_set[i] for i in batch_index]
#x = np.array([sample[0] for sample in batch])
#t = np.array([sample[1] for sample in batch])
#print(x.shape)
#print(t.shape)

#max_epoch = 300
#batch_size = 30
#hidden_size = 10
#lr = 1.0
#train_data = Spiral(train = True)
#model = MultiLayerPerceptron((hidden_size, 10))
#optimizer = StochasticGradientDecent(lr).setup(model)
#data_size = len(train_data)
#max_iter = math.ceil(data_size / batch_size)
#for epoch in range(max_epoch):
#    index = np.random.permutation(data_size)
#    sum_loss = 0.0
#    for i in range(max_iter):
#        batch_index = index[i * batch_size:(i + 1) * batch_size]
#        batch = [train_data[i] for i in batch_index]
#        batch_x = np.array([data[0] for data in batch])
#        batch_t = np.array([data[1] for data in batch])
#        (y,) = model(batch_x)
#        loss = softmax_cross_entropy(y, batch_t)
#        model.clear_grads()
#        loss.backward()
#        optimizer.update()
#        sum_loss += float(loss.data) * len(batch_t)  # loss is average for N
#    avg_loss = sum_loss / data_size
#    print(f"epoch = {epoch + 1}, loss = {avg_loss}")

#batch_size = 10
#max_epoch = 1
#train_set = Spiral(train = True)
#test_set = Spiral(train = False)
#train_loader = DataLoader(train_set, batch_size)
#test_loader = DataLoader(test_set, batch_size, shuffle = False)
#for epoch in range(max_epoch):
#    for x, t in train_loader:
#        print(x.shape, t.shape)
#        break
#    for x, t in test_loader:
#        print(x.shape, t.shape)
#        break

#y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
#t = np.array([1, 2, 0])
#accuracy = accuracy(y, t)
#print(accuracy)

#max_epoch = 300
#batch_size = 30
#hidden_size = 10
#lr = 1.0
#train_set = Spiral(train = True)
#test_set = Spiral(train = False)
#train_loader = DataLoader(train_set, batch_size)
#test_loader = DataLoader(test_set, batch_size, shuffle = False)
#model = MultiLayerPerceptron((hidden_size, 10))
#optimizer = StochasticGradientDecent(lr).setup(model)
#for epoch in range(max_epoch):
#    sum_loss = 0.0
#    sum_acc = 0.0
#    for x, t in train_loader:
#        (y,) = model(x)
#        loss = softmax_cross_entropy(y, t)
#        acc = accuracy(y, t)
#        model.clear_grads()
#        loss.backward()
#        optimizer.update()
#        sum_loss += float(loss.data) * len(t)
#        sum_acc += float(acc.data) * len(t)
#    print(f"epoch = {epoch + 1}, loss = {sum_loss / len(train_set)}, accuracy = {sum_acc / len(train_set)}")
#    sum_loss = 0.0
#    sum_acc = 0.0
#    with no_grad():
#        for x, t in test_loader:
#            (y,) = model(x)
#            loss = softmax_cross_entropy(y, t)
#            acc = accuracy(y, t)
#            sum_loss += float(loss.data) * len(t)
#            sum_acc += float(acc.data) * len(t)
#    print(f"test loss = {sum_loss / len(test_set)}, accuracy = {sum_acc / len(test_set)}")

#train_set = MNIST(train = True, transform = None, target_transform = None)
#test_set = MNIST(train = False, transform = None, target_transform = None)
#print(len(train_set))
#print(len(test_set))
#x, t = train_set[0]
#print(type(x), x.shape)
#print(t)
#train_set.show_samples()
#test_set.show_samples()



# MNIST learning
def preproc(x: np.ndarray) -> np.ndarray:
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x
train_set = MNIST(train = True, transform = preproc, target_transform = None)
test_set = MNIST(train = False, transform = preproc, target_transform = None)
max_epoch = 5
batch_size = 100
hidden_size = 1000
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle = False)
model = MultiLayerPerceptron((hidden_size, hidden_size, 10), activation = relu)
optimizer = Adam().setup(model)
if gpu_enabled:
    train_loader.to_gpu()
    test_loader.to_gpu()
    model.to_gpu()
for epoch in range(max_epoch):
    sum_loss = 0.0
    sum_acc = 0.0
    for x, t in train_loader:
        #print(f"x.tpye = {type(x)}, t.type = {type(t)}")
        (y,) = model(x)
        loss = softmax_cross_entropy(y, t)
        acc = accuracy(y, t)
        model.clear_grads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    print(f"epoch = {epoch + 1}, loss = {sum_loss / len(train_set):.4f}, accuracy = {sum_acc / len(train_set):.4f}")
    sum_loss = 0.0
    sum_acc = 0.0
    with no_grad():
        for x, t in test_loader:
            #print(f"x.tpye = {type(x)}, t.type = {type(t)}")
            (y,) = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print(f"test loss = {sum_loss / len(test_set):.4f}, accuracy = {sum_acc / len(test_set):.4f}")

