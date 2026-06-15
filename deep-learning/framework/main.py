#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import override
import numpy as np
import math
import matplotlib.pyplot as plt

if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from deepzero import Variable
from deepzero import Function
from deepzero import use_config, no_grad
from deepzero import Visualize
from deepzero.Function import add, mul, sub, rsub, div, rdiv, neg, pow, sin, cos, tanh



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

x = Variable(np.array(1.0))
y = tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph = True)
iters = 8
for i in range(iters):
    gx = x.grad
    x.clear_grad()
    gx.backward(create_graph = True)
gx = x.grad
gx.name = "gx" + str(iters + 1)
Visualize.plot_dot_graph(gx, verbose = False, to_file = "~/.deepzero/tanh.png")



