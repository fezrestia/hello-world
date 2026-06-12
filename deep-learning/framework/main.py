#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import override
import numpy as np

# include same dir layer.
same_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, same_dir)

from Variable import Variable
from Function import Function
from Config import Config
from Config import use_config, no_grad



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

x = Variable(np.array(2.0))
y = -x
print(y)
y1 = 2.0 - x
y2 = x - 2.0
print(y1)
print(y2)
d1 = 2.0 / x
d2 = x / 2.0
print(d1)
print(d2)
p = x ** 3
print(p)

