import sys
from pathlib import Path
from typing import override
import numpy as np

# include same dir layer.
same_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, same_dir)

from Variable import Variable

class Function:
    def __call__(self, input: Variable) -> Variable:
        self.input: Variable = input

        x = input.data
        y = self.forward(x)

        output: Variable = Variable(as_array(y))
        output.set_creator(self)
        self.output = output

        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    @override
    def forward(self, x):
        y = x ** 2
        return y

    @override
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x: Variable):
    return Square()(x)


class Exp(Function):
    @override
    def forward(self, x):
        y = np.exp(x)
        return y

    @override
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x: Variable):
    return Exp()(x)


def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x



# RUN
if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Function()
    try:
        f(x)
    except NotImplementedError:
        print("not implemented error")


    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)


    A: Function = Square()
    B: Function = Exp()
    C: Function = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)

    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    def ff(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))
    x = Variable(np.array(0.5))
    dy = numerical_diff(ff, x)
    print(dy)

    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    y.grad = np.array(1.0)
    C = y.creator
    b = C.input
    b.grad = C.backward(y.grad)
    B = b.creator if b is not None and b.creator is not None else Function()
    a = B.input
    a.grad = B.backward(b.grad)
    A = a.creator if a is not None and a.creator is not None else Function()
    x = A.input
    x.grad = A.backward(a.grad)
    print(x.grad)

    x.grad = np.array(0.0)
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)

