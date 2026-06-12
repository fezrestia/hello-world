import sys
from pathlib import Path
from typing import override
import numpy as np
import weakref

# include same dir layer.
same_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, same_dir)

from Variable import Variable
from Config import Config
from Type import Scalar, ScalarTypes

class Function:
    def __call__(self, *raw_inputs: Variable|np.ndarray) -> tuple[Variable, ...]:
        inputs: tuple[Variable, ...] = tuple([as_variable(x) for x in raw_inputs])

        xs: tuple[np.ndarray, ...] = tuple([x.data for x in inputs])
        ys: tuple[np.ndarray, ...] = self.forward(xs)

        outputs: tuple[Variable, ...] = tuple([Variable(as_array(y)) for y in ys])

        if Config.enable_backprop:
            self.generation: int = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs: tuple[Variable, ...] = inputs
            self.outputs = [weakref.ref(o) for o in outputs]

        return outputs

    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        raise NotImplementedError()

    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        raise NotImplementedError()


class Square(Function):
    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x: np.ndarray = xs[0]
        y: np.ndarray = x ** 2
        return (y,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        gy: np.ndarray = gys[0]
        x: np.ndarray = self.inputs[0].data  # square has only 1 input stored in tuple[Variable]
        gx: np.ndarray = 2 * x * gy
        return (gx,)

def square(x: Variable) -> Variable:
    y: Variable
    y, = Square()(x)
    return y


class Exp(Function):
    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x: np.ndarray = xs[0]
        y: np.ndarray = np.exp(x)
        return (y,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x: np.ndarray = self.inputs[0].data  # exp has only 1 input stored in tuple[Variable]
        gy: np.ndarray = gys[0]
        gx: np.ndarray = np.exp(x) * gy
        return (gx,)

def exp(x: Variable) -> Variable:
    y: Variable
    y, = Exp()(x)
    return y


class Add(Function):
    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x0: np.ndarray = xs[0]
        x1: np.ndarray = xs[1]
        y: np.ndarray = x0 + x1
        return (y,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        gy: np.ndarray = gys[0]
        return (gy, gy)

def add(x0: Variable, x1: Variable|np.ndarray|Scalar) -> Variable:
    y: Variable
    if isinstance(x1, ScalarTypes):
        y, = Add()(x0, as_array(x1))
    else:
        y, = Add()(x0, x1)
    return y


class Mul(Function):
    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x0: np.ndarray = xs[0]
        x1: np.ndarray = xs[1]
        y: np.ndarray = x0 * x1
        return (y,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x0: np.ndarray = self.inputs[0].data
        x1: np.ndarray = self.inputs[1].data
        gy: np.ndarray = gys[0]
        gx0: np.ndarray = gy * x1
        gx1: np.ndarray = gy * x0
        return (gx0, gx1)

def mul(x0: Variable, x1: Variable|np.ndarray|Scalar) -> Variable:
    y: Variable
    if isinstance(x1, ScalarTypes):
        y, = Mul()(x0, as_array(x1))
    else:
        y, = Mul()(x0, x1)
    return y


class Neg(Function):
    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x: np.ndarray = xs[0]
        return (-x,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        gy: np.ndarray = gys[0]
        return (-gy,)

def neg(x: Variable) -> Variable:
    return Neg()(x)[0]


class Sub(Function):
    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x0: np.ndarray = xs[0]
        x1: np.ndarray = xs[1]
        y: np.ndarray = x0 - x1
        return (y,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        gy: np.ndarray = gys[0]
        return (gy, -gy)

def sub(x0: Variable|np.ndarray|Scalar, x1: Variable|np.ndarray|Scalar) -> Variable:
    y: Variable
    if isinstance(x0, ScalarTypes):
        if isinstance(x1, ScalarTypes):
            y, = Sub()(as_array(x0), as_array(x1))
        else:
            y, = Sub()(as_array(x0), x1)
    else:
        if isinstance(x1, ScalarTypes):
            y, = Sub()(x0, as_array(x1))
        else:
            y, = Sub()(x0, x1)
    return y


class Div(Function):
    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x0: np.ndarray = xs[0]
        x1: np.ndarray = xs[1]
        y: np.ndarray = x0 / x1
        return (y,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x0: np.ndarray = self.inputs[0].data
        x1: np.ndarray = self.inputs[1].data
        gy: np.ndarray = gys[0]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return (gx0, gx1)

def div(x0: Variable|np.ndarray|Scalar, x1: Variable|np.ndarray|Scalar) -> Variable:
    y: Variable
    if isinstance(x0, ScalarTypes):
        if isinstance(x1, ScalarTypes):
            y, = Div()(as_array(x0), as_array(x1))
        else:
            y, = Div()(as_array(x0), x1)
    else:
        if isinstance(x1, ScalarTypes):
            y, = Div()(x0, as_array(x1))
        else:
            y, = Div()(x0, x1)
    return y


class Pow(Function):
    def __init__(self, c: int):
        self.c: int = c

    @override
    def forward(self, xs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x: np.ndarray = xs[0]
        y: np.ndarray = x ** self.c
        return (y,)

    @override
    def backward(self, gys: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        x: np.ndarray = self.inputs[0].data
        c: int = self.c
        gy: np.ndarray = gys[0]
        gx: np.ndarray = c * x ** (c - 1) * gy
        return (gx,)

def pow(x: Variable, c: int) -> Variable:
    return Pow(c)(x)[0]




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

def as_variable(x: np.ndarray|Variable) -> Variable:
    if isinstance(x, Variable):
        return x
    return Variable(x)

