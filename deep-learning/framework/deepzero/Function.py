import sys
from pathlib import Path
from typing import override
from types import ModuleType
import numpy as np
import weakref
from weakref import ReferenceType
from collections.abc import Callable

from .Variable import Variable
from .Config import Config
from .Type import Scalar, ScalarTypes, Array, ArrayTypes
from .Log import log_d, log_e
from .cuda import npcp

class Function:
    def __call__(self, *raw_inputs: Variable|Array) -> tuple[Variable, ...]:
        inputs: tuple[Variable, ...] = tuple([as_variable(x) for x in raw_inputs])

        xs: tuple[Array, ...] = tuple([x.data for x in inputs])
        ys: tuple[Array, ...] = self.forward(xs)

        outputs: tuple[Variable, ...] = tuple([Variable(as_array(y)) for y in ys])

        if Config.enable_backprop:
            self.generation: int = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs: tuple[Variable, ...] = inputs
            self.outputs: tuple[ReferenceType[Variable], ...] = tuple([weakref.ref(o) for o in outputs])

        return outputs

    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        raise NotImplementedError()

    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        raise NotImplementedError()


class Square(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        y: Array = x ** 2
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        x: Variable = self.inputs[0]  # square has only 1 input stored in tuple[Variable]
        gx: Variable = 2 * x * gy
        return (gx,)

def square(x: Variable) -> Variable:
    y: Variable
    y, = Square()(x)
    return y


class Exp(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y: Array = xp.exp(x)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        y: Variable|None = self.outputs[0]()  # weak ref
        if y is not None:
            gx: Variable = gy * y
            return (gx,)
        else:
            log_e(self, "y is None")
            assert y is not None

def exp(x: Variable) -> Variable:
    return Exp()(x)[0]


class Log(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y: Array = xp.log(x)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        x: Variable = self.inputs[0]
        gx: Variable = gy / x
        return (gx,)

def log(x: Variable) -> Variable:
    return Log()(x)[0]


class Add(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x0: Array = xs[0]
        x1: Array = xs[1]
        xp: ModuleType = npcp(x0)

        self.x0_shape: tuple[int, ...] = x0.shape
        self.x1_shape: tuple[int, ...] = x1.shape
        y: Array = x0 + x1
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        gx0: Variable = gy
        gx1: Variable = gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return (gx0, gx1)

def add(x0: Variable, x1: Variable|Array|Scalar) -> Variable:
    xp: ModuleType = npcp(x0.data)

    y: Variable
    if isinstance(x1, ScalarTypes):
        y, = Add()(x0, as_array(x1, xp))
    else:
        y, = Add()(x0, x1)
    return y


class Mul(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x0: Array = xs[0]
        x1: Array = xs[1]

        y: Array = x0 * x1
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x0: Variable = self.inputs[0]
        x1: Variable = self.inputs[1]
        gy: Variable = gys[0]
        gx0: Variable = gy * x1
        gx1: Variable = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return (gx0, gx1)

def mul(x0: Variable, x1: Variable|Array|Scalar) -> Variable:
    xp: ModuleType = npcp(x0.data)

    y: Variable
    if isinstance(x1, ScalarTypes):
        y, = Mul()(x0, as_array(x1, xp))
    else:
        y, = Mul()(x0, x1)
    return y


class Neg(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        return (-x,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        return (-gy,)

def neg(x: Variable) -> Variable:
    return Neg()(x)[0]


class Sub(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x0: Array = xs[0]
        x1: Array = xs[1]
        self.x0_shape: tuple[int, ...] = x0.shape
        self.x1_shape: tuple[int, ...] = x1.shape
        y: Array = x0 - x1
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        gx0: Variable = gy
        gx1: Variable = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return (gx0, gx1)

def sub(x0: Variable|Array|Scalar, x1: Variable|Array|Scalar) -> Variable:
    y: Variable
    if isinstance(x0, ScalarTypes):
        if isinstance(x1, ScalarTypes):
            y, = Sub()(as_array(x0), as_array(x1))
        else:
            y, = Sub()(as_array(x0, npcp(x1)), x1)
    else:
        if isinstance(x1, ScalarTypes):
            y, = Sub()(x0, as_array(x1, npcp(x0)))
        else:
            y, = Sub()(x0, x1)
    return y

def rsub(x0: Variable|Array|Scalar, x1: Variable|Array|Scalar) -> Variable:
    return sub(x1, x0)


class Div(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x0: Array = xs[0]
        x1: Array = xs[1]
        y: Array = x0 / x1
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x0: Variable = self.inputs[0]
        x1: Variable = self.inputs[1]
        gy: Variable = gys[0]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return (gx0, gx1)

def div(x0: Variable|Array|Scalar, x1: Variable|Array|Scalar) -> Variable:
    y: Variable
    if isinstance(x0, ScalarTypes):
        if isinstance(x1, ScalarTypes):
            y, = Div()(as_array(x0), as_array(x1))
        else:
            y, = Div()(as_array(x0, npcp(x1)), x1)
    else:
        if isinstance(x1, ScalarTypes):
            y, = Div()(x0, as_array(x1, npcp(x0)))
        else:
            y, = Div()(x0, x1)
    return y

def rdiv(x0: Variable|Array|Scalar, x1: Variable|Array|Scalar) -> Variable:
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c: int):
        self.c: int = c

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        y: Array = x ** self.c
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Variable = self.inputs[0]
        c: int = self.c
        gy: Variable = gys[0]
        gx: Variable = c * x ** (c - 1) * gy
        return (gx,)

def pow(x: Variable, c: int) -> Variable:
    return Pow(c)(x)[0]


class Sin(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y: Array = xp.sin(x)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Variable = self.inputs[0]
        gy: Variable = gys[0]
        gx: Variable = gy * cos(x)
        return (gx,)

def sin(x: Variable) -> Variable:
    return Sin()(x)[0]


class Cos(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y: Array = xp.cos(x)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Variable = self.inputs[0]
        gy: Variable = gys[0]
        gx: Variable = -gy * sin(x)
        return (gx,)

def cos(x: Variable) -> Variable:
    return Cos()(x)[0]


class Tanh(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y = xp.tanh(x)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        y: Variable|None = self.outputs[0]()  # weak ref
        if y is not None:
            gy: Variable = gys[0]
            gx: Variable = gy * (1.0 - y * y)
            return (gx,)
        else:
            log_e(self, "y is None")
            assert y is not None

def tanh(x: Variable) -> Variable:
    return Tanh()(x)[0]


class Sum(Function):
    def __init__(self, axis: int|tuple[int, ...]|None, keepdims: bool):
        self.axis: int|tuple[int, ...]|None = axis
        self.keepdims: bool = keepdims

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        self.x_shape: tuple[int, ...] = x.shape
        y: Array = x.sum(axis = self.axis, keepdims = self.keepdims)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]

        ndim: int = len(self.x_shape)  # batch

        tupled_axis: tuple[int, ...]|None
        if self.axis is None:
            tupled_axis = None
        elif not isinstance(self.axis, tuple):
            tupled_axis = (self.axis,)
        else:
            tupled_axis = self.axis

        shape: tuple[int, ...]
        if not (ndim == 0 or tupled_axis is None or self.keepdims):
            # convert axis = -1 to -1 + ndim
            actual_axis: tuple[int, ...] = tuple([a if a >= 0 else a + ndim for a in tupled_axis])
            cur_shape = list(gy.shape)
            for a in sorted(actual_axis):
                cur_shape.insert(a, 1)
            shape = tuple(cur_shape)
        else:
            shape = gy.shape

        gy = gy.reshape(shape)
        gx: Variable = broadcast_to(gy, self.x_shape)
        return (gx,)

def sum(x: Variable, axis: int|tuple[int, ...]|None = None, keepdims: bool = False):
    return Sum(axis, keepdims)(x)[0]


class SumTo(Function):
    def __init__(self, target_shape: tuple[int, ...]):
        self.target_shape: tuple[int, ...] = target_shape

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        self.x_shape: tuple[int, ...] = x.shape

        ndim: int = len(self.target_shape)
        lead = x.ndim - ndim  # delete dim num, e.g.) x.shape = (1,2,3,4,5), target_shape = (1,4,1), lead = 2
        lead_axis: tuple[int, ...] = tuple(range(lead))  # e.g.) (0, 1), additional dims by broadcast.

        # e.g.) enum:
        #    i  s
        #   [0, 1]
        #   [1, 4]
        #   [2, 1]
        # axis: (0 + lead, 2 + lead) = (2,4)
        axis: tuple[int, ...] = tuple([i + lead for i, s in enumerate(self.target_shape) if s == 1])

        # lead_axis + axis = (0,1,2,4) : sum target axis
        # e.g.)
        #   x.shape: (1,2,3,4,5) -> (1,1,1,4,1)
        y: Array = x.sum(lead_axis + axis, keepdims = True)

        if lead > 0:
            # remove size 1 dim.
            # e.g.)
            #   x.shape : (1,1,1,4,1), lead_axis : (0,1)
            #       -> (1,4,1)
            y = y.squeeze(lead_axis)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        gx: Variable = broadcast_to(gy, self.x_shape)
        return (gx,)

def sum_to(x: Variable, shape: tuple[int, ...]):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)[0]


class BroadcastTo(Function):
    def __init__(self, target_shape: tuple[int, ...]):
        self.target_shape: tuple[int, ...] = target_shape

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        self.x_shape: tuple[int, ...] = x.shape
        y: Array = xp.broadcast_to(x, self.target_shape)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        gx: Variable = sum_to(gy, self.x_shape)
        return (gx,)

def broadcast_to(x: Variable, target_shape: tuple[int, ...]):
    if x.shape == target_shape:
        return as_variable(x)
    return BroadcastTo(target_shape)(x)[0]


class Reshape(Function):
    def __init__(self, target_shape: tuple[int, ...]):
        self.target_shape: tuple[int, ...] = target_shape

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        self.x_shape: tuple[int, ...] = x.shape
        y: Array = x.reshape(self.target_shape)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        gx: Variable = reshape(gy, self.x_shape)
        return (gx,)

def reshape(x: Variable, target_shape: tuple[int, ...]) -> Variable:
    if x.shape == target_shape:
        return as_variable(x)
    return Reshape(target_shape)(x)[0]


class Transpose(Function):
    def __init__(self, axes: tuple[int, ...]|None = None):
        self.axes: tuple[int, ...]|None = axes

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        y: Array = x.transpose(self.axes)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]

        gx: Variable
        if self.axes is None:
            gx = transpose(gy)
        else:
            axes_len = len(self.axes)
            inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
            gx = transpose(gy, inv_axes)
        return (gx,)

def transpose(x: Variable, axes: tuple[int, ...]|None = None) -> Variable:
    return Transpose(axes)(x)[0]


class Matmul(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        W: Array = xs[1]
        # x: (N,D)
        # W: (D,H)
        # y: (N,H)
        y: Array = x.dot(W)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Variable = self.inputs[0]
        W: Variable = self.inputs[1]
        gy: Variable = gys[0]
        gx: Variable = matmul(gy, W.T)
        gW: Variable = matmul(x.T, gy)
        return (gx, gW)

def matmul(x: Variable, W: Variable):
    return Matmul()(x, W)[0]


class MeanSquaredError(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x0: Array = xs[0]
        x1: Array = xs[1]
        diff: Array = x0 - x1
        y: Array = (diff ** 2).sum() / len(diff)  # (N,) -> (1,)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x0: Variable = self.inputs[0]
        x1: Variable = self.inputs[1]
        diff: Variable = x0 - x1
        gy: Variable = gys[0]
        gy = broadcast_to(gy, diff.shape)
        gx0: Variable = gy * diff * 2.0 / len(diff)
        gx1 = -gx0
        return (gx0, gx1)

def mean_squared_error(x0: Variable, x1: Variable):
    return MeanSquaredError()(x0, x1)[0]


class Linear(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        W: Array = xs[1]
        y: Array = x.dot(W)
        if len(xs) > 2:
            b: Array = xs[2]
            y += b
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Variable = self.inputs[0]
        W: Variable = self.inputs[1]

        gy: Variable = gys[0]

        gx: Variable = matmul(gy, W.T)
        gW: Variable = matmul(x.T, gy)

        gb: Variable|None = None
        if len(self.inputs) > 2:
            b: Variable = self.inputs[2]
            if b.data is not None:
                gb = sum_to(gy, b.shape)

        if gb is not None:
            return (gx, gW, gb)
        else:
            return (gx, gW)

def linear(x: Variable, W: Variable, b: Variable|None) -> Variable:
    if b is not None:
        return Linear()(x, W, b)[0]
    else:
        return Linear()(x, W)[0]


class Sigmoid(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y = 1.0 / (1.0 + xp.exp(-x))
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        y: Variable|None = self.outputs[0]()  # weak ref
        if y is not None:
            gx: Variable = gy * y * (1 - y)
            return (gx,)
        else:
            log_e(self, "y is None")
            assert y is not None

def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)[0]


class ReLU(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y: Array = xp.maximum(x, 0.0)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        x: Variable = self.inputs[0]
        mask: Array = x.data > 0
        gx: Variable = gy * mask
        return (gx,)

def relu(x: Variable) -> Variable:
    return ReLU()(x)[0]


class LeakyReLU(Function):
    def __init__(self, slope: float) -> None:
        self.slope: float = slope

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        y: Array = x.copy()
        y[x <= 0] *= self.slope
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        x: Variable = self.inputs[0]
        mask: Array = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx: Variable = gy * mask
        return (gx,)

def leaky_relu(x: Variable, slope: float = 0.2) -> Variable:
    return LeakyReLU(slope)(x)[0]


class Softmax(Function):
    def __init__(self, axis: int = 1) -> None:
        self.axis: int = axis

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        xp: ModuleType = npcp(x)

        y: Array = x - x.max(axis = self.axis, keepdims = True)
        y = xp.exp(y)
        y /= y.sum(axis = self.axis, keepdims = True)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        y: Variable|None = self.outputs[0]()  # weak ref
        if y is not None:
            gx: Variable = y * gy
            sum_dx: Variable = gx.sum(axis = self.axis, keepdims = True)
            gx -= y * sum_dx
            return (gx,)
        else:
            log_e(self, "y is None")
            assert y is not None

def softmax(x: Variable, axis: int = 1) -> Variable:
    return Softmax(axis)(x)[0]


class SoftmaxCrossEntropy(Function):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        t: Array = xs[1]
        N: int = x.shape[0]
        log_z: Array = logsumexp(x, axis = 1)
        log_p: Array = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y: Array = -log_p.sum() / np.float32(N)  # average for N
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Variable = self.inputs[0]
        t: Variable = self.inputs[1]
        N, CLS_NUM = x.shape

        gy: Variable = gys[0]

        gy *= 1.0 / N
        y: Variable = softmax(x)
        xp: ModuleType = npcp(t)
        t_onehot: Array = xp.eye(CLS_NUM, dtype = t.dtype)[t.data]
        gx: Variable = (y - t_onehot) * gy
        return (gx,)

def softmax_cross_entropy(x: Variable, t: Variable) -> Variable:
    return SoftmaxCrossEntropy()(x, t)[0]


class GetItem(Function):
    def __init__(self, slices: slice) -> None:
        self.slices: slice = slices

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        y: Array = x[self.slices]
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        x: Variable = self.inputs[0]
        gx: Variable = get_item_grad(gy, self.slices, x.shape)
        return (gx,)

def get_item(x: Variable, slices: slice) -> Variable:
    return GetItem(slices)(x)[0]


class GetItemGrad(Function):
    def __init__(self, slices: slice, original_shape: tuple[int, ...]) -> None:
        self.slices: slice = slices
        self.original_shape: tuple[int, ...] = original_shape

    @override
    def forward(self, gys: tuple[Array, ...]) -> tuple[Array, ...]:
        gy: Array = gys[0]
        xp: ModuleType = npcp(gy)

        gx: Array = xp.zeros(self.original_shape, dtype = gy.dtype)
        xp.add.at(gx, self.slices, gy)  # type: ignore[arg-type]
        return (gx,)

    @override
    def backward(self, xs: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Variable = xs[0]
        y = get_item(x, self.slices)
        return (y,)

def get_item_grad(gy: Variable, slices: slice, original_shape: tuple[int, ...]) -> Variable:
    return GetItemGrad(slices, original_shape)(gy)[0]



def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4):
    xp: ModuleType = npcp(x)

    x0 = Variable(as_array(x.data - eps, xp))
    x1 = Variable(as_array(x.data + eps, xp))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def as_array(x: Scalar|Array, xp: ModuleType = np):
    if xp.isscalar(x):
        return xp.array(x)
    return x

def as_variable(x: Array|Variable) -> Variable:
    if isinstance(x, Variable):
        return x
    return Variable(x)

def logsumexp(x: Array, axis: int = 1) -> Array:
    xp: ModuleType = npcp(x)

    m: Array = x.max(axis = axis, keepdims = True)
    y: Array = x - m
    xp.exp(y, out = y)  # over write
    s: Array = y.sum(axis = axis, keepdims = True)
    xp.log(s, out = s)  # over write
    m += s
    return m

def accuracy(actual: Variable, expect: Variable) -> Variable:
    actual = as_variable(actual)
    expect = as_variable(expect)

    # expect:
    #   [2, 1, 2, 3]
    #
    # actual:
    #   [[0, 1, 2, 0],
    #    [1, 2, 0, 1],
    #    [3, 1, 0, 2],
    #    [1, 0, 1, 2]]
    # -> argmax axis 1
    #   [[2],
    #    [1],
    #    [0],
    #    [3]]
    # -> reshape
    #   [2, 1, 0, 3]
    predict: Array = actual.data.argmax(axis = 1).reshape(expect.shape)

    # result: [True, True, False, True] == [1, 1, 0, 1]
    result: Array = (predict == expect.data)

    accuracy: float = result.mean()
    return Variable(as_array(accuracy))

