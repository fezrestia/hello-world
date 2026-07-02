import sys
from pathlib import Path
from typing import override, Any
from types import ModuleType
import numpy as np
from numpy.typing import DTypeLike
import weakref
from weakref import ReferenceType
from collections.abc import Callable

from .Variable import Variable
from .Config import Config
from .Type import Scalar, ScalarTypes, Array, ArrayTypes
from .Log import log_d, log_e
from .cuda import npcp, gpu_enabled, use_np, use_cp
from .utils import get_conv_outsize, get_deconv_outsize

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


def max_backward_shape(x: Array, axis: int|tuple[int, ...]|None) -> tuple[int, ...]:
    if axis is None:
        axis = tuple(range(x.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape: tuple[int, ...] = tuple(s if ax not in axis else 1 for ax, s in enumerate(x.shape))
    return shape

class Max(Function):
    def __init__(
            self,
            axis:int|tuple[int, ...]|None = None,
            keepdims: bool = False,
    ) -> None:
        self.axis: int|tuple[int, ...]|None = axis
        self.keepdims: bool = keepdims

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        y: Array = x.max(axis = self.axis, keepdims=self.keepdims)
        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        x: Variable = self.inputs[0]
        y: Variable|None = self.outputs[0]()  # weakref

        if y is None:
            raise Exception("Unexpected, y is None.")

        shape: tuple[int, ...] = max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond: Array = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return (gy * cond,)

def var_max(
        x: Variable,
        axis: int|tuple[int, ...]|None = None,
        keepdims: bool = False,
) -> Variable:
    return Max(axis, keepdims)(x)[0]


class Min(Max):
    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]
        y: Array = x.min(axis = self.axis, keepdims = self.keepdims)
        return (y,)

def var_min(
        x: Variable,
        axis: int|tuple[int, ...]|None = None,
        keepdims: bool = False,
) -> Variable:
    return Min(axis, keepdims)(x)[0]


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

def matmul(x: Variable, W: Variable) -> Variable:
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


class DropOut(Function):
    def __init__(self, dropout_ratio: float = 0.5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask: Array
        self.scale: Array

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]

        if Config.train:
            xp: ModuleType = npcp(x)

            self.mask = xp.random.rand(*x.shape) > self.dropout_ratio
            self.scale = xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            y: Array = x * self.mask / self.scale
            return (y,)
        else:
            return (x,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]

        gx: Variable = gy * self.mask / self.scale
        return (gx,)

def dropout(x: Variable, dropout_ratio: float = 0.5) -> Variable:
    return DropOut(dropout_ratio)(x)[0]


class GetItem(Function):
    def __init__(self, slices: slice|tuple[slice|int, ...]|Any) -> None:
        self.slices: slice|tuple[slice|int, ...]|Any = slices

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

def get_item(x: Variable, slices: slice|tuple[slice|int, ...]|Any) -> Variable:
    return GetItem(slices)(x)[0]


class GetItemGrad(Function):
    def __init__(
            self,
            slices: slice|tuple[slice|int, ...]|Any,
            original_shape: tuple[int, ...],
    ) -> None:
        self.slices: slice|tuple[slice|int, ...]|Any = slices
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

def get_item_grad(
        gy: Variable,
        slices: slice|tuple[slice|int, ...]|Any,
        original_shape: tuple[int, ...],
) -> Variable:
    return GetItemGrad(slices, original_shape)(gy)[0]



# img: (N, C, H, W)
# kernel_size: (KH, KW)
# return: (N * OH * OW, C * KH * KW)
def img2col(
        img: Array,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        to_matrix: bool = True,
) -> Array:
    N, C, H, W = img.shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp: ModuleType
    col: Array
    if gpu_enabled:
        xp = use_cp()

        dh: int = 1
        dw: int = 1

        col = xp.empty((N, C, KH, KW, OH, OW), dtype = img.dtype)

        xp.ElementwiseKernel(
                "raw T img, int32 h, int32 w, int32 out_h, int32 out_w,"
                "int32 kh, int32 kw, int32 sh, int32 sw, int32 ph, int32 pw,"
                "int32 dh, int32 dw",
                "T col",
                '''
                    int c0 = i / (kh * kw * out_h * out_w);
                    int ky = i / (kw * out_h * out_w) % kh;
                    int kx = i / (out_h * out_w) % kw;
                    int out_y = i / out_w % out_h;
                    int out_x = i % out_w;
                    int in_y = ky * dh + out_y * sh - ph;
                    int in_x = kx * dw + out_x * sw - pw;
                    if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
                        col = img[in_x + w * (in_y + h * c0)];
                    } else {
                        col = 0;
                    }
                ''',
                "img2col")(img, H, W, OH, OW, KH, KW, SH, SW, PH, PW, dh, dw, col)
    else:
        xp = use_np()

        img = np.pad(
                img,
                ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),  # padding for N, C, H, W
                mode = "constant",
                constant_values = (0,),
        )

        col = np.ndarray((N, C, KH, KW, OH, OW), dtype = img.dtype)

        for y in range(KH):
            y_lim: int = y + SH * OH
            for x in range(KW):
                x_lim: int = x + SW * OW
                col[:, :, y, x, :, :] = img[:, :, y:y_lim:SH, x:x_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col

# col:(N * OH * OW, C * KH * KW )
# img_shape: (N, C, H, W)
# return: (N, C, H, W)
def col2img(
        col: Array,
        img_shape: tuple[int, int, int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        from_matrix: bool = True,
) -> Array:
    N, C, H, W = img_shape
    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if from_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    img: Array
    xp: ModuleType
    if gpu_enabled:
        xp = use_cp()

        dw: int = 1
        dh: int = 1

        img = xp.empty((N, C, H, W), dtype = col.dtype)

        xp.ElementwiseKernel(
                "raw T col, int32 H, int32 W, int32 OH, int32 OW,"
                "int32 KH, int32 KW, int32 SH, int32 SW, int32 PH, int32 PW,"
                "int32 dh, int32 dw",
                "T img",
                '''
                    int c0 = i / (H * W);
                    int y = i / W % H;
                    int x = i % W;
                    T val = 0;
                    for (int ky = 0; ky < KH; ++ky) {
                        int out_y = (y + PH - ky * dh);
                        if (0 > out_y || out_y >= OH * SH) continue;
                        if (out_y % SH != 0) continue;
                        out_y /= SH;
                        for (int kx = 0; kx < KW; ++kx) {
                            int out_x = (x + PW - kx * dw);
                            if (0 > out_x || out_x >= OW * SW) continue;
                            if (out_x % SW != 0) continue;
                            out_x /= SW;
                            int k = out_y + OH * (kx + KW * (ky + KH * c0));
                            val = val + col[out_x + OW * k];
                        }
                    }
                    img = val;
                ''',
                "col2img")(col, H, W, OH, OW, KH, KW, SH, SW, PH, PW, dh, dw, img)

        return img
    else:
        xp = use_np()

        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype = col.dtype)

        for y in range(KH):
            y_lim = y + SH * OH
            for x in range(KW):
                x_lim = x + SW * OW
                img[:, :, y:y_lim:SH, x:x_lim:SW] += col[:, :, y, x, :, :]

        return img[:, :, PH:H + PH, PW:W + PW]


class Conv2d(Function):
    def __init__(
            self,
            stride: tuple[int, int] = (1, 1),
            padding: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()

        self.stride: tuple[int, int] = stride
        self.padding: tuple[int, int] = padding

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]  # (N, C, H, W)
        W: Array = xs[1]  # (OC, C, KH, KW)
        b: Array|None = None  # (OC,)
        if len(xs) > 2:
            b = xs[2]

        xp: ModuleType = npcp(x)

        _, _, KH, KW = W.shape
        col: Array = img2col(  # (N, C, KH, KW, OH, OW)
                x,
                (KH, KW),  # kernel
                self.stride,
                self.padding,
                to_matrix = False,
        )

        # calc tensor on axis 1, 2, 3 for col and W.
        # col : (N,  C, KH, KW, OH, OW)
        # W   : (OC, C, KH, KW)
        # axis 1, 2, 3 = C, KH, KW
        # y   : (N, OH, OW, OC)
        y: Array = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))

        if b is not None:
            y += b

        # y : (N, OC, OH, OW)
        y = xp.transpose(y, (0, 3, 1, 2))

        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Array = self.inputs[0]  # (N, C, H, W)
        Weight: Array = self.inputs[1]  # (OC, C, KH, KW)
        bias: Array|None = None  # (OC,)
        if len(self.inputs) > 2:
            bias = self.inputs[2]

        _, _, H, W = x.shape

        gy: Variable = gys[0]
        gx: Variable = deconv2d(gy, as_variable(Weight), as_variable(bias), stride = self.stride, padding = self.padding, outsize = (H, W))
        gW: Variable = Conv2DGradW(self)(x, gy)[0]

        if bias is not None:
            gb: Variable = gy.sum(axis = (0, 2, 3))
            return (gx, gW, gb)
        else:
            return (gx, gW)

# x : (N, C, H, W)
# W : (OC, C, KH, KW)
# b : (OC,)
# return : (N, OC, OH, OW)
def conv2d(
        x: Variable,
        W: Variable,
        b: Variable|None = None,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
) -> Variable:
    return Conv2d(stride, padding)(x, W, b)[0]


class Deconv2d(Function):
    def __init__(
            self,
            stride: tuple[int, int] = (1, 1),
            padding: tuple[int, int] = (0, 0),
            outsize: tuple[int, int]|None = None,
    ) -> None:
        super().__init__()

        self.stride: tuple[int, int] = stride
        self.padding: tuple[int, int] = padding
        self.outsize: tuple[int, int]|None = outsize

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]  # (N, C, H, W)
        Weight: Array = xs[1]  # (C, OC, KH, KW)
        bias: Array|None = None  # (OC,)
        if len(xs) > 2:
            bias = xs[2]

        xp: ModuleType = npcp(x)

        SH, SW = self.stride
        PH, PW = self.padding
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            OH: int = get_deconv_outsize(H, KH, SH, PH)
            OW: int = get_deconv_outsize(W, KW, SW, PW)
        else:
            OH, OW = self.outsize
        img_shape: tuple[int, int, int, int] = (N, OC, OH, OW)

        # Weight : (C, OC, KH, KW)
        # x      : (N,  C,  H,  W)
        # col    : (OC, KH, KW, N, H, W)
        col: Array = xp.tensordot(Weight, x, (0, 1))  # Weight axis 0, and, x axis 1
        # col : (N, OC, KH, KW, H, W)
        col = xp.transpose(col, (3, 0, 1, 2, 4, 5))

        # y : (N, OC, OH, OW) = img_shape
        y: Array = col2img(col, img_shape, (KH, KW), self.stride, self.padding, from_matrix = False)

        if bias is not None:
            # bias: (OC,) -> (1, OC, 1, 1)
            y += bias.reshape((1, bias.size, 1, 1))

        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Array = self.inputs[0]
        W: Array = self.inputs[1]
        b: Array|None = None
        if len(self.inputs) > 2:
            b = self.inputs[2]

        gy: Array = gys[0]

        gx: Array = conv2d(as_variable(gy), as_variable(W), as_variable(b), stride = self.stride, padding = self.padding)

        gW: Array = Conv2DGradW(self)(gy, x)[0]

        if b is not None:
            gb: Array = gy.sum(axis = (0, 2, 3))
            return (as_variable(gx), as_variable(gW), as_variable(gb))
        else:
            return (as_variable(gx), as_variable(gW))

# x : (N, C, H, W)
# W : (C, OC, KH, KW)
# b : (OC,)
# return : (N, OC, OH, OW)
def deconv2d(
        x: Variable,
        W: Variable,
        b: Variable|None = None,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        outsize: tuple[int, int]|None = None,
) -> Variable:
    return Deconv2d(stride, padding, outsize)(x, W, b)[0]


class Conv2DGradW(Function):
    def __init__(self, conv2d) -> None:
        W: Array = conv2d.inputs[1]
        _N, _C, KH, KW = W.shape

        self.kernel_size = (KH, KW)
        self.stride = conv2d.stride
        self.padding = conv2d.padding

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]  # (N, C, H, W)
        gy: Array = xs[1]  # (N, C, H, W)

        xp: ModuleType = npcp(x)

        # col: (N, OC, KH, KW, H, W)
        col: Array = img2col(x, self.kernel_size, self.stride, self.padding, to_matrix = False)

        # gW : (C, OC, KH, KW)
        gW: Array = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))  # target = N, H, W
        return (gW,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        x: Array = self.inputs[0]  # (N, C, H, W)
        gy: Array = self.inputs[0]  # (N, C, H, W)
        gW: Array = self.outputs[0]()  # (C, OC, KH, KW)

        _N, _C, H, W = x.shape

        gx: Array = deconv2d(as_variable(gy), as_variable(gW), stride = self.stride, padding = self.padding, outsize = (H, W))
        ggy: Array = conv2d(as_variable(x), as_variable(gW), stride = self.stride, padding = self.padding)
        return (as_variable(gx), as_variable(ggy))


class Pooling(Function):
    def __init__(
            self,
            kernel_size: tuple[int, int],
            stride: tuple[int, int] = (1, 1),
            padding: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()

        self.kernel_size: tuple[int, int] = kernel_size
        self.stride: tuple[int, int] = stride
        self.padding: tuple[int, int] = padding

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]

        col: Array = img2col(x, self.kernel_size, self.stride, self.padding, to_matrix = False)
        N, C, KH, KW, OH, OW = col.shape

        col = col.reshape(N, C, KH * KW, OH, OW)

        self.indexes: Array = col.argmax(axis = 2)  # max index in KH*KW
        y: Array = col.max(axis = 2)  # max value in KH*KW

        return (y,)

    @override
    def backward(self, gys: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gy: Variable = gys[0]
        return (Pooling2DGrad(self)(gy)[0],)

def pooling(
        x: Variable,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
) -> Variable:
    return Pooling(kernel_size, stride, padding)(x)[0]


class Pooling2DGrad(Function):
    def __init__(self, pooling: Pooling) -> None:
        self.pooling: Pooling = pooling
        self.kernel_size: tuple[int, int] = pooling.kernel_size
        self.stride: tuple[int, int] = pooling.stride
        self.padding: tuple[int, int] = pooling.padding
        self.input_shape: tuple[int, int, int, int] = pooling.inputs[0].shape  # (N, C, H, W)
        self.dtype: DTypeLike = pooling.inputs[0].dtype
        self.indexes: Array = pooling.indexes

    @override
    def forward(self, gys: tuple[Array, ...]) -> tuple[Array, ...]:
        gy: Array = gys[0]
        xp: ModuleType = npcp(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = self.kernel_size

        # col : (N, C, OH, OW, KH, KW)
        col: Array = xp.zeros((N * C * OH * OW * KH * KW), dtype = self.dtype)

        # index of max value for each kernel
        indexes: Array = (
                self.indexes.ravel()  # for each kernel, max value index
                + xp.arange(0, self.indexes.size * KH * KW, KH * KW)  # for each kernel, head index
        )

        col[indexes] = gy.ravel()
        col = col.reshape(N, C, OH, OW, KH, KW)
        col = xp.transpose(col, (0, 1, 4, 5, 2, 3))  # (N, C, KH, KW, OH, OW)

        gx: Array = col2img(col, (N, C, H, W), self.kernel_size, self.stride, self.padding, from_matrix = False)
        return (gx,)

    @override
    def backward(self, gxs: tuple[Variable, ...]) -> tuple[Variable, ...]:
        gx: Variable = gxs[0]
        return (Pooling2DWithIndexes(self.pooling)(gx)[0],)


class Pooling2DWithIndexes(Function):
    def __init__(self, pooling: Pooling) -> None:
        self.kernel_size: tuple[int, int] = pooling.kernel_size
        self.stride: tuple[int, int] = pooling.stride
        self.padding: tuple[int, int] = pooling.padding
        self.input_shape: tuple[int, int, int, int] = pooling.inputs[0].shape  # (N, C, H, W)
        self.dtype: DTypeLike = pooling.inputs[0].dtype
        self.indexes: Array = pooling.indexes

    @override
    def forward(self, xs: tuple[Array, ...]) -> tuple[Array, ...]:
        x: Array = xs[0]

        col: Array = img2col(x, self.kernel_size, self.stride, self.padding, to_matrix = False)
        N, C, KH, KW, OH, OW = col.shape

        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2)  # (N, C, OH, OW, KH*KW)
        col = col.reshape(-1, KH * KW)  # (N*C*OH*OW, KH*KW)

        indexes: Array = self.indexes.ravel()  # (N*C*OH*OW,) = window count

        col = col[np.arange(len(indexes)), indexes]  # [[0, 1, 2, ..], [max value indexes]] = (N*C*OH*OW,)
        col = col.reshape(N, C, OH, OW)

        return (col,)



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

