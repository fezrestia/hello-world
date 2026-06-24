import numpy as np
from numpy.typing import DTypeLike
import weakref
from weakref import ReferenceType
from typing import override
from collections.abc import Iterator
from types import ModuleType
import os

from .Variable import Variable
from .Parameter import Parameter
from .Function import linear, conv2d, tanh, sigmoid
from .Log import log_d, log_e
from .Type import Array
from .cuda import npcp, gpu_enabled

class Layer:
    DUMMY_W: Array = np.empty((0,))

    def __init__(self) -> None:
        self._params: set[str] = set()

    # called on instance variable is set
    def __setattr__(self, name: str, value: object):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: Variable) -> tuple[Variable, ...]:
        outputs: tuple[Variable, ...] = self.forward(*inputs)

        self.inputs: tuple[ReferenceType[Variable], ...] = tuple([weakref.ref(x) for x in inputs])
        self.outputs: tuple[ReferenceType[Variable], ...] = tuple([weakref.ref(y) for y in outputs])

        return outputs

    def forward(self, *inputs: Variable) -> tuple[Variable, ...]:
        raise NotImplementedError()

    def params(self) -> Iterator[Parameter]:
        for name in self._params:
            obj: object = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            elif isinstance(obj, Parameter):
                yield obj
            else:
                log_e(self, "obj is NOT Parameter or Layer")
                assert isinstance(obj, (Layer, Parameter))

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()

    def to_cpu(self) -> None:
        for param in self.params():
            param.to_cpu()

    def to_gpu(self) -> None:
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, key_vs_param: dict[str, Parameter], parent_key: str = "") -> None:
        for name in self._params:
            obj: object = self.__dict__[name]
            if parent_key:
                key: str = f"{parent_key}/{name}"
            else:
                key = name

            if isinstance(obj, Layer):
                obj._flatten_params(key_vs_param, key)
            elif isinstance(obj, Parameter):
                key_vs_param[key] = obj
            else:
                log_e(self, "obj is NOT Parameter or Layer")
                assert isinstance(obj, (Layer, Parameter))

    def save_weights(self, path: str) -> None:
        if gpu_enabled:
            self.to_cpu()

        key_vs_param: dict[str, Parameter] = {}
        self._flatten_params(key_vs_param)

        key_vs_array: dict[str, np.ndarray] = {key: param.data for key, param in key_vs_param.items() if param is not None}

        try:
            np.savez_compressed(path, allow_pickle = True, **key_vs_array)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path: str) -> None:
        npz: dict[str, np.ndarray] = np.load(path)
        key_vs_param: dict[str, Parameter] = {}
        self._flatten_params(key_vs_param)
        for key, param in key_vs_param.items():
            param.data = npz[key]

        if gpu_enabled:
            self.to_gpu()


class Linear(Layer):
    def __init__(self, out_size: int, nobias = False, dtype = np.float32, in_size = None):
        super().__init__()

        self.in_size: int|None = in_size
        self.out_size: int = out_size
        self.dtype: np.dtype = dtype

        self.W: Parameter = Parameter(Layer.DUMMY_W, name = "W")
        if self.in_size is not None:
            self._init_W(np)

        self.b: Parameter|None
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_size, dtype = dtype), name = "b")

    def _init_W(self, xp: ModuleType) -> None:
        if self.in_size is not None:
            I:int = self.in_size
        else:
            log_e(self, "self.in_size is None")
            assert self.in_size is not None
        O: int = self.out_size

        W_data: Array = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1.0 / I)
        self.W.data = W_data

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]
        xp: ModuleType = npcp(x)

        if self.W.data.shape == (0,):  # check Layer.DUMMY_W
            self.in_size = x.shape[1]  # x:(N, D), W: (D, H)
            self._init_W(xp)

        y: Variable = linear(x, self.W, self.b)
        return (y,)


class Conv2d(Layer):
    def __init__(
            self,
            out_channels: int,
            kernel_size: tuple[int, int],
            stride: tuple[int, int] = (1, 1),
            padding: tuple[int, int] = (0, 0),
            nobias: bool = False,
            dtype: DTypeLike = np.float32,
            in_channels: int|None = None,
    ) -> None:
        super().__init__()

        self.in_channels: int|None = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = kernel_size
        self.stride: tuple[int, int] = stride
        self.padding: tuple[int, int] = padding
        self.dtype: DTypeLike = dtype

        self.W: Parameter = Parameter(None, name = "W")
        if in_channels is not None:
            self._init_W()

        self.b: Parameter|None
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_channels, dtype = dtype), name = "b")

    def _init_W(self, xp: ModuleType = np) -> None:
        if self.in_channels is None:
            log_e(self, "self.in_channels is None.")
            assert self.in_channels is not None

        C: int = self.in_channels
        OC: int = self.out_channels
        KH, KW = self.kernel_size
        scale = np.sqrt(1.0 / (C * KH * KW))
        W_data: Array = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp: ModuleType = npcp(x)
            self._init_W(xp)

        y: Variable = conv2d(x, self.W, self.b, self.stride, self.padding)
        return (y,)


class RNN(Layer):
    def __init__(self, hidden_size: int, in_size: int|None = None) -> None:
        super().__init__()

        self.x2h: Layer = Linear(hidden_size, in_size = in_size)
        self.h2h: Layer = Linear(hidden_size, in_size = in_size, nobias = True)
        self.h: Variable|None = None

    def reset_state(self) -> None:
        self.h = None

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        h_new: Variable
        if self.h is None:
            h_new = tanh(self.x2h(x)[0])
        else:
            h_new = tanh(self.x2h(x)[0] + self.h2h(self.h)[0])

        self.h = h_new
        return (h_new,)


class LSTM(Layer):
    def __init__(self, hidden_size: int, in_size: int|None = None) -> None:
        super().__init__()

        H: int = hidden_size
        I: int|None = in_size

        self.x2f = Linear(H, in_size = I)
        self.x2i = Linear(H, in_size = I)
        self.x2o = Linear(H, in_size = I)
        self.x2u = Linear(H, in_size = I)

        self.h2f = Linear(H, in_size = H, nobias = True)
        self.h2i = Linear(H, in_size = H, nobias = True)
        self.h2o = Linear(H, in_size = H, nobias = True)
        self.h2u = Linear(H, in_size = H, nobias = True)

        self.reset_state()

    def reset_state(self) -> None:
        self.h: Variable|None = None
        self.c: Variable|None = None

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        f: Variable
        i: Variable
        o: Variable
        u: Variable
        if self.h is None:
            f = sigmoid(self.x2f(x)[0])
            i = sigmoid(self.x2i(x)[0])
            o = sigmoid(self.x2o(x)[0])
            u = tanh(self.x2u(x)[0])
        else:
            f = sigmoid(self.x2f(x)[0] + self.h2f(self.h)[0])
            i = sigmoid(self.x2i(x)[0] + self.h2i(self.h)[0])
            o = sigmoid(self.x2o(x)[0] + self.h2o(self.h)[0])
            u = tanh(self.x2u(x)[0] + self.h2u(self.h)[0])

        c_new: Variable
        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new: Variable = o * tanh(c_new)

        self.h = h_new
        self.c = c_new
        return (h_new,)

