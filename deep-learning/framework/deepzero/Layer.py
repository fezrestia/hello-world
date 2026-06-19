import numpy as np
import weakref
from weakref import ReferenceType
from typing import override
from collections.abc import Iterator

from .Variable import Variable
from .Parameter import Parameter
from .Function import linear
from .Log import log_d, log_e

class Layer:
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


class Linear(Layer):
    def __init__(self, out_size: int, nobias = False, dtype = np.float32, in_size = None):
        super().__init__()

        self.in_size: int|None = in_size
        self.out_size: int = out_size
        self.dtype: np.dtype = dtype

        self.W: Parameter = Parameter(np.zeros((1, 1)), name = "W")
        if self.in_size is not None:
            self._init_W()

        self.b: Parameter|None
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_size, dtype = dtype), name = "b")

    def _init_W(self) -> None:
        if self.in_size is not None:
            I:int = self.in_size
        else:
            log_e(self, "self.in_size is None")
            assert self.in_size is not None
        O: int = self.out_size

        W_data: np.ndarray = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1.0 / I)
        self.W.data = W_data

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        if self.in_size is None:
            self.in_size = x.shape[1]  # x:(N, D), W: (D, H)
            self._init_W()

        y: Variable = linear(x, self.W, self.b)
        return (y,)

