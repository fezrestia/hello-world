from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from types import ModuleType

if TYPE_CHECKING:
    from .Function import Function

from .Log import log_d, log_e
from .Type import Scalar, Array, ArrayTypes
from .Config import use_config
from .cuda import npcp, as_np, as_cp

class Variable:
    __array_priority__ = 100  # has priority to numpy.ndarray add/mul

    def __init__(self, data: Array, name = None):
        if data is not None:
            if not isinstance(data, ArrayTypes):
                raise TypeError(f"{type(data)} is not supported.")

        self.data: Array = data
        self.grad: Variable|None = None

        self.creator: Function|None = None
        self.generation: int = 0

        self.name: str|None = name

    def to_cpu(self) -> None:
        self.data = as_np(self.data)

    def to_gpu(self) -> None:
        self.data = as_cp(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "Variable(None)"

        p = str(self.data).replace(f"\n", f"\n         ")
        return f"Variable({p}), dtype:{self.dtype}, shape:{self.shape}"


    # for mypy
    def __add__(self: Variable, other: object):
        pass
    # for mypy
    def __radd__(self: Variable, other: object):
        pass
    # for mypy
    def __mul__(self: Variable, other: object):
        pass
    # for mypy
    def __rmul__(self: Variable, other: object):
        pass
    # for mypy
    def __sub__(self: Variable, other: object):
        pass
    # for mypy
    def __rsub__(self: Variable, other: object):
        pass
    # for mypy
    def __truediv__(self: Variable, other: object):
        pass
    # for mypy
    def __rtruediv__(self: Variable, other: object):
        pass
    # for mypy
    def __neg__(self: Variable):
        pass
    # for mypy
    def __pow__(self: Variable, c: int):
        pass


    def set_creator(self, func: Function):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, keep_grad = False, create_graph = False) -> None:
        if self.grad is None:
            xp: ModuleType = npcp(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        func_queue: list[Function] = []
        func_set: set[Function] = set()

        def add_func(f: Function):
            if f not in func_set:
                func_queue.append(f)
                func_set.add(f)
                func_queue.sort(key = lambda x: x.generation)

        if self.creator is not None:
            add_func(self.creator)

        while func_queue:
            f: Function = func_queue.pop()  # get and remove last element (last generation)

            gys: list[Variable] = []
            for o_ref in f.outputs:
                o: Variable|None = o_ref()
                if o is not None and o.grad is not None:
                    gys.append(o.grad)
                else:
                    log_e(self, "output.grad is None.")

            with use_config("enable_backprop", create_graph):
                gxs: tuple[Variable, ...] = f.backward(tuple(gys))

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx  # cut reference from other Variant.grad

                    if x.creator is not None:
                        add_func(x.creator)
                    else:
                        log_d(self, "x.creator is None, maybe root param.")

            # only root input grads can survive.
            if not keep_grad:
                for y_ref in f.outputs:
                    y: Variable|None = y_ref()
                    if y is not None:
                        y.grad = None

    def clear_grad(self):
        self.grad = None


    def reshape(self, target_shape: tuple[int, ...]) -> Variable:
        from .Function import reshape
        return reshape(self, target_shape)

    def transpose(self, axes: tuple[int, ...]|None = None) -> Variable:
        from .Function import transpose
        return transpose(self, axes)

    @property
    def T(self) -> Variable:
        return self.transpose()

    def sum(self, axis: int|tuple[int, ...]|None = None, keepdims: bool = False):
        from .Function import sum
        return sum(self, axis, keepdims)

