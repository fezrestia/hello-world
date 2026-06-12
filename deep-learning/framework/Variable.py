from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING

# include same dir layer.
same_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, same_dir)

if TYPE_CHECKING:
    from Function import Function

from Log import Log
from Type import Scalar

class Variable:
    __array_priority__ = 100  # has priority to numpy.ndarray add/mul

    def __init__(self, data: np.ndarray, name = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data: np.ndarray = data
        self.grad: np.ndarray|None = None

        self.creator: Function|None = None
        self.generation: int = 0

        self.name: str|None = name

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

    def __mul__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        from Function import mul
        return mul(self, other)

    def __rmul__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        return self.__mul__(other)

    def __add__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        from Function import add
        return add(self, other)

    def __radd__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        return self.__add__(other)

    def __neg__(self) -> Variable:
        from Function import neg
        return neg(self)

    def __sub__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        from Function import sub
        return sub(self, other)

    def __rsub__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        from Function import sub
        return sub(other, self)

    def __truediv__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        from Function import div
        return div(self, other)

    def __rtruediv__(self, other: Variable|np.ndarray|Scalar) -> Variable:
        from Function import div
        return div(other, self)

    def __pow__(self, c: int) -> Variable:
        from Function import pow
        return pow(self, c)


    def set_creator(self, func: Function):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, keep_grad = False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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

            gys: list[np.ndarray] = []
            for o_ref in f.outputs:
                o: Variable|None = o_ref()
                if o is not None and o.grad is not None:
                    gys.append(o.grad)
                else:
                    Log.e(self, "output.grad is None.")
            gxs: tuple[np.ndarray, ...] = f.backward(tuple(gys))

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # cut reference from other Variant.grad

                if x.creator is not None:
                    add_func(x.creator)
                else:
                    Log.d(self, "x.creator is None, maybe root param.")

            # only root input grads can survive.
            if not keep_grad:
                for y_ref in f.outputs:
                    y: Variable|None = y_ref()
                    if y is not None:
                        y.grad = None

    def clear_grad(self):
        self.grad = None

