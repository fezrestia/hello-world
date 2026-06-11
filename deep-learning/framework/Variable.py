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

class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.data: np.ndarray = data
        self.grad: np.ndarray|None = None

        self.creator: Function|None = None

    def set_creator(self, func: Function):
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function|None] = [self.creator]
        while funcs:
            f: Function|None = funcs.pop()  # get and remove last element

            if f is not None:
                x: Variable = f.input
                y: Variable = f.output

                if y.grad is not None:
                    x.grad = f.backward(y.grad)

                    if x.creator is not None:
                        funcs.append(x.creator)
                    else:
                        Log.d(self, "x.creator is None, maybe root param.")
                else:
                    Log.e(self, "y.grad is None")
            else:
                Log.e(self, "f is None")

