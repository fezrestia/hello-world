from __future__ import annotations

import numpy as np
from types import ModuleType
from typing import TYPE_CHECKING, Any

gpu_enabled: bool
try:
    import cupy as cp
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

from .Type import Array, ArrayTypes, Scalar, ScalarTypes
if TYPE_CHECKING:
    from .Variable import Variable
from .Log import log_d, log_e

def use_np() -> ModuleType:
    return np

def use_cp() -> ModuleType:
    if not gpu_enabled:
        raise Exception("cupy is not loaded.")
    return cp

def npcp(x: Variable|Array|Scalar) -> ModuleType:
    if not gpu_enabled:
        return np

    from .Variable import Variable  # escape from circular import

    if isinstance(x, Variable):
        return cp.get_array_module(x.data)
    elif isinstance(x, ArrayTypes):
        return cp.get_array_module(x)
    elif isinstance(x, ScalarTypes):
        return np
    else:
        raise Exception(f"Unexpected x type = {type(x)}")

def as_np(x: Array|Scalar) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        return cp.asnumpy(x)

def as_cp(x: Array) -> cp.ndarray:
    if not gpu_enabled:
        raise Exception("cupy is not loaded.")
    return cp.asarray(x)

