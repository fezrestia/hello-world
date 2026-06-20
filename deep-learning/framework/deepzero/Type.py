from typing import TypeAlias
import numpy as np

from .cuda import gpu_enabled

if gpu_enabled:
    import cupy as cp
else:
    cp = np

Scalar: TypeAlias = int|float|np.number|cp.number|bool
ScalarTypes: tuple[type, ...] = (int, float, np.number, cp.number, bool)

Array: TypeAlias = np.ndarray|cp.ndarray
ArrayTypes: tuple[type, ...] = (np.ndarray, cp.ndarray)

