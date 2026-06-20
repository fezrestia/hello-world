import numpy as np
import math
from collections.abc import Iterator
from typing import Self
from types import ModuleType

from .DataSet import DataSet
from .cuda import use_np, use_cp
from .Type import Array

class DataLoader:
    def __init__(self, dataset: DataSet, batch_size: int, shuffle: bool = True, use_gpu: bool = False) -> None:
        self.dataset: DataSet = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.data_size: int = len(dataset)
        self.max_iter: int = math.ceil(self.data_size / batch_size)
        self.use_gpu: bool = use_gpu

        self.reset()

    def reset(self) -> None:
        self.iteration: int = 0

        self.index: Array
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[Array, Array]:
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i: int = self.iteration
        batch_size: int = self.batch_size
        batch_index: Array = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp: ModuleType = use_cp() if self.use_gpu else use_np()
        x: Array = xp.array([data[0] for data in batch])
        t: Array = xp.array([data[1] for data in batch])

        self.iteration += 1

        return (x, t)

    def next(self) -> tuple[Array, Array]:
        return self.__next__()

    def to_cpu(self) -> None:
        self.use_gpu = False

    def to_gpu(self) -> None:
        self.use_gpu = True

