import numpy as np
import math
from collections.abc import Iterator
from typing import Self

from .DataSet import DataSet

class DataLoader:
    def __init__(self, dataset: DataSet, batch_size: int, shuffle: bool = True) -> None:
        self.dataset: DataSet = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.data_size: int = len(dataset)
        self.max_iter: int = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self) -> None:
        self.iteration: int = 0

        self.index: np.ndarray
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i: int = self.iteration
        batch_size: int = self.batch_size
        batch_index: np.ndarray = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x: np.ndarray = np.array([data[0] for data in batch])
        t: np.ndarray = np.array([data[1] for data in batch])

        self.iteration += 1

        return (x, t)

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        return self.__next__()

