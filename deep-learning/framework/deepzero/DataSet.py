import numpy as np
from numpy.typing import DTypeLike
from typing import override, Callable, TypeVar
from pathlib import Path
import matplotlib.pyplot as plt
import gzip
from types import ModuleType

from .Log import log_d, log_e
from .dataset import spiral
from .Type import Array

class DataSet:
    def __init__(self,
            train: bool = True,
            transform: Callable[[np.ndarray], np.ndarray]|None = None,
            target_transform: Callable[[np.ndarray], np.ndarray]|None = None,
    ) -> None:
        self.train: bool = train

        self.transform: Callable[[np.ndarray], np.ndarray] = lambda x: x
        self.target_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x
        if transform is not None:
            self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform

        self.data: np.ndarray|None = None
        self.label: np.ndarray|None = None
        self.prepare()

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray|None]:
        assert np.isscalar(index)

        if self.data is not None:
            if self.label is None:
                return self.transform(self.data[index]), None  # type: ignore[index]
            else:
                return self.transform(self.data[index]), self.target_transform(self.label[index])  # type: ignore[index]
        else:
            log_e(self, "self.data is None.")
            assert self.data is not None

    def __len__(self) -> int:
        if self.data is not None:
            return len(self.data)
        else:
            log_e(self, "self.data is None.")
            assert self.data is not None

    def prepare(self) -> None:
        pass


class Spiral(DataSet):
    @override
    def prepare(self) -> None:
        if self.train:
            self.data, self.label = spiral.load_train_data()
        else:
            self.data, self.label = spiral.load_test_data()


class MNIST(DataSet):
    def __init__(self,
            train = True,
            transform: Callable[[np.ndarray], np.ndarray]|None = None,
            target_transform: Callable[[np.ndarray], np.ndarray]|None = None,
    ) -> None:
        super().__init__(train, transform, target_transform)

    @override
    def prepare(self) -> None:
        script_dir: str = str(Path(__file__).resolve().parent)
        mnist_dir: str = f"{script_dir}/../../mnist"

        train_files: dict[str, str] = {
            "target": "train-images-idx3-ubyte.gz",
            "label": "train-labels-idx1-ubyte.gz",
        }
        test_files: dict[str, str] = {
            "target": "t10k-images-idx3-ubyte.gz",
            "label": "t10k-labels-idx1-ubyte.gz",
        }

        files: dict[str, str]
        if self.train:
            files = train_files
        else:
            files = test_files

        data_path = f"{mnist_dir}/{files["target"]}"
        label_path = f"{mnist_dir}/{files["label"]}"
        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_data(self, filepath: str) -> np.ndarray:
        data: np.ndarray
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def _load_label(self, filepath: str) -> np.ndarray:
        labels: np.ndarray
        with gzip.open(filepath, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset = 8)
        return labels

    def show_samples(self, row: int = 10, col: int = 10) -> None:
        if self.data is None:
            log_e(self, "self.data is None.")
            return

        H: int = 28
        W: int = 28

        img: np.ndarray = np.zeros((H * row, W * col))  # whole img row x col of H x W
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] \
                        = self.data[np.random.randint(0, len(self.data) -1)].reshape(H, W)

        plt.imshow(img, cmap = "gray", interpolation = "nearest")
        plt.axis("off")
        plt.show()


class SinCurve(DataSet):
    def __init__(
            self,
            train: bool = True,
            transform: Callable[[np.ndarray], np.ndarray]|None = None,
            target_transform: Callable[[np.ndarray], np.ndarray]|None = None,
            xp: ModuleType = np,
    ) -> None:
        self.xp: ModuleType = xp

        super().__init__(train, transform, target_transform)  # calling prepare()

    @override
    def prepare(self) -> None:
        num_data: int = 1024
        dtype: DTypeLike = np.float32

        x: Array = self.xp.linspace(0, 2 * np.pi, num_data)
        noise_range: tuple[float, float] = (-0.05, 0.05)
        noise: Array = self.xp.random.uniform(noise_range[0], noise_range[1], size = x.shape)

        y: Array
        if self.train:
            y = self.xp.sin(x) + noise
        else:
            y = self.xp.cos(x)

        y = y.astype(dtype)

        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]

