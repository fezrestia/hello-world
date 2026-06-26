import numpy as np

class Bandit:
    def __init__(self, arms: int = 10) -> None:
        self.rates: np.ndarray = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate: float = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class NonStatBandit:
    def __init__(self, arms: int = 10) -> None:
        self.arms: int = arms
        self.rates: np.ndarray = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate: float = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0

