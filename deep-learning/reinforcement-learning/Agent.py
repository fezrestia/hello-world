import numpy as np

class Agent:
    def __init__(self, epsilon: float, action_size: int = 10) -> None:
        self.epsilon: float = epsilon
        self.Qs: np.ndarray = np.zeros(action_size)
        self.ns: np.ndarray = np.zeros(action_size)

    def update(self, action: int, reward: float) -> None:
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return int(np.argmax(self.Qs))

