import numpy as np
from collections import defaultdict

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



class AlphaAgent:
    def __init__(self, epsilon: float, alpha: float, action_size: int = 10) -> None:
        self.epsilon: float = epsilon
        self.Qs: np.ndarray = np.zeros(action_size)
        self.alpha: float = alpha

    def update(self, action: int, reward: float) -> None:
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return int(np.argmax(self.Qs))



class RandomAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.action_size: int = 4

        random_actions: dict[int, float] = {
                0: 0.25,
                1: 0.25,
                2: 0.25,
                3: 0.25,
        }

        self.pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: random_actions)
        self.V: dict[tuple[int, int], float] = defaultdict(lambda: 0)
        self.cnts: dict[tuple[int, int], int] = defaultdict(lambda: 0)
        self.memory: list[tuple[tuple[int, int], int, float]] = []

    def get_action(self, state: tuple[int, int]) -> int:
        action_probs: dict[int, float] = self.pi[state]
        actions: list[int] = list(action_probs.keys())
        probs: list[float] = list(action_probs.values())
        return np.random.choice(actions, p = probs)

    def add(self, state: tuple[int, int], action: int, reward: float) -> None:
        data: tuple[tuple[int, int], int, float] = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def eval(self) -> None:
        G: float = 0
        for data in reversed(self.memory):  # from new to old
            (state, action, reward) = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]



def greedy_probs(
        Q: dict[tuple[tuple[int, int], int], float],
        state: tuple[int, int],
        epsilon: float = 0.0,
        action_size: int = 4,
) -> dict[int, float]:
    qs: list[float] = [Q[(state, action)] for action in range(action_size)]
    max_action: int =  int(np.argmax(qs))

    base_prob: float = epsilon / action_size
    action_probs: dict[int, float] = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1.0 - epsilon)
    return action_probs



class MonteCarloAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.epsilon: float = 0.1  # for epsilon greedy
        self.alpha: float = 0.1  # for Q update
        self.action_size: int = 4

        random_actions: dict[int, float] = {
                0: 0.25,
                1: 0.25,
                2: 0.25,
                3: 0.25,
        }

        self.pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: random_actions)
        self.Q: dict[tuple[tuple[int, int], int], float] = defaultdict(lambda: 0)
        self.memory: list[tuple[tuple[int, int], int, float]] = []

    def get_action(self, state: tuple[int, int]) -> int:
        action_probs: dict[int, float] = self.pi[state]
        actions: list[int] = list(action_probs.keys())
        probs: list[float] = list(action_probs.values())
        return np.random.choice(actions, p = probs)

    def add(self, state: tuple[int, int], action: int, reward: float) -> None:
        data: tuple[tuple[int, int], int, float] = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def update(self) -> None:
        G: float = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key: tuple[tuple[int, int], int] = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha

            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

