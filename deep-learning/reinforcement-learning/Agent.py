import numpy as np
from collections import defaultdict, deque

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



class TemporalDifferenceAgent:
    def __init__(self) ->None:
        self.gamma: float = 0.9
        self.alpha: float = 0.01
        self.action_size: int = 4

        random_actions: dict[int, float] = {
                0: 0.25,
                1: 0.25,
                2: 0.25,
                3: 0.25,
        }

        self.pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: random_actions)
        self.V: dict[tuple[int, int], float] = defaultdict(lambda: 0)

    def get_action(self, state: tuple[int, int]) -> int:
        action_probs: dict[int, float] = self.pi[state]
        actions: list[int] = list(action_probs.keys())
        probs: list[float] = list(action_probs.values())
        return np.random.choice(actions, p = probs)

    def eval(
        self,
        state: tuple[int, int],
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        next_V: float = 0.0 if done else self.V[next_state]
        target: float = reward + self.gamma * next_V

        self.V[state] += (target - self.V[state]) * self.alpha



class SarsaAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1
        self.action_size: int = 4

        random_actions: dict[int, float] = {
                0: 0.25,
                1: 0.25,
                2: 0.25,
                3: 0.25,
        }

        self.pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: random_actions)
        self.Q: dict[tuple[tuple[int, int], int], float] = defaultdict(lambda: 0)
        self.memory: deque[tuple[tuple[int, int], int, float, bool]] = deque(maxlen = 2)

    def get_action(self, state: tuple[int, int]) -> int:
        action_probs: dict[int, float] = self.pi[state]
        actions: list[int] = list(action_probs.keys())
        probs: list[float] = list(action_probs.values())
        return np.random.choice(actions, p = probs)

    def reset(self) -> None:
        self.memory.clear()

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        next_q: float = 0.0 if done else self.Q[next_state, next_action]

        target: float = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)



class SarsaOffPolicyAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1
        self.action_size: int = 4

        random_actions: dict[int, float] = {
                0: 0.25,
                1: 0.25,
                2: 0.25,
                3: 0.25,
        }

        self.pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: random_actions)
        self.b: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: random_actions)
        self.Q: dict[tuple[tuple[int, int], int], float] = defaultdict(lambda: 0)
        self.memory: deque[tuple[tuple[int, int], int, float, bool]] = deque(maxlen = 2)

    def get_action(self, state: tuple[int, int]) -> int:
        action_probs: dict[int, float] = self.b[state]
        actions: list[int] = list(action_probs.keys())
        probs: list[float] = list(action_probs.values())
        return np.random.choice(actions, p = probs)

    def reset(self) -> None:
        self.memory.clear()

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        next_q: float
        rho: float
        if done:
            next_q = 0.0
            rho = 1.0
        else:
            next_q = self.Q[next_state, next_action]
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        target: float = rho * (reward + self.gamma * next_q)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, 0.0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)



class QLearningAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1
        self.action_size: int = 4

        self.Q: dict[tuple[tuple[int, int], int], float] = defaultdict(lambda: 0)

    def get_action(self, state: tuple[int, int]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs: list[float] = [self.Q[state, a] for a in range(self.action_size)]
            return int(np.argmax(qs))

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        next_q_max: float
        if done:
            next_q_max = 0.0
        else:
            next_qs: list[float] = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target: float = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

