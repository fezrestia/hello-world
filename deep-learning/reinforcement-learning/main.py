#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Bandit import Bandit, NonStatBandit
from Agent import Agent, AlphaAgent
from GridWorld import GridWorld



import importlib
import sys
#importlib.reload(sys.modules["GridWorld"])



#runs: int = 200
#steps: int = 1000
#epsilon: float = 0.1
#all_rates: np.ndarray = np.zeros((runs, steps))
#for run in range(runs):
#    #bandit: Bandit = Bandit()
#    #agent: Agent = Agent(epsilon)
#    bandit: NonStatBandit = NonStatBandit()
#    agent: AlphaAgent = AlphaAgent(epsilon, 0.8)
#    total_reward: float = 0.0
#    total_rewards: list[float] = []
#    rates: list[float] = []
#    for step in range(steps):
#        action: int = agent.get_action()
#        reward: float = bandit.play(action)
#        agent.update(action, reward)
#        total_reward += reward
#        total_rewards.append(total_reward)
#        rates.append(total_reward / (step + 1))
#    all_rates[run] = rates
#avg_rates: np.ndarray = np.average(all_rates, axis = 0)  # average for each step
#plt.ylabel("rates")
#plt.xlabel("steps")
#plt.plot(avg_rates)
#plt.show()

#r: float = 0.9
#V = {
#    "L1": 0.0,
#    "L2": 0.0,
#}
#cnt: int = 0
#while True:
#    tmp_l1 = 0.5 * (-1 + r * V["L1"]) + 0.5 * (1 + r * V["L2"])
#    delta_l1: float = abs(tmp_l1 - V["L1"])
#    V["L1"] = tmp_l1
#    tmp_l2 = 0.5 * (0 + r * V["L1"]) + 0.5 * (-1 + r * V["L2"])
#    delta_l2: float = abs(tmp_l2 - V["L2"])
#    V["L2"] = tmp_l2
#    delta = max(delta_l1, delta_l2)
#    cnt += 1
#    if delta < 0.0001:
#        print(V)
#        print(cnt)
#        break

#env = GridWorld()
#print(env.height)
#print(env.width)
#print(env.shape)
#for action in env.actions():
#    print(action)
#print("------")
#for state in env.states():
#    print(state)

#env = GridWorld()
#V = {}
#for state in env.states():
#    V[state] = np.random.randn()
#env.render_v(V)

# state vs. action poicy
pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 0.25,
})
state: tuple[int, int] = (0, 1)
print(pi[state])

def eval_onestep(
        pi: dict[tuple[int, int], dict[int, float]],
        V: dict[tuple[int, int], float],
        env: GridWorld,
        gamma: float = 0.9,
) -> dict[tuple[int, int], float]:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0.0
            continue

        action_probs: dict[int, float] = pi[state]
        new_V: float = 0.0

        for action, action_prob in action_probs.items():
            next_state: tuple[int, int] = env.next_state(state, action)
            r: float|None = env.reward(state, action, next_state)

            if r is None:
                raise Exception("Unexpected: r is None")

            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V

    return V

def policy_eval(
        pi: dict[tuple[int, int], dict[int, float]],
        V: dict[tuple[int, int], float],
        env: GridWorld,
        gamma: float,
        threshold: float = 0.001,
) -> dict[tuple[int, int], float]:
    while True:
        old_V: dict[tuple[int, int], float] = V.copy()

        V = eval_onestep(pi, V, env, gamma)

        delta: float = 0.0
        for state in V.keys():
            dif: float = abs(V[state] - old_V[state])
            if delta < dif:
                delta = dif

        if delta < threshold:
            break

    return V

env = GridWorld()
gamma = 0.9
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
V: dict[tuple[int, int], float] = defaultdict(lambda: 0.0)
V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)

