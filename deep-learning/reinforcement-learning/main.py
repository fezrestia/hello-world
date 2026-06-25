#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Bandit import Bandit
from Agent import Agent



runs: int = 200
steps: int = 1000
epsilon: float = 0.1
all_rates: np.ndarray = np.zeros((runs, steps))
for run in range(runs):
    bandit: Bandit = Bandit()
    agent: Agent = Agent(epsilon)
    total_reward: float = 0.0
    total_rewards: list[float] = []
    rates: list[float] = []
    for step in range(steps):
        action: int = agent.get_action()
        reward: float = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))
    all_rates[run] = rates
avg_rates: np.ndarray = np.average(all_rates, axis = 0)  # average for each step
plt.ylabel("rates")
plt.xlabel("steps")
plt.plot(avg_rates)
plt.show()

