#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import override, Any

if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



import importlib
import sys
#importlib.reload(sys.modules["GridWorld"])
#importlib.reload(sys.modules["Agent"])


from Bandit import Bandit, NonStatBandit
from Agent import Agent, AlphaAgent, RandomAgent, MonteCarloAgent, TemporalDifferenceAgent, SarsaAgent, SarsaOffPolicyAgent, QLearningAgent, DNNQLearningAgent, DQNAgent
from GridWorld import GridWorld



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

## state vs. action poicy
#pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: {
#        0: 0.25,
#        1: 0.25,
#        2: 0.25,
#        3: 0.25,
#})
#state: tuple[int, int] = (0, 1)
#print(pi[state])

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

#env = GridWorld()
#gamma = 0.9
#pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
#V: dict[tuple[int, int], float] = defaultdict(lambda: 0.0)
#V = policy_eval(pi, V, env, gamma)
#env.render_v(V, pi)

def argmax(d: dict[int, float]) -> int:
    max_val: float = max(d.values())
    max_key: int = 0
    for key, val in d.items():
        if val == max_val:
            max_key = key
    return max_key

#action_values = {0: 0.1, 1: -0.3, 2: 9.9, 3: -1.3}
#max_act = argmax(action_values)
#print(max_act)

def greedy_policy(
        V: dict[tuple[int, int], float],
        env: GridWorld,
        gamma: float,
) -> dict[tuple[int, int], dict[int, float]]:
    pi: dict[tuple[int, int], dict[int, float]] = {}

    for state in env.states():
        action_values: dict[int, float] = {}

        for action in env.actions():
            next_state: tuple[int, int] = env.next_state(state, action)

            r: float|None = env.reward(state, action, next_state)

            if r is None:
                raise Exception("Unexpected: r is None")

            value: float = r + gamma * V[next_state]
            action_values[action] = value

        max_action: int = argmax(action_values)

        action_probs: dict[int, float] = {
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.0,
        }
        action_probs[max_action] = 1.0

        pi[state] = action_probs

    return pi

def policy_iter(
        env: GridWorld,
        gamma: float,
        threshold: float = 0.001,
        is_render: bool = False,
) -> dict[tuple[int, int], dict[int, float]]:
    pi: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: {
            0: 0.25,
            1: 0.25,
            2: 0.25,
            3: 0.25,
    })

    V: dict[tuple[int, int], float] = defaultdict(lambda: 0.0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        new_pi: dict[tuple[int, int], dict[int, float]] = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi:
            break

        pi = new_pi

    return pi

#env = GridWorld()
#gamma = 0.9
#pi = policy_iter(env, gamma, 0.0001, True)

def value_iter_onestep(
        V: dict[tuple[int, int], float],
        env: GridWorld,
        gamma: float = 0.9,
) -> dict[tuple[int, int], float]:
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0.0
            continue

        action_values: list[float] = []

        for action in env.actions():
            next_state: tuple[int, int] = env.next_state(state, action)
            r: float|None = env.reward(state, action, next_state)

            if r is None:
                raise Exception("Unexpected: r is None")

            value: float = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)

    return V

def value_iter(
        V: dict[tuple[int, int], float],
        env: GridWorld,
        gamma: float = 0.9,
        threshold: float = 0.001,
        is_render = True,
) -> dict[tuple[int, int], float]:
    while True:
        if is_render:
            env.render_v(V)

        old_V: dict[tuple[int, int], float] = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta: float = 0.0
        for state in V.keys():
            d: float = abs(V[state] - old_V[state])
            if delta < d:
                delta = d

        if delta < threshold:
            break

    return V

#V: dict[tuple[int, int], float] = defaultdict(lambda: 0.0)
#env = GridWorld()
#gamma = 0.9
#V = value_iter(V, env, gamma)
#pi = greedy_policy(V, env, gamma)
#env.render_v(V, pi)

#env = GridWorld()
#action = 0
#(next_state, reward, done) = env.step(action)
#print(f"next_state = {next_state}, reward = {reward}, done = {done}")

#env = GridWorld()
#agent = MonteCarloAgent()
#episodes = 10000
#for episode in range(episodes):
#    state = env.reset()
#    agent.reset()
#    while True:
#        action = agent.get_action(state)
#        next_state, reward, done = env.step(action)
#        agent.add(state, action, reward)
#        if done:
#            agent.update()
#            break
#        state = next_state
#env.render_q(agent.Q)

#x = np.array([1,2,3])
#pi = np.array([0.1,0.1,0.8])
#e = np.sum(x * pi)
#print(f"E = {e}")
#n = 100
#samples = []
#for _ in range(n):
#    s = np.random.choice(x, p = pi)
#    samples.append(s)
#mean = np.mean(samples)
#var = np.var(samples)
#print(f"mean = {mean}, var = {var}")
#b = np.array([0.2, 0.2, 0.6])
#n = 100
#samples = []
#for _ in range(n):
#    idx = np.arange(len(b))
#    i = np.random.choice(idx, p = b)
#    s = x[i]
#    rho = pi[i] / b[i]
#    samples.append(rho * s)
#mean = np.mean(samples)
#var = np.var(samples)
#print(f"mean = {mean}, var = {var}")

#env = GridWorld()
#agent = TemporalDifferenceAgent()
#episodes = 1000
#for episode in range(episodes):
#    state = env.reset()
#    while True:
#        action = agent.get_action(state)
#        next_state, reward, done = env.step(action)
#        agent.eval(state, reward, next_state, done)
#        if done:
#            break
#        state = next_state
#env.render_v(agent.V)

#env = GridWorld()
#agent = SarsaOffPolicyAgent()
#episodes = 10000
#for episode in range(episodes):
#    state = env.reset()
#    agent.reset()
#    while True:
#        action = agent.get_action(state)
#        next_state, reward, done = env.step(action)
#        agent.update(state, action, reward, done)
#        if done:
#            agent.update(next_state, 0, 0.0, False)
#            break
#        state = next_state
#env.render_q(agent.Q)

#env = GridWorld()
#agent = QLearningAgent()
#episodes = 10000
#for episode in range(episodes):
#    state = env.reset()
#    while True:
#        action = agent.get_action(state)
#        next_state, reward, done = env.step(action)
#        agent.update(state, action, reward, next_state, done)
#        if done:
#            break
#        state = next_state
#env.render_q(agent.Q)



# newral network included

from deepzero import Variable
from deepzero import Parameter
from deepzero import Function
from deepzero import use_config, no_grad, test_mode
from deepzero import Visualize
from deepzero.Function import add, mul, sub, rsub, div, rdiv, neg, pow, sin, cos, tanh, sum, reshape, transpose, matmul, linear, sigmoid, mean_squared_error, as_variable, get_item, softmax, softmax_cross_entropy, accuracy, relu, dropout, img2col, col2img, conv2d, deconv2d, pooling, as_array, log
from deepzero import Layer
from deepzero.Layer import Linear, RNN, LSTM
from deepzero import Model
from deepzero.Model import MultiLayerPerceptron, VGG16, SimpleRNN, LSTMRNN
from deepzero import Optimizer
from deepzero.Optimizer import StochasticGradientDecent, MomentumSGD, Adam
from deepzero import DataSet
from deepzero.DataSet import Spiral, MNIST, SinCurve
from deepzero import DataLoader, SeqDataLoader
from deepzero.cuda import gpu_enabled, use_cp, use_np, as_np, as_cp
from deepzero import get_conv_outsize
from deepzero.utils import get_file
from deepzero.Type import Array

from Model import QNet



def one_hot(state: tuple[int, int]) -> np.ndarray:
    HEIGHT: int = 3
    WIDTH: int = 4

    vec: np.ndarray = np.zeros(HEIGHT * WIDTH, dtype = np.float32)

    y, x = state

    idx: int = WIDTH * y + x
    vec[idx] = 1.0

    return vec[np.newaxis, :]  # for batch axis



#state = (2, 0)
#x = one_hot(state)
#print(x.shape)
#print(x)

#qnet = QNet()
#state = (2, 0)
#state_array = one_hot(state)
#(qs,) = qnet(as_variable(state_array))
#print(qs.shape)
#print(qs)



# DNN Q-Learning
#
#env = GridWorld()
#agent = DNNQLearningAgent()
#episodes = 1000
#loss_history = []
#for episode in range(episodes):
#    state = env.reset()
#    state_array = one_hot(state)
#    total_loss = 0.0
#    cnt = 0
#    done = False
#    while not done:
#        action = agent.get_action(state_array)
#        next_state, reward, done = env.step(action)
#        next_state_array = one_hot(next_state)
#        loss = agent.update(state_array, action, reward, next_state_array, done)
#        total_loss += loss
#        cnt += 1
#        state_array = next_state_array
#    average_loss = total_loss / cnt
#    loss_history.append(average_loss)
#    if episode % 100 == 0:
#        print(f"average_loss = {average_loss}")
#plt.xlabel('episode')
#plt.ylabel('loss')
#plt.plot(range(len(loss_history)), loss_history)
#plt.show()
#Q = {}
#for state in env.states():
#    for action in env.action_space:
#        q = agent.qnet(one_hot(state))[0][:, action]
#        Q[state, action] = float(q.data[0])
#env.render_q(Q)



import gymnasium as gym

#env = gym.make("CartPole-v1")
#state = env.reset()
#print(state)
#action_space = env.action_space
#print(action_space)
#action = 0
#next_state, reward, terminated, truncated, info = env.step(action)
#print(next_state)



# DQN
#
#episodes = 300
#sync_interval = 20
#env = gym.make("CartPole-v1")
#agent = DQNAgent()
#reward_history = []
#for episode in range(episodes):
#    state, info = env.reset()
#    done = False
#    total_reward = 0.0
#    while not done:
#        action = agent.get_action(state)
#        next_state, reward, terminated, truncated, info = env.step(action)
#        if terminated or truncated:
#            done = True
#        agent.update(state, action, float(reward), next_state, done)
#        state = next_state
#        total_reward += float(reward)
#    if episode % sync_interval == 0:
#        agent.sync_qnet()
#    reward_history.append(total_reward)
#    if episode % 10 == 0:
#        print(f"total_reward = {total_reward}")
#plt.xlabel('episode')
#plt.ylabel('total_reward')
#plt.plot(range(len(reward_history)), reward_history)
#plt.show()



class Policy(Model):
    def __init__(self, action_size: int) -> None:
        super().__init__()

        self.l1: Linear = Linear(128)
        self.l2: Linear = Linear(action_size)

    @override
    def forward(self, *inputs: Variable) -> tuple[Variable, ...]:
        x: Variable = inputs[0]

        (x,) = self.l1(x)
        x = relu(x)
        (x,) = self.l2(x)
        x = softmax(x)

        return (x,)

class PolicyBasedAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.98
        self.learning_rate: float = 0.0002
        self.action_size: int = 2

        self.memory: list[tuple[float, Variable]] = []

        self.pi: Policy = Policy(self.action_size)
        self.optimizer = Adam(self.learning_rate)
        self.optimizer.setup(self.pi)

    def get_action(self, state: Array) -> tuple[int, Array]:
        state = state[np.newaxis, :]  # for batch axis

        probs: Variable
        (probs,) = self.pi(as_variable(state))

        one_prob: Variable = probs[0]

        action: int = np.random.choice(len(one_prob), p = one_prob.data)

        return action, one_prob[action]

    def add(self, reward: float, prob: Variable) -> None:
        data: tuple[float, Variable] = (reward, prob)
        self.memory.append(data)

    def update(self) -> None:
        self.pi.clear_grads()

        G: float = 0.0
        loss: Variable = Variable(as_array(0.0))


        # REINFORCE, use G immediately. total G icludes total timeline.
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -log(prob) * G

        #for reward, prob in reversed(self.memory):
        #    G = reward + self.gamma * G
        #for reward, prob in self.memory:
        #    loss += -log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []



#env = gym.make("CartPole-v1")
#state, info = env.reset()
#agent = PolicyBasedAgent()
#print(f"state = {state}")
#action, prob = agent.get_action(state)
#print(f"action = {action}")
#print(f"prob = {prob}")
#G = 100.0
#J = G * log(as_variable(prob))
#print(f"J = {J}")
#J.backward(keep_grad = True)
#print(f"J.grad = {J.grad}")



#episodes = 3000
#env = gym.make("CartPole-v1")
#agent = PolicyBasedAgent()
#reward_history = []
#for episode in range(episodes):
#    state, info = env.reset()
#    done = False
#    total_reward = 0.0
#    while not done:
#        action, prob = agent.get_action(state)
#        next_state, reward, terminated, truncated, info = env.step(action)
#        if terminated or truncated:
#            done = True
#        agent.add(float(reward), as_variable(prob))
#        state = next_state
#        total_reward += float(reward)
#    agent.update()
#    reward_history.append(total_reward)
#    if episode % 100 == 0:
#        print(f"total_reward = {total_reward}")
#plt.xlabel('episode')
#plt.ylabel('total_reward')
#plt.plot(range(len(reward_history)), reward_history)
#plt.show()



class PolicyNet(Model):
    def __init__(self, action_size: int = 2) -> None:
        super().__init__()
        self.l1: Linear = Linear(128)
        self.l2: Linear = Linear(action_size)

    @override
    def forward(self, *inputs: Variable) -> tuple[Variable, ...]:
        x: Variable = inputs[0]

        (x,) = self.l1(x)
        x = relu(x)
        (x,) = self.l2(x)
        x = softmax(x)

        return (x,)

class ValueNet(Model):
    def __init__(self) -> None:
        super().__init__()
        self.l1: Linear = Linear(128)
        self.l2: Linear = Linear(1)

    @override
    def forward(self, *inputs: Variable) -> tuple[Variable, ...]:
        x: Variable = inputs[0]

        (x,) = self.l1(x)
        x = relu(x)
        (x,) = self.l2(x)

        return (x,)

class ActorCriticAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.98
        self.learning_rate_pi: float = 0.0002
        self.learning_rate_v: float = 0.0005
        self.action_size: int = 2

        self.pi: PolicyNet = PolicyNet(self.action_size)
        self.v: ValueNet = ValueNet()

        self.optimizer_pi = Adam(self.learning_rate_pi).setup(self.pi)
        self.optimizer_v = Adam(self.learning_rate_v).setup(self.v)

    def get_action(self, state: Array) -> tuple[int, Array]:
        state = state[np.newaxis, :]  # for batch axis

        probs: Variable
        (probs,) = self.pi(as_variable(state))

        one_prob: Variable = probs[0]

        action: int = np.random.choice(len(one_prob), p = one_prob.data)

        return action, one_prob[action]

    def update(
            self,
            state: Array,
            action_prob: Array,
            reward: float,
            next_state: Array,
            done: bool,
    ) -> None:
        # add batch axis
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]


        # for Value
        target: Variable = reward + self.gamma * self.v(as_variable(next_state))[0] * (1.0 - done)
        target.unchain()
        v: Variable = self.v(as_variable(state))[0]
        loss_v: Variable = mean_squared_error(v, target)


        # for Policy
        delta: Variable = target - v
        delta.unchain()
        loss_pi: Variable = -log(as_variable(action_prob)) * delta


        self.v.clear_grads()
        self.pi.clear_grads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()



episodes = 3000
env = gym.make("CartPole-v1")
agent = ActorCriticAgent()
reward_history = []
for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
        agent.update(state, prob, float(reward), next_state, done)
        state = next_state
        total_reward += float(reward)
    reward_history.append(total_reward)
    if episode % 100 == 0:
        print(f"total_reward = {total_reward}")
plt.xlabel('episode')
plt.ylabel('total_reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()


