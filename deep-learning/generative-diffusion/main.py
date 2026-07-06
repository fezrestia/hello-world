#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import norm
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent



## Gaus Distribution

def normal(xs: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    ys: np.ndarray = 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-(xs - mu) ** 2 / (2.0 * sigma ** 2))
    return ys

#xs: np.ndarray = np.linspace(-10.0, 10.0, 100)
#ys0 = normal(xs, mu = -3.0)
#ys1 = normal(xs, mu = 0.0)
#ys2 = normal(xs, mu = 5.0)
#
#plt.plot(xs, ys0, label = "mu = -3.0")
#plt.plot(xs, ys1, label = "mu = 0.0")
#plt.plot(xs, ys2, label = "mu = 5.0")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.legend()
#plt.show()
#
#ys0 = normal(xs, mu = 0.0, sigma = 0.5)
#ys1 = normal(xs, mu = 0.0, sigma = 1.0)
#ys2 = normal(xs, mu = 0.0, sigma = 2.0)
#
#plt.plot(xs, ys0, label = "sigma = 0.5")
#plt.plot(xs, ys1, label = "sigma = 1.0")
#plt.plot(xs, ys2, label = "sigma = 2.0")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.legend()
#plt.show()
#
#
#
## Central Limit Theorem
#N = 10
#
#x_means = []
#
#for _ in range(10000):
#    x_arr = []
#
#    for n in range(N):
#        x: float = np.random.rand()
#        x_arr.append(x)
#
#    x_mean = np.mean(x_arr)
#    x_means.append(x_mean)
#
#plt.hist(x_means, bins = "auto", density = True)
#plt.title(f"N = {N}")
#plt.xlabel("x")
#plt.ylabel("Probability Density")
#plt.xlim(-0.05, 1.05)
#plt.ylim(0, 5)
#plt.show()
#
#
#
#x_sum_arr: list[float] = []
#N = 5
#
#for _ in range(10000):
#    x_arr = []
#    for n in range(N):
#        x = np.random.rand()
#        x_arr.append(x)
#    t: float = np.sum(x_arr)
#    x_sum_arr.append(t)
#
#x_norm = np.linspace( -5, 5, 1000)
#mu = N / 2
#sigma = np.sqrt(N / 12)
#y_norm = normal(x_norm, mu, sigma)
#
#plt.hist(x_sum_arr, bins = "auto", density = True)
#plt.title(f"N = {N}")
#plt.xlim(-1, 6)
#plt.show()



## Maximum Likelihood Estimation
#
#path = os.path.join(SCRIPT_DIR, "dataset/height.txt")
#xs = np.loadtxt(path)
#print(xs.shape)
#
#mu = float(np.mean(xs))
#sigma = float(np.std(xs))
#
#print(f"mu = {mu}, sigma = {sigma}")
#
#x = np.linspace(150, 190, 1000)
#y = normal(x, mu, sigma)
#
#samples = np.random.normal(mu, sigma, 10000)
#
#plt.hist(xs, bins = "auto", density = True, alpha = 0.7, label = "original")
#plt.hist(samples, bins = "auto", density = True, alpha = 0.7, label = "generated")
#plt.plot(x, y)
#plt.xlabel("Height(cm)")
#plt.ylabel("Probability Density")
#plt.show()
#
#
#xx = 1.0
#p = norm.cdf(xx, loc = 0.0, scale = 1.0)
#print(f"xx = {xx}, p = {p}")



# Multi-Dimension Normal Distribution

# xs: input array (D,)
# mu: average array (D,)
# cov: co-variance matrix (D, D)
# return: output array D dimens (1,)
def multi_dim_normal(xs: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    det: float = np.linalg.det(cov)
    inv: np.ndarray = np.linalg.inv(cov)  # (D, D)
    D: int = len(xs)
    z: float = 1.0 / np.sqrt((2.0 * np.pi) ** D * det)
    # (xs-mu).T : (1,D)
    # inv : (D,D)
    # (xs-mu) : (D,)
    # ys : (1,D) x (D,D) x (D,) = (1,)
    ys: np.ndarray = z * np.exp((xs - mu).T @ inv @ (xs - mu) / -2.0)
    return ys

#x = np.array([[0],[0]])
#mu = np.array([[1],[2]])
#cov = np.array([[1,0], [0,1]])
#y: np.ndarray = multi_dim_normal(x, mu, cov)
#print(f"y.shape = {y.shape}")
#print(y)



X = np.array([[-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2]])
Y = np.array([[-2, -2, -2, -2, -2],
              [-1, -1, -1, -1, -1],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [2, 2, 2, 2, 2]])
Z = X ** 2 + Y ** 2

ax: Any = plt.axes(projection = "3d")
ax.plot_surface(X, Y, Z, cmap = "viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

