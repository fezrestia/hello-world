#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import norm
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent

np.random.seed(0)



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



#X = np.array([[-2, -1, 0, 1, 2],
#              [-2, -1, 0, 1, 2],
#              [-2, -1, 0, 1, 2],
#              [-2, -1, 0, 1, 2],
#              [-2, -1, 0, 1, 2]])
#Y = np.array([[-2, -2, -2, -2, -2],
#              [-1, -1, -1, -1, -1],
#              [0, 0, 0, 0, 0],
#              [1, 1, 1, 1, 1],
#              [2, 2, 2, 2, 2]])
#Z = X ** 2 + Y ** 2
#
#ax: Any = plt.axes(projection = "3d")
#ax.plot_surface(X, Y, Z, cmap = "viridis")
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")
#plt.show()
#
#
#
#xs = np.arange(-2, 2, 0.1)
#ys = np.arange(-2, 2, 0.1)
#X, Y = np.meshgrid(xs, ys)
#Z = X ** 2 + Y ** 2
#
#ax2: Any = plt.axes(projection = "3d")
#ax2.plot_surface(X, Y, Z, cmap = "viridis")
#ax2.set_xlabel("x")
#ax2.set_ylabel("y")
#ax2.set_zlabel("z")
#plt.show()
#
#ax3 = plt.axes()
#ax3.contour(X, Y, Z)
#ax3.set_xlabel("x")
#ax3.set_ylabel("y")
#plt.show()
#
#
#
#mu = np.array([0.5, -0.2])
#cov = np.array([[2.0, 0.3],
#                [0.3, 0.5]])
#
#xs = ys = np.arange(-5, 5, 0.1)
#X, Y = np.meshgrid(xs, ys)
#Z = np.zeros_like(X)
#
#for i in range(X.shape[0]):
#    for j in range(X.shape[1]):
#        x = np.array([X[i, j], Y[i, j]])
#        Z[i, j] = multi_dim_normal(x, mu, cov)
#
#fig = plt.figure()
#f1: Any = fig.add_subplot(1, 2, 1, projection = "3d")
#f1.set_xlabel("x")
#f1.set_ylabel("y")
#f1.set_zlabel("z")
#f1.plot_surface(X, Y, Z, cmap = "viridis")
#f2 = fig.add_subplot(1, 2, 2)
#f2.set_xlabel("x")
#f2.set_ylabel("y")
#f2.contour(X, Y, Z)
#plt.show()
#
#
#
#N = 10000
#D = 2
#xs = np.random.rand(N, D)
#
#mu = np.sum(xs, axis = 0)
#mu /= N
#
#cov = np.zeros((D, D))
#
#for n in range(N):
#    x = xs[n]
#    z = x - mu
#    z = z[:, np.newaxis]
#    cov += z @ z.T
#
#cov /= N
#
#print(f"mu = {mu}, cov = {cov}")
#
#
#
#path = os.path.join(SCRIPT_DIR, "dataset/height_weight.txt")
#xs = np.loadtxt(path)
#print(f"xs.shape = {xs.shape}")
#
#small_xs = xs[:500]
#plt.scatter(small_xs[:, 0], small_xs[:, 1])
#plt.xlabel("Height[cm]")
#plt.ylabel("Weight[kg]")
#plt.show()
#
#mu = np.mean(xs, axis = 0)
#cov = np.cov(xs, rowvar = False)
#
#X, Y = np.meshgrid(np.arange(150, 195, 0.5),
#                   np.arange(45, 75, 0.5))
#Z = np.zeros_like(X)
#
#for i in range(X.shape[0]):
#    for j in range(X.shape[1]):
#        x = np.array([X[i, j], Y[i, j]])
#        Z[i, j] = multi_dim_normal(x, mu, cov)
#
#fig = plt.figure()
#f3: Any = fig.add_subplot(1, 2, 1, projection = "3d")
#f3.set_xlabel("x")
#f3.set_ylabel("y")
#f3.set_zlabel("z")
#f3.plot_surface(X, Y, Z, cmap = "viridis")
#f4 = fig.add_subplot(1, 2, 2)
#f4.scatter(small_xs[:, 0], small_xs[:, 1])
#f4.set_xlabel("x")
#f4.set_ylabel("y")
#f4.contour(X, Y, Z)
#plt.show()



# Gausian Mixture Model

#path = os.path.join(SCRIPT_DIR, "dataset/old_faithful.txt")
#xs = np.loadtxt(path)
#print(f"xs.shape = {xs.shape}")
#print(f"xs[0] = {xs[0]}")
#
#plt.scatter(xs[:, 0], xs[:, 1])
#plt.xlabel("eruptions[min]")
#plt.ylabel("waiting[min]")
#plt.show()
#
#
#mus = np.array([[2.0, 54.50],
#                [4.3, 80.0]])
#covs = np.array([[[0.07, 0.44],
#                  [0.44, 33.7]],
#                 [[0.17, 0.94],
#                  [0.94, 36.0]]])
#phis = np.array([0.35, 0.65])
#
#def sample() -> np.ndarray:
#    z = np.random.choice(2, p = phis)
#    mu, cov = mus[z], covs[z]
#    x = np.random.multivariate_normal(mu, cov)
#    return x
#
#N = 500
#xs = np.zeros((N, 2))
#for i in range(N):
#    xs[i] = sample()
#
#plt.scatter(xs[:, 0], xs[:, 1], color = "orange", alpha = 0.7)
#plt.xlabel("x")
#plt.ylabel("y")
#plt.show()



# D: input dim
# P: phi size (count of normal distribution)
#
# xs: (D,)
# phis: (P,)
# mus: (D,)
# covs: (D, D)
# return: scalar
def gmm(
        xs: np.ndarray,
        phis: np.ndarray,
        mus: np.ndarray,
        covs: np.ndarray,
) -> np.ndarray:
    K: int = len(phis)

    # y: scalar
    y: np.ndarray = np.array(0.0)

    for k in range(K):
        # phi: scalar
        # mu: (D,)
        # cov: (D, D)
        phi: np.ndarray = phis[k]
        mu: np.ndarray = mus[k]
        cov: np.ndarray = covs[k]

        y += phi * multi_dim_normal(xs, mu, cov)

    return y



#mus = np.array([[2.0, 54.50],
#                [4.3, 80.0]])
#
#covs = np.array([[[0.07, 0.44],
#                  [0.44, 33.7]],
#                 [[0.17, 0.94],
#                  [0.94, 36.00]]])
#
#phis = np.array([0.35, 0.65])
#
#xs = np.arange(1, 6, 0.1)
#ys = np.arange(40, 100, 0.1)
#X, Y = np.meshgrid(xs, ys)
#Z = np.zeros_like(X)
#
#for i in range(X.shape[0]):
#    for j in range(X.shape[1]):
#        x = np.array([X[i, j], Y[i, j]])
#        Z[i, j] = gmm(x, phis, mus, covs)
#
#fig = plt.figure()
#ax1: Any = fig.add_subplot(1, 2, 1, projection = "3d")
#ax1.set_xlabel("x")
#ax1.set_ylabel("y")
#ax1.set_zlabel("z")
#ax1.plot_surface(X, Y, Z, cmap = "viridis")
#ax2 = fig.add_subplot(1, 2, 2)
#ax2.set_xlabel("x")
#ax2.set_ylabel("y")
#ax2.contour(X, Y, Z)
#plt.show()



# Estimation-Maximization Algorithm

path = os.path.join(SCRIPT_DIR, "dataset/old_faithful.txt")
xs = np.loadtxt(path)
print(f"old_faithful : xs.shape = {xs.shape}")

phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0],
                [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)])  # identity matrix

K = len(phis)  # 2
N = len(xs)  # 272
MAX_ITERS = 100
THRESHOLD = 1e-4



# xs : (D,)
# phis : (N,)
# mus: (D,)
# covs: (D, D)
# return : scalar
def likelihood(
        xs: np.ndarray,
        phis: np.ndarray,
        mus: np.ndarray,
        covs: np.ndarray,
) -> np.ndarray:
    eps: float = 1e-8
    L: np.ndarray = np.array(0.0)
    N: int = len(xs)

    for x in xs:
        y: np.ndarray = gmm(x, phis, mus, covs)
        L += np.log(y + eps)

    return L / N



current_likelihood = likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS):
    # E step
    qs = np.zeros((N, K))  # N: data dimens, K: phi len
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi = phis[k]
            mu = mus[k]
            cov = covs[k]

            qs[n, k] = phi * multi_dim_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)

    # M step
    qs_sum = qs.sum(axis = 0)
    for k in range(K):
        # phi
        phis[k] = qs_sum[k] / N

        # mu
        c = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]

        # cov
        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:, np.newaxis]
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]

    # converged or not
    print(f"current_likelihood = {current_likelihood:.3f}")

    next_likelihood = likelihood(xs, phis, mus, covs)

    diff = np.abs(next_likelihood - current_likelihood)
    if diff < THRESHOLD:
        break

    current_likelihood = next_likelihood


N = 500
new_xs = np.zeros((N, 2))
for n in range(N):
    k = np.random.choice(2, p = phis)
    mu = mus[k]
    cov = covs[k]
    new_xs[n] = np.random.multivariate_normal(mu, cov)


# visualize
def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multi_dim_normal(x, mu, cov)
    plt.contour(X, Y, Z)

plt.scatter(xs[:,0], xs[:,1], alpha = 0.7, label ="original")
plt.scatter(new_xs[:,0], new_xs[:,1], alpha = 0.7, label = "generated")
plot_contour(phis, mus, covs)
plt.xlabel("eruptions[min])")
plt.ylabel("waiting[min]")
plt.show()

