#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import norm  # type: ignore[import-untyped]
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torchvision  # type: ignore[import-untyped]
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

SCRIPT_DIR = Path(__file__).resolve().parent

np.random.seed(0)
torch.manual_seed(0)



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

#path = os.path.join(SCRIPT_DIR, "dataset/old_faithful.txt")
#xs = np.loadtxt(path)
#print(f"old_faithful : xs.shape = {xs.shape}")
#
#phis = np.array([0.5, 0.5])
#mus = np.array([[0.0, 50.0],
#                [0.0, 100.0]])
#covs = np.array([np.eye(2), np.eye(2)])  # identity matrix
#
#K = len(phis)  # 2
#N = len(xs)  # 272
#MAX_ITERS = 100
#THRESHOLD = 1e-4



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



#current_likelihood = likelihood(xs, phis, mus, covs)
#
#for iter in range(MAX_ITERS):
#    # E step
#    qs = np.zeros((N, K))  # N: data dimens, K: phi len
#    for n in range(N):
#        x = xs[n]
#        for k in range(K):
#            phi = phis[k]
#            mu = mus[k]
#            cov = covs[k]
#
#            qs[n, k] = phi * multi_dim_normal(x, mu, cov)
#        qs[n] /= gmm(x, phis, mus, covs)
#
#    # M step
#    qs_sum = qs.sum(axis = 0)
#    for k in range(K):
#        # phi
#        phis[k] = qs_sum[k] / N
#
#        # mu
#        c = 0
#        for n in range(N):
#            c += qs[n, k] * xs[n]
#        mus[k] = c / qs_sum[k]
#
#        # cov
#        c = 0
#        for n in range(N):
#            z = xs[n] - mus[k]
#            z = z[:, np.newaxis]
#            c += qs[n, k] * z @ z.T
#        covs[k] = c / qs_sum[k]
#
#    # converged or not
#    print(f"current_likelihood = {current_likelihood:.3f}")
#
#    next_likelihood = likelihood(xs, phis, mus, covs)
#
#    diff = np.abs(next_likelihood - current_likelihood)
#    if diff < THRESHOLD:
#        break
#
#    current_likelihood = next_likelihood
#
#
#N = 500
#new_xs = np.zeros((N, 2))
#for n in range(N):
#    k = np.random.choice(2, p = phis)
#    mu = mus[k]
#    cov = covs[k]
#    new_xs[n] = np.random.multivariate_normal(mu, cov)
#
#
## visualize
#def plot_contour(w, mus, covs):
#    x = np.arange(1, 6, 0.1)
#    y = np.arange(40, 100, 1)
#    X, Y = np.meshgrid(x, y)
#    Z = np.zeros_like(X)
#
#    for i in range(X.shape[0]):
#        for j in range(X.shape[1]):
#            x = np.array([X[i, j], Y[i, j]])
#
#            for k in range(len(mus)):
#                mu, cov = mus[k], covs[k]
#                Z[i, j] += w[k] * multi_dim_normal(x, mu, cov)
#    plt.contour(X, Y, Z)
#
#plt.scatter(xs[:,0], xs[:,1], alpha = 0.7, label ="original")
#plt.scatter(new_xs[:,0], new_xs[:,1], alpha = 0.7, label = "generated")
#plot_contour(phis, mus, covs)
#plt.xlabel("eruptions[min])")
#plt.ylabel("waiting[min]")
#plt.show()



# torch

#def rosenbrock(x0, x1):
#    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
#    return y
#
#x0 = torch.tensor(0.0, requires_grad = True)
#x1 = torch.tensor(2.0, requires_grad = True)
#
#lr = 0.001
#iters = 10000
#
#for i in range(iters):
#    if i % 1000 == 0:
#        print(x0.item(), x1.item())
#
#    y = rosenbrock(x0, x1)
#
#    y.backward()
#
#    if x0.grad is not None and x1.grad is not None:
#        x0.data -= lr * x0.grad.data
#        x1.data -= lr * x1.grad.data
#
#        x0.grad.zero_()
#        x1.grad.zero_()
#    else:
#        raise Exception("x0.grad or x1.grad is None")
#
#print(x0.item(), x1.item())
#
#
#x = torch.rand(100, 1)
#y = 2 * x + 5 + torch.rand(100, 1)
#
#W = torch.zeros((1, 1), requires_grad = True)
#b = torch.zeros(1, requires_grad = True)
#
#def predict(x):
#    y = x @ W + b
#    return y



def mean_squared_error(x0, x1):
    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N



#lr = 0.1
#iters = 100
#for i in range(iters):
#    y_hat = predict(x)
#    loss = mean_squared_error(y, y_hat)
#
#    loss.backward()
#
#    if W.grad is None or b.grad is None:
#        raise Exception("W or b is None.")
#
#    W.data -= lr * W.grad.data
#    b.data -= lr * b.grad.data
#
#    W.grad.zero_()
#    b.grad.zero_()
#
#    if i % 10 == 0:
#        print(f"loss = {loss.item()}")
#
#print(f"loss = {loss.item()}")
#print(f"W = {W}")
#print(f"b = {b}")


#W = nn.Parameter(torch.zeros(1, 1))
#b = nn.Parameter(torch.zeros(1))
#
#print(W)
#print(b)


#class Model(nn.Module):
#    def __init__(self):
#        super().__init__()
#
#        self.W = nn.Parameter(torch.zeros(1, 1))
#        self.b = nn.Parameter(torch.zeros(1))
#
#    def forward(self, x):
#        y = x @ self.W + self.b
#        return y
#
#
#model = Model()
#
#for param in model.parameters():
#    print(param)


#class Model(nn.Module):
#    def __init__(self, input_size = 1, output_size = 1):
#        super().__init__()
#
#        self.linear = nn.Linear(input_size, output_size)
#
#    def forward(self, x):
#        y = self.linear(x)
#        return y
#
#model = Model()
#for param in model.parameters():
#    print(param)
#
#
#x = torch.rand(100, 1)
#y = 2 * x + 5 + torch.rand(100, 1)
#lr = 0.1
#iters = 100
#
#
##optimizer = torch.optim.SGD(model.parameters(), lr = lr)
#optimizer = torch.optim.Adam(model.parameters(), lr = lr)
#
#for i in range(iters):
#    y_hat = model(x)
#
#    loss = nn.functional.mse_loss(y, y_hat)
#
#    loss.backward()
#
#    optimizer.step()
#    optimizer.zero_grad()


#x = torch.rand(100, 1)
#y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)
#
#class Model(nn.Module):
#    def __init__(self, input_size = 1, hidden_size = 10, output_size = 1):
#        super().__init__()
#
#        self.linear1 = nn.Linear(input_size, hidden_size)
#        self.linear2 = nn.Linear(hidden_size, output_size)
#
#    def forward(self, x):
#        y = self.linear1(x)
#        y = F.sigmoid(y)
#        y = self.linear2(y)
#        return y
#
#lr = 0.2
#iters = 10000
#
#model = Model()
#optimizer = torch.optim.SGD(model.parameters(), lr = lr)
#
#for i in range(iters):
#    y_pred = model(x)
#    loss = F.mse_loss(y, y_pred)
#
#    loss.backward()
#
#    optimizer.step()
#    optimizer.zero_grad()
#
#    if i % 1000 == 0:
#        print(loss.item())
#
#print(loss.item())
#
## plot
#plt.scatter(x.detach().numpy(), y.detach().numpy(), s=10)
#x = torch.linspace(0, 1, 100).reshape(-1, 1)
#y = model(x).detach().numpy()
#plt.plot(x, y, color = "red")
#plt.show()
#
#
#dataset = torchvision.datasets.MNIST(
#    root = ".tmp",
#    train = True,
#    transform = None,
#    download = True,
#)
#
#x, label = dataset[0]
#print(f"size = {len(dataset)}")
#print(f"type = {type(x)}")
#print(f"label = {label}")
#
#plt.imshow(x, cmap = "gray")
#plt.show()
#
#
#transform = transforms.ToTensor()
#
#dataset = torchvision.datasets.MNIST(
#    root = ".tmp",
#    train = True,
#    transform = transform,
#    download = True,
#)
#
#x, label = dataset[0]
#print(f"size = {len(dataset)}")
#print(f"type = {type(x)}")
#print(f"label = {label}")
#
#
#dataloader = torch.utils.data.DataLoader(
#    dataset,
#    batch_size = 32,
#    shuffle = True,
#)
#
#for x, label in dataloader:
#    print(f"x.shape = {x.shape}")
#    print(f"label.shape = {label.shape}")
#    break



# Variational Auto Encoder

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h: Tensor = self.linear(x)
        h = F.relu(h)

        mu: Tensor = self.linear_mu(h)
        logvar: Tensor = self.linear_logvar(h)

        sigma: Tensor = torch.exp(0.5 * logvar)

        return (mu, sigma)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor) -> Tensor:
        h: Tensor = self.linear1(z)
        h = F.relu(h)

        h = self.linear2(h)
        x_hat: Tensor = F.sigmoid(h)

        return x_hat

def reparameterize(mu: Tensor, sigma: Tensor) -> Tensor:
    eps: Tensor = torch.randn_like(sigma)
    z: Tensor = mu + eps * sigma
    return z

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def get_loss(self, x: Tensor) -> Tensor:
        mu: Tensor
        sigma: Tensor
        (mu, sigma) = self.encoder(x)

        z: Tensor = reparameterize(mu, sigma)
        x_hat: Tensor = self.decoder(z)

        L1: Tensor = F.mse_loss(x_hat, x, reduction = "sum")
        L2: Tensor = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)

        batch_size: int = len(x)
        return (L1 + L2) / batch_size


#input_dim = 784  # mnist image 28x28
#hidden_dim = 200
#latent_dim = 20  # z vector dim
#
#epochs = 30
#learning_rate = 3e-4
#batch_size = 32
#
#transform = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Lambda(torch.flatten),
#])
#
#dataset = torchvision.datasets.MNIST(
#    root = ".tmp",
#    train = True,
#    transform = transform,
#    download = True,
#)
#
#dataloader = torch.utils.data.DataLoader(
#        dataset,
#        batch_size = batch_size,
#        shuffle = True,
#)
#
#model = VAE(input_dim, hidden_dim, latent_dim)
#optimizer = optim.Adam(model.parameters(), lr = learning_rate)
#
#losses = []
#
#for epoch in range(epochs):
#    loss_sum = 0.0
#    cnt = 0
#
#    for x, label in dataloader:
#        optimizer.zero_grad()
#        loss = model.get_loss(x)
#        loss.backward()
#        optimizer.step()
#
#        loss_sum += loss.item()
#        cnt += 1
#
#    loss_avg = loss_sum / cnt
#    losses.append(loss_avg)
#    print(f"loss_avg = {loss_avg}")
#
#
## plot losses
#epochs_list = list(range(1, epochs + 1))
#plt.plot(epochs_list, losses, marker = "o", linestyle = "-")
#plt.xlabel("epoch")
#plt.ylabel("loss")
#plt.show()
#
#
## generate image
#with torch.no_grad():
#    sample_size = 64
#    z = torch.randn(sample_size, latent_dim)
#    x = model.decoder(z)
#    generated_images = x.view(sample_size, 1, 28, 28)
#
#grid_img = torchvision.utils.make_grid(
#        generated_images,
#        nrow = 8,
#        padding = 2,
#        normalize = True,
#)
#
#plt.imshow(grid_img.permute(1, 2, 0))
#plt.axis("off")
#plt.show()



# U-Net

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding = 1),  # 3 : kernel size
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding = 1),  # 3 : kernel size
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convs(x)

class UNet(nn.Module):
    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)

        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(128 + 64, 64)

        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)  # 2 : kernel size
        self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")

    def forward(self, x: Tensor) -> Tensor:
        x1: Tensor
        x2: Tensor

        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)

        x = self.bot1(x)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim = 1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim = 1)
        x = self.up1(x)

        x = self.out(x)

        return x


#model = UNet()
#x = torch.randn(10, 1, 28, 28)
#y = model(x)
#print(f"y.shape = {y.shape}")


# t : scalar
# return v : (output_dim,)
def _pos_encoding(t: Tensor, output_dim: int, device = "cpu") -> Tensor:
    D: int = output_dim
    v: Tensor = torch.zeros(D, device = device)

    i: Tensor = torch.arange(0, D, device = device)
    div_term: Tensor = 10000 ** (i / D)

    v[0::2] = torch.sin(t / div_term[0::2])  # even num
    v[1::2] = torch.cos(t / div_term[1::2])  # odd num

    return v


#v = _pos_encoding(Tensor(1), 16)
#print(f"v.shape = {v.shape}")


# ts: (N,)
# return v : (N, output_dim)
def pos_encoding(ts: Tensor, output_dim: int, device = "cpu") -> Tensor:
    batch_size: int = len(ts)
    v: Tensor = torch.zeros(batch_size, output_dim, device = device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
    return v


#xs = Tensor([0, 1, 2, 3])
#y = pos_encoding(xs, 16)
#print(f"y.shape = {y.shape}")



class TimeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_embed_dim: int) -> None:
        super().__init__()

        self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding = 1),  # 3 : kernel size
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding = 1),  # 3 : kernel size
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
        )

        self.mlp = nn.Sequential(
                nn.Linear(time_embed_dim, in_ch),
                nn.ReLU(),
                nn.Linear(in_ch, in_ch),
        )

    # x : (N, C, H, W)
    # v : (N, time_embed_dim,)
    # mlp(v) : (N, C,)
    def forward(self, x: Tensor, v: Tensor) -> Tensor:
        N, C, _H, _W = x.shape

        v = self.mlp(v)
        v = v.view(N, C, 1, 1)

        y: Tensor = self.convs(x + v)
        return y

class TimeUNet(nn.Module):
    def __init__(self, in_ch: int = 1, time_embed_dim: int = 100) -> None:
        super().__init__()

        self.time_embed_dim: int = time_embed_dim

        self.down1 = TimeConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = TimeConvBlock(64, 128, time_embed_dim)

        self.bot1 = TimeConvBlock(128, 256, time_embed_dim)
        self.up2 = TimeConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = TimeConvBlock(128 + 64, 64, time_embed_dim)

        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)  # 2 : kernel size
        self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        x1: Tensor
        x2: Tensor

        v: Tensor = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim = 1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim = 1)
        x = self.up1(x, v)

        x = self.out(x)

        return x


# load data
file_path = os.path.join(SCRIPT_DIR, "dataset/flower.png")
image = plt.imread(file_path)
print(f"loaded image.shape = {image.shape}")

# image pre-proc
preprocess = transforms.ToTensor()
x = preprocess(image)
print(f"pre-processed image x.shape = {x.shape}")



def reverse_to_img(x: Tensor) -> Image.Image:
    x = x * 255.0
    x = x.clamp(0.0, 255.0)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)



T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
imgs = []

for tt in range(T):
    if tt % 100 == 0:
        img = reverse_to_img(x)
        imgs.append(img)

    beta = betas[tt]

    eps = torch.randn_like(x)

    # x_t <- x_t-1
    x = torch.sqrt(1.0 - beta) * x + torch.sqrt(beta) * eps

plt.figure(figsize = (15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Noise: {i * 100}")
    plt.axis("off")
plt.show()


def add_noise(x_0: Tensor, t: int, betas: Tensor) -> Tensor:
    T: int = len(betas)
    assert t >= 1 and t <= T

    alphas: Tensor = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim = 0)
    t_idx: int = t - 1
    alpha_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x_0)  # noise

    x_t: Tensor = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * eps

    return x_t


x = preprocess(image)  # reload

noise_t = 100
x_t = add_noise(x, noise_t, betas)

img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f"Noise: {noise_t}")
plt.axis("off")
plt.show()



class Diffuser:
    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            device = "cpu",
    ) -> None:
        self.num_timesteps: int = num_timesteps
        self.device = device

        self.betas: Tensor = torch.linspace(  # (timesteps,)
                beta_start,
                beta_end,
                num_timesteps,
                device = device,
        )

        self.alphas: Tensor = 1.0 - self.betas  # (timesteps,)
        self.alpha_bars: Tensor = torch.cumprod(self.alphas, dim = 0)  # (timesteps,)

    # x_0 : (N, C, H, W)
    # t : (timesteps,)
    def add_noise(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        T: int = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx: Tensor = t - 1

        alpha_bar = self.alpha_bars[t_idx]  # (timesteps,)
        N: int = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (timesteps, 1, 1, 1)

        noise: Tensor = torch.randn_like(x_0, device = self.device)  # (N, C, H, W)

        # x_t : (N, C, H, W)
        x_t: Tensor = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise

        return x_t, noise  # (N, C, H, W), (N, C, H, W)

    # x: (N, C, H, W)
    # t: (N,)
    def denoise(self, model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        T: int = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx: Tensor = t - 1  # (N,)
        alpha: Tensor = self.alphas[t_idx]  # (N,) <- (timesteps,)
        alpha_bar: Tensor = self.alpha_bars[t_idx]  # (N,) <- (timesteps,)
        alpha_bar_prev: Tensor = self.alpha_bars[t_idx - 1]  # (N,) <- (timesteps,)

        N: int = alpha.size(0)  # scalar <- (N,)
        alpha = alpha.view(N, 1, 1, 1)  # (N, 1, 1, 1) <- (N,)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1) <- (N,)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)  # (N, 1, 1, 1) <- (N,)

        model.eval()

        with torch.no_grad():
            eps: Tensor = model(x, t)  # (N, C, H, W) <- (N, C, H, W), (N,)

        model.train()

        noise = torch.randn_like(x, device = self.device)  # (N, C, H, W)
        noise[t == 1] = 0

        # mu : (N, C, H, W)
        # std : (N, C, H, W)
        mu: Tensor = (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * eps) / torch.sqrt(alpha)
        std: Tensor = torch.sqrt((1.0 - alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))

        return mu + noise * std  # (N, C, H, W)

    # x : (N, C, H, W)
    def reverse_to_img(self, x: Tensor) -> Image.Image:
        x = x * 255.0
        x = x.clamp(0.0, 255.0)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(
            self,
            model: nn.Module,
            x_shape: tuple[int, int, int, int] = (20, 1, 28, 28),
    ) -> list[Image.Image]:
        batch_size: int = x_shape[0]

        x: Tensor = torch.randn(x_shape, device = self.device)  # (N, C, H, W)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t: Tensor = torch.tensor(
                    [i] * batch_size,
                    device = self.device,
                    dtype = torch.long,
            )  # (N,)

            x = self.denoise(model, x, t)

        images: list[Image.Image] = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images


print(f"------------ U-Net and Diffuser ------------")

img_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


def show_images(images, rows = 2, cols = 10):
    fig = plt.figure(figsize = (cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap = "gray")
            plt.axis("off")
            i += 1
    plt.show()


preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
        root = ".tmp",
        download = True,
        transform = preprocess,
)
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
)
diffuser = Diffuser(num_timesteps, device = device)
model = TimeUNet()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)

losses = []

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    images = diffuser.sample(model)
    show_images(images)

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()

        x = images.to(device)
        t = torch.randint(1, num_timesteps + 1, (len(x),), device = device)

        x_noisy, noise = diffuser.add_noise(x, t)

        noise_pred = model(x_noisy, t)

        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"epoch = {epoch}, loss = {loss_avg}")

plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

images = diffuser.sample(model)
show_images(images)

