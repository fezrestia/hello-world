import numpy as np

def load_data(seed = 1984):
    np.random.seed(seed)

    N = 100  # samples for each class
    DIM = 2  # data dimension
    CLS_NUM = 3  # class num

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype = np.int32)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t

def load_train_data(seed = 1984):
    np.random.seed(seed)

    N = 100  # samples for each class
    DIM = 2  # data dimension
    CLS_NUM = 3  # class num

    x = np.zeros((N * CLS_NUM, DIM), dtype = np.float32)
    t = np.zeros(N * CLS_NUM, dtype = int)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j

    indices = np.random.permutation(N * CLS_NUM)
    x = x[indices]
    t = t[indices]

    return x, t

def load_test_data():
    return load_train_data(seed = 2020)

