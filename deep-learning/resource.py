import numpy as np
import matplotlib.pyplot as plot
import importlib
import sys
import os
import gzip
import pickle
from PIL import Image

def reload():
    importlib.reload(sys.modules[__name__])


# Natural Languagge

def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_vs_id = {}
    id_vs_word = {}

    for word in words:
        if word not in word_vs_id:
            new_id = len(word_vs_id)
            word_vs_id[word] = new_id
            id_vs_word[new_id] = word

    copus = np.array([word_vs_id[w] for w in words])

    return copus, word_vs_id, id_vs_word

def create_co_matrix(corpus, vocab_size, window_size = 1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype = np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps = 1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_vs_id, id_vs_word, word_vs_vec, top = 5):
    if query not in word_vs_id:
        print(f"query = {query} is not found.")
        return

    print(f"\n[query] {query}")

    query_id = word_vs_id[query]
    query_vec = word_vs_vec[query_id]

    vocab_size = len(id_vs_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_vs_vec[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_vs_word[i] == query:
            continue

        print(f"{id_vs_word[i]} : {similarity[i]}")

        count += 1
        if count >= top:
            return

def ppmi(word_vs_vec, verbose = False, eps = 1e-8):
    M = np.zeros_like(word_vs_vec, dtype = np.float32)
    N = np.sum(word_vs_vec)
    S = np.sum(word_vs_vec, axis = 0)
    total = word_vs_vec.shape[0] * word_vs_vec.shape[1]
    cnt = 0

    for i in range(word_vs_vec.shape[0]):
        for j in range(word_vs_vec.shape[1]):
            pmi = np.log2(word_vs_vec[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0.0, pmi)

            if verbose:
                cnt += 1

                if cnt % (total // 100 + 1) == 0:
                    print(f"done rate = {100.0 * cnt / total:.2f} %")

    return M

def create_contexts_vs_target(corpus, window_size = 1):
    target = corpus[window_size : -window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        ctx = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                # t == 0 means target position
                continue
            ctx.append(corpus[idx + t])
        contexts.append(ctx)

    return np.array(contexts), np.array(target)

# ids : 1 or 2 dimens word id list
# vocab_size : num of words
# return : 2 or 3 dimens onehot array
def convert_idx_to_onehot(ids, vocab_size):
    N = ids.shape[0]

    if ids.ndim == 1:
        # target word id list.
        onehot = np.zeros((N, vocab_size), dtype = np.int32)
        for n, word_id in enumerate(ids):
            onehot[n, word_id] = 1

    elif ids.ndim == 2:
        # context word id set list.
        ctx_num = ids.shape[1]
        onehot = np.zeros((N, ctx_num, vocab_size), dtype = np.int32)
        for n, ctx_ids in enumerate(ids):
            for i, ctx_id in enumerate(ctx_ids):
                onehot[n, i, ctx_id] = 1

    return onehot

def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

def analogy(a, b, x, word_vs_id, id_vs_word, word_vs_vec, top = 5, answer = None):
    for word in (a, b, x):
        if word not in word_vs_id:
            print(f"{word} is not found.")
            return

    print(f"\n[analogy] {a}->{b} : {x}->?")
    a_vec = word_vs_vec[word_vs_id[a]]
    b_vec = word_vs_vec[word_vs_id[b]]
    x_vec = word_vs_vec[word_vs_id[x]]

    query_vec = b_vec - a_vec + x_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_vs_vec, query_vec)

    if answer is not None:
        if answer not in word_vs_id:
            print(f"answer = {answer} is not found")
        else:
            print(f"  > {answer} : {str(np.dot(word_vs_vec[word_vs_id[answer]], query_vec))}")

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_vs_word[i] in (a, b, x):
            continue
        print(f"  {format(id_vs_word[i])} : {similarity[i]}")

        count += 1
        if count >= top:
            return



# MNIST

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _load_label(file_path):
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def _load_img(file_path):
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    return data

def _change_to_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_mnist(dir_path, normalize = True, flatten = True, one_hot_label = False):
    pkl_path = dir_path + "/" + "mnist.pkl"

    if not os.path.exists(pkl_path):
        train_img_file = dir_path + "/" + "train-images-idx3-ubyte.gz"
        train_label_file = dir_path + "/" + "train-labels-idx1-ubyte.gz"
        test_img_file = dir_path + "/" + "t10k-images-idx3-ubyte.gz"
        test_label_file = dir_path + "/" + "t10k-labels-idx1-ubyte.gz"

        dataset = {}
        dataset["train_img"] = _load_img(train_img_file)
        dataset["train_label"] = _load_label(train_label_file)
        dataset["test_img"] = _load_img(test_img_file)
        dataset["test_label"] = _load_label(test_label_file)

        with open(pkl_path, "wb") as f:
            pickle.dump(dataset, f, -1)

    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset["train_label"] = _change_to_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_to_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])

def mnist_flatten_img_show(flatten_img):
    img = flatten_img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()



# Logic Func

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.2

    result = np.sum(w * x) + b

    if result <= 0.0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    result = AND(s1, s2)
    return result



# Func

def identity_func(x):
    return x

def step_func(x):
    y = x > 0.0
    return y.astype(int)

def sigmoid_func(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu_func(x):
    return np.maximum(0.0, x)

def softmax_func(x):
    c = np.max(x, axis = -1, keepdims = True)  # for last dim
    exp_x = np.exp(x - c)  # for anti overflow
    sum_exp_x = np.sum(exp_x, axis = -1, keepdims = True)
    y = exp_x / sum_exp_x
    return y



# Numerical Func

def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

def _numerical_gradient(f, x):
    h = 1e-4  # 0.0001

    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # recover

    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient(f, x)

        return grad

# input data shape  : N(batch num), Channel, Height, Width
# output data shape : (N x OutH x OutW), (Channel x filter_h x filter_w)
def img2col(input_nchw_data, filter_h, filter_w, stride = 1, padding = 0):
    N, C, H, W = input_nchw_data.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(input_nchw_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

# col : reshapable to (N, out_h, out_w, C, filter_h, filter_w)
# img : N, C, H, W
def col2img(col, input_shape, filter_h, filter_w, stride = 1, padding = 0):
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    # col shape : (N x OutH x OutW), (Channel x filter_h x filter_w)
    #               -> N, OutH, OutW, Channel, filter_h, filter_w
    #               -> N, Channel, filter_h, filter_w, OutH, OutW
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding]



# Learning Method

# lr : learning late
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)



# Evaluate Func

def sum_squared_error(act, exp):
    return 0.5 * np.sum((act - exp) ** 2)

def cross_entropy_error(act, exp):
    if act.ndim == 1:
        act = act.reshape(1, act.size)
        exp = exp.reshape(1, exp.size)

    delta = 1e-7
    batch_size = act.shape[0]

    if exp.size == act.size:
        # Change one-hot-vector to grand truth label.
        exp = exp.argmax(axis = 1)

    return -1.0 * np.sum(np.log(act[np.arange(batch_size), exp] + delta)) / batch_size



# Trainer

# integrate grads for shared params.
def remove_duplicate(params, grads):
    params = params[:]  # shallow copy
    grads = grads[:]  # shallow copy

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # shared weight
                if params[i] is params[j]:
                    grads[i] += grads[j]  # integrate grads
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # shared weight with transpose (weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T  # integrate grads
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

# clip grads range.
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1.0:
        for grad in grads:
            grad *= rate

def eval_perplexity(model, corpus, batch_size = 10, time_size = 35):
    corpus_size = len(corpus)
    total_loss = 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, is_training = False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        print(f"calc perplexity ... [{iters}/{max_iters}]")

    ppl = np.exp(total_loss / max_iters)
    return ppl



# Graph

# elms : [ {x, y, linestyle, label}, ... ]
def plot_graph(elms):
    plt.figure()
    for elm in elms:
        plot.plot(elm["x"], elm["y"], linestyle=elm["linestyle"], label=elm["label"])
    plot.xlabel("x")
    plot.ylabel("y")
    plot.title("graph")
    plot.legend()
    plot.show()

def plot_2d(pos_x, pos_y, xlim, ylim, x_label, y_label):
    plot.figure()
    plot.plot([xlim[0] - 1.0, xlim[1] + 1.0], [0, 0], "--b")
    plot.plot([0, 0], [ylim[0] - 1.0, ylim[1] + 1.0], "--b")
    plot.plot(pos_x, pos_y, 'o')
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.show()



# pos_x/y, vec_x/y : array
# xlim, ylim : array [2], min, max
def plot_2d_vector_field(pos_x, pos_y, vec_x, vec_y, xlim, ylim, x_label, y_label):
    plot.figure()
    plot.quiver(pos_x, pos_y, vec_x, vec_y, angles="xy")
    plot.xlim(xlim)
    plot.ylim(ylim)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.grid()
    plot.draw()
    plot.show()







