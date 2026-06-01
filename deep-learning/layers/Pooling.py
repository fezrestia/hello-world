import numpy as np

import resource

class Pooling:
    def __init__(self, pool_h, pool_w, stride = 2, padding = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        # for backward
        self.x = None
        self.arg_max = None

    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = resource.img2col(x, self.pool_h, self.pool_w, self.stride, self.padding)

        # col : (N, out_h, out_w) x (C, pool_h, pool_w)
        #         -> (N, out_h, out_w, C) x (pool_h, pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.arg_max = np.argmax(col, axis = 1)
        out = np.max(col, axis = 1)  # for (pool_h, pool_w) axis

        # out : (N, out_h, out_w, C) x (pool_h, pool_w)
        #         -> (N, out_h, out_w, C) x 1
        #         -> (N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        # dout : (N, C, out_h, out_w)
        #          -> (N, out_h, out_w, C)
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w

        # dmax : (N x C x out_h x out_w), pool_size
        dmax = np.zeros((dout.size, pool_size))

        # np.arange(arg_max.size) : [0, 1, 2, 3, 4, ...]
        # arg_max.flatten()       : [1, 2, 1, 4, 3, ...]
        # -> dmax(0, 1), dmax(1, 2), dmax(2, 1), dmax(3, 4), dmax(4, 3) ...
        # -> = dout.flatten() [10, 20, 30, 40, 50, ...]
        # -> [
        #      [0,  10,  0,  0,  0, ...],
        #      [0,  0,  20,  0,  0, ...],
        #      [0,  30,  0,  0,  0, ...],
        #      [0,  0,   0,  0, 40, ...],
        #      [0,  0,   0, 50,  0, ...],
        #      ...
        #    ]
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # dmax : (N x C x out_h x out_w x pool_size)
        #          -> (N, out_h, out_w, C, pool_size)
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # dcol : (N x out_h x out_w, C x pool_size)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        # dx : N, C, H, W
        dx = resource.col2img(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)

        return dx

