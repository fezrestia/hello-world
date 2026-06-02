import numpy as np

import resource

class Convolution:
    def __init__(self, W, b, stride = 1, padding = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding

        # for backward
        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    # x shape   : N, C, H, W
    # out shape : N, FN, OutH, OutW
    def forward(self, x):
        FN, C, FH, FW = self.W.shape  # filter num, channel, filter height, filter width
        N, C, H, W = x.shape  # x batch num, channel, height, width

        out_h = int(1 + (H + 2 * self.padding - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.padding - FW) / self.stride)

        # col shape : (N, out_h, out_w) x (C, FH, FW)
        col = resource.img2col(x, FH, FW, self.stride, self.padding)

        # col_W shape : (C, FH, FW) x FN
        col_W = self.W.reshape(FN, -1).T

        # out shape : (N, out_h, out_w) x FN
        out = np.dot(col, col_W) + self.b

        # (N, out_h, out_w) x FN
        #   -> N, out_h, out_w, FN
        #   -> N, FN, out_h, out_w
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # N, FN, out_h, out_w
        #   -> N, out_h, out_w, FN
        #   -> (N x out_h x out_w), FN
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis = 0)

        # col : (N, out_h, outw) x (C, FH, FW)
        # col.T : (C, FH, FW) x (N, out_h, out_w)
        # dout : (N, out_h, out_w), FN
        # col.T x dout : (C, FH, FW) x FN
        self.dW = np.dot(self.col.T, dout)

        # dW : (C, FH, FW) x FN
        # dW.T : FN x (C, FH, FW)
        # dW.T.reshape : FN, C, FH, FW
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dout : (N, out_h, out_w), FN
        # col_W : (C, FH, FW) x FN
        # col_W.T : FN x (C, FH, FW)
        # dcol : (N, out_h, out_w) x (C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)

        # dx : N, C, H, W
        dx = resource.col2img(dcol, self.x.shape, FH, FW, self.stride, self.padding)

        return dx

