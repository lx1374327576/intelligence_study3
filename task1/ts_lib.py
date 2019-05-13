import numpy as np


def fc(x, w, b):
    out = x.reshape(x.shape[0], np.prod(x.shape[1:])).dot(w) + b
    cache = (x, w, b)
    return out, cache


def fc_back(dout, cache):
    x, w, b = cache
    db = np.sum(dout, axis=0)
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], w.shape[0]).T.dot(dout)
    return dx, dw, db


def softmax(x, y):

    minus = x - np.max(x, axis=1, keepdims=True)
    x_log = minus - np.log(np.sum(np.exp(minus), axis=1, keepdims=True))
    x_exp = np.exp(x_log)
    N = x.shape[0]
    loss = -np.sum(x_log[np.arange(N), y]) / N
    dx = x_exp.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def relu(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_back(dout, cache):
    x = cache
    dx = dout
    dx[x <= 0] = 0
    return dx


def conv(x, w, b, stride, pad):
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_new = 1 + int((H + 2*pad - HH) / stride)
    W_new = 1 + int((W + 2*pad - WW) / stride)
    out = np.zeros((N, F, H_new, W_new))
    for i in range(N):
        for j in range(F):
            for k in range(H_new):
                for l in range(W_new):
                    out[i, j, k, l] = np.sum(x_pad[i, :, k*stride:k*stride+HH, l*stride:l*stride+WW] * w[j]) + b[j]
    cache = (x, w, b, stride, pad)
    return out, cache


def conv_back(dout, cache):
    x, w, b, stride, pad = cache
    dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_new, W_new = dout.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_pad = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    db = np.sum(dout, axis=(0, 2, 3))
    for i in range(N):
        for j in range(F):
            for k in range(H_new):
                for l in range(W_new):
                    dw[j] += x_pad[i, :, k * stride:k * stride + HH, l * stride:l * stride + WW] * dout[i, j, k, l]
                    dx_pad[i, :, k * stride:k * stride + HH, l * stride:l * stride + WW] += dout[i, j, k, l] * w[j]
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db


def max_pool(x, pool_height, pool_width, stride):
    N, C, H, W = x.shape
    H_new, W_new = int(1 + (H - pool_height) / stride), int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    out[i, j, k, l] = np.max(x[i, j, k*stride:k*stride+pool_height, l*stride:l*stride+pool_width])
    cache = (x, pool_height, pool_width, stride)
    return out, cache


def max_pool_back(dout, cache):
    x, pool_height, pool_width, stride = cache
    N, C, H, W = x.shape
    N, C, H_new, W_new = dout.shape
    dx = np.zeros(x.shape)
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    m = np.max(x[i, j, k*stride:k*stride+pool_height, l*stride:l*stride+pool_width])
                    dx[i, j, k * stride:k * stride + pool_height, l * stride:l * stride + pool_width] \
                        += (x[i, j, k*stride:k*stride+pool_height, l*stride:l*stride+pool_width] == m) * \
                           dout[i, j, k, l]
    return dx
