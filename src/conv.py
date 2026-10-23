"""卷积与池化的前向传播：im2col 实现 + 逐元素数值验证。

覆盖 CNN 的核心计算机制；完整 CNN 训练需要框架级性能，超出 numpy 教学范围（见 README 验证边界）。
"""
import numpy as np


def im2col(X, kh, kw, stride=1, padding=0):
    """(n, c, h, w) -> (kh*kw*c, n*out_h*out_w) 列块矩阵。"""
    n, c, h, w = X.shape
    if padding > 0:
        X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        h, w = h + 2 * padding, w + 2 * padding
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    cols = np.empty((c, kh, kw, n, out_h, out_w))
    for i in range(kh):
        for j in range(kw):
            patch = X[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride]
            cols[:, i, j] = patch.transpose(1, 0, 2, 3)
    return cols.reshape(c * kh * kw, n * out_h * out_w), out_h, out_w


def conv2d(X, kernel, bias=None, stride=1, padding=0):
    """2D 卷积（互相关实现，与深度学习框架一致）。

    X: (n, c_in, h, w)；kernel: (c_out, c_in, kh, kw)；bias: (c_out,) 或 None。
    """
    c_out, c_in, kh, kw = kernel.shape
    cols, out_h, out_w = im2col(X, kh, kw, stride, padding)
    K = kernel.reshape(c_out, -1)
    out = K @ cols  # (c_out, n*out_h*out_w)
    if bias is not None:
        out += bias[:, None]
    return out.reshape(c_out, len(X), out_h, out_w).transpose(1, 0, 2, 3)


def maxpool2d(X, size=2):
    """非重叠最大池化。(n,c,h,w) -> (n,c,h//size,w//size)。"""
    n, c, h, w = X.shape
    oh, ow = h // size, w // size
    trimmed = X[:, :, :oh * size, :ow * size]
    blocks = trimmed.reshape(n, c, oh, size, ow, size)
    return blocks.max(axis=(3, 5))


def edge_kernel():
    """3x3 垂直边缘检测核（Prewitt），单通道。"""
    return np.array([[1, 0, -1],
                     [1, 0, -1],
                     [1, 0, -1]], dtype=float).reshape(1, 1, 3, 3)


def conv2d_naive(X, kernel, stride=1, padding=0):
    """六重循环参考实现，用于交叉验证 im2col 路径。"""
    c_out, c_in, kh, kw = kernel.shape
    if padding > 0:
        X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    n, c, h, w = X.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.zeros((n, c_out, out_h, out_w))
    for b in range(n):
        for co in range(c_out):
            for i in range(out_h):
                for j in range(out_w):
                    acc = 0.0
                    for ci in range(c_in):
                        for ki in range(kh):
                            for kj in range(kw):
                                acc += (X[b, ci, i * stride + ki, j * stride + kj]
                                        * kernel[co, ci, ki, kj])
                    out[b, co, i, j] = acc
    return out
