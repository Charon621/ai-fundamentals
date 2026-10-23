import numpy as np
import pytest

import conv


def test_conv2d_matches_naive_reference():
    """im2col 路径与六重循环参考实现一致（随机输入、多通道、padding）。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 3, 7, 7))
    K = rng.normal(size=(4, 3, 3, 3))
    for padding in (0, 1):
        fast = conv.conv2d(X, K, padding=padding)
        slow = conv.conv2d_naive(X, K, stride=1, padding=padding)
        assert fast.shape == slow.shape
        assert np.allclose(fast, slow, atol=1e-10)


def test_conv2d_bias_adds_per_channel():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 3, 5, 5))
    K = rng.normal(size=(4, 3, 3, 3))
    b = rng.normal(size=4)
    with_bias = conv.conv2d(X, K, b)
    without = conv.conv2d(X, K)
    assert np.allclose(with_bias, without + b[None, :, None, None])


def test_conv2d_known_hand_case():
    """1D 化的手工算例：全 1 核对全 1 输入，输出等于窗口元素个数。"""
    X = np.ones((1, 1, 4, 4))
    K = np.ones((1, 1, 3, 3))
    out = conv.conv2d(X, K)
    assert out.shape == (1, 1, 2, 2)
    assert np.all(out == 9.0)


def test_maxpool2d_known_case():
    X = np.array([[[[1, 2, 5, 6],
                    [3, 4, 7, 8],
                    [9, 10, 13, 14],
                    [11, 12, 15, 16]]]], dtype=float)
    out = conv.maxpool2d(X, size=2)
    assert out.shape == (1, 1, 2, 2)
    assert out[0, 0].tolist() == [[4.0, 8.0], [12.0, 16.0]]


def test_edge_kernel_detects_vertical_edge():
    """垂直边缘核在左右亮暗交界处应有强响应。"""
    img = np.zeros((1, 1, 8, 8))
    img[0, 0, :, 4:] = 1.0  # 右半亮
    response = conv.conv2d(img, conv.edge_kernel())[0, 0]
    assert np.abs(response).max() > 0.5
    # 平坦区域响应为零
    flat = np.full((1, 1, 8, 8), 0.5)
    assert np.allclose(conv.conv2d(flat, conv.edge_kernel()), 0.0)


def test_conv_gradient_check():
    """对卷积核做中心差分数值梯度检验。"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1, 2, 5, 5))
    K = rng.normal(size=(2, 2, 3, 3)) * 0.1

    def loss(k):
        return float((conv.conv2d(X, k) ** 2).sum())

    num = np.zeros_like(K)
    eps = 1e-5
    it = np.nditer(K, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = K[i]
        K[i] = orig + eps
        lp = loss(K)
        K[i] = orig - eps
        lm = loss(K)
        K[i] = orig
        num[i] = (lp - lm) / (2 * eps)
        it.iternext()

    # 解析梯度：d/dK (Y²).sum = 2 * X_cols @ Yᵀ，这里直接用 im2col 验证
    cols, oh, ow = conv.im2col(X, 3, 3)
    Y = conv.conv2d(X, K)
    grad = (2 * (Y.reshape(2, -1)) @ cols.T).reshape(K.shape)
    denom = max(np.linalg.norm(num) + np.linalg.norm(grad), 1e-8)
    assert np.linalg.norm(num - grad) / denom < 1e-6
