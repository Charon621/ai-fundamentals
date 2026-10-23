import numpy as np
import pytest

import linear


def make_data(n=200, noise=0.3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 1))
    y = 4.0 + 2.5 * X[:, 0] + rng.normal(0, noise, n)
    return X, y


def test_closed_form_recovers_true_line():
    X, y = make_data()
    w = linear.closed_form(X, y)
    assert w[0] == pytest.approx(4.0, abs=0.1)
    assert w[1] == pytest.approx(2.5, abs=0.05)


def test_gd_converges_to_closed_form():
    X, y = make_data()
    w_cf = linear.closed_form(X, y)
    w_gd, hist = linear.grad_descent(X, y, lr=0.1, epochs=5000)
    assert np.allclose(w_cf, w_gd, atol=1e-3)
    assert hist[-1] < hist[0]  # 损失单调下降（全批量、固定学习率下应成立）


def test_ridge_shrinks_weights():
    X, y = make_data(noise=2.0)
    w0 = linear.closed_form(X, y, ridge=0.0)
    w1 = linear.closed_form(X, y, ridge=100.0)
    assert abs(w1[1]) < abs(w0[1])  # 强正则使斜率收缩
    # 截距不正则：接近 y 的均值
    assert w1[0] == pytest.approx(float(np.mean(y)), abs=0.5)


def test_mse_matches_manual():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])
    assert linear.mse(y, pred) == pytest.approx(1.0 / 3.0)
