import numpy as np
import pytest

import logistic


def make_separable(n=300, seed=7):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(-2.0, 0.8, size=(n // 2, 2))
    X1 = rng.normal(2.0, 0.8, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_sigmoid_extremes_and_center():
    z = np.array([-1000.0, 0.0, 1000.0])
    s = logistic.sigmoid(z)
    assert s[0] == pytest.approx(0.0, abs=1e-12)
    assert s[1] == pytest.approx(0.5)
    assert s[2] == pytest.approx(1.0, abs=1e-12)


def test_separable_data_fits_perfectly():
    X, y = make_separable()
    w, hist = logistic.fit(X, y, lr=1.0, epochs=1000)
    assert logistic.accuracy(y, logistic.predict(X, w)) == 1.0
    assert hist[-1] < hist[0]


def test_gradient_check():
    """解析梯度与中心差分数值梯度一致。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.integers(0, 2, size=20)
    w = rng.normal(size=4)

    def f(w):
        return logistic.loss(X, y, w, l2=0.1)

    num = np.zeros_like(w)
    eps = 1e-5
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        num[i] = (f(wp) - f(wm)) / (2 * eps)

    Xb = np.column_stack([np.ones(len(X)), X])
    p = logistic.sigmoid(Xb @ w)
    ana = Xb.T @ (p - y) / len(y)
    ana[1:] += 0.1 * w[1:] / len(y)
    assert np.allclose(num, ana, atol=1e-6)
