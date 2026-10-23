import numpy as np
import pytest

import mlp


def make_model():
    return mlp.MLP([3, 5, 4, 2], activation="tanh", seed=0)


def test_forward_shapes():
    model = make_model()
    X = np.random.default_rng(1).normal(size=(10, 3))
    probs = model.predict_proba(X)
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_gradient_check_tanh():
    """数值梯度检验：解析梯度与中心差分相对误差 < 1e-6。"""
    model = make_model()
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 3))
    y = np.array([0, 1, 1, 0, 1])
    grad_W, grad_b = model.backward(X, y)
    num_W, num_b = mlp.numerical_gradient(model, X, y)
    for a, b in zip(grad_W + grad_b, num_W + num_b):
        denom = max(np.linalg.norm(a) + np.linalg.norm(b), 1e-8)
        assert np.linalg.norm(a - b) / denom < 1e-6


def test_gradient_check_relu():
    model = mlp.MLP([4, 6, 3], activation="relu", seed=3)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6, 4)) + 0.1  # 远离 relu 折点，数值差分稳定
    y = np.array([0, 2, 1, 0, 2, 1])
    grad_W, grad_b = model.backward(X, y)
    num_W, num_b = mlp.numerical_gradient(model, X, y)
    for a, b in zip(grad_W + grad_b, num_W + num_b):
        denom = max(np.linalg.norm(a) + np.linalg.norm(b), 1e-8)
        assert np.linalg.norm(a - b) / denom < 1e-6


def test_mlp_solves_xor():
    """经典验收：2-8-2 网络必须解决 XOR。"""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    model = mlp.MLP([2, 8, 2], activation="tanh", seed=42)
    hist = model.fit(X, y, lr=1.0, epochs=2000)
    assert model.predict(X).tolist() == [0, 1, 1, 0]
    assert hist[-1] < 0.05


def test_loss_decreases_on_training():
    rng = np.random.default_rng(5)
    X0 = rng.normal(-1, 0.5, size=(30, 2))
    X1 = rng.normal(1, 0.5, size=(30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)
    model = mlp.MLP([2, 8, 2], activation="tanh", seed=6)
    hist = model.fit(X, y, lr=0.5, epochs=500)
    assert hist[-1] < hist[0] * 0.2
    assert np.mean(model.predict(X) == y) > 0.95
