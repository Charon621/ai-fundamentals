"""逻辑回归：批量梯度下降 + L2 正则，二分类。

标签约定 y ∈ {0, 1}。
"""
import numpy as np


def sigmoid(z):
    """数值稳定的 sigmoid。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _design(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(len(X)), X])


def loss(X, y, w, l2=0.0):
    """平均交叉熵 + L2（不含截距）。"""
    Xb = _design(X)
    p = sigmoid(Xb @ w)
    eps = 1e-12
    ce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return float(ce + l2 * np.sum(w[1:] ** 2) / (2 * len(y)))


def fit(X, y, lr=0.5, epochs=2000, l2=0.0):
    Xb = _design(X)
    y = np.asarray(y, dtype=float)
    w = np.zeros(Xb.shape[1])
    history = []
    n = len(y)
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n
        w -= lr * grad
        history.append(loss(X, y, w, l2))
    return w, history


def predict_proba(X, w):
    return sigmoid(_design(X) @ w)


def predict(X, w, threshold=0.5):
    return (predict_proba(X, w) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
