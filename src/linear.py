"""线性回归：闭式解（正规方程 / 岭回归）与梯度下降。

只依赖 numpy。约定输入 X 形状 (n, d)，y 形状 (n,)。
"""
import numpy as np


def design_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(len(X)), X])


def closed_form(X, y, ridge=0.0):
    """正规方程求解。ridge>0 时为岭回归（截距不正则）。"""
    Xb = design_matrix(X)
    y = np.asarray(y, dtype=float)
    A = Xb.T @ Xb
    if ridge > 0:
        penalty = np.eye(A.shape[0]) * ridge
        penalty[0, 0] = 0.0  # 截距不正则
        A = A + penalty
    return np.linalg.solve(A, Xb.T @ y)


def grad_descent(X, y, lr=0.05, epochs=2000, ridge=0.0):
    """批量梯度下降。返回 (w, loss_history)。"""
    Xb = design_matrix(X)
    y = np.asarray(y, dtype=float)
    w = np.zeros(Xb.shape[1])
    history = []
    n = len(y)
    for _ in range(epochs):
        residual = Xb @ w - y
        grad = 2.0 / n * (Xb.T @ residual)
        if ridge > 0:
            grad[1:] += 2.0 * ridge / n * w[1:]
        w -= lr * grad
        history.append(float(np.mean(residual ** 2)))
    return w, history


def predict(X, w):
    return design_matrix(X) @ w


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))
