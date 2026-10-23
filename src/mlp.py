"""全连接神经网络：numpy 手写前向与反向传播。

- 隐藏层激活 tanh / relu；输出层 softmax + 交叉熵。
- 提供 numerical_gradient 供梯度检验（中心差分），这是本模块正确性的金标准。
"""
import numpy as np


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # 数值稳定
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


ACTIVATIONS = {"tanh": (tanh, lambda z: 1.0 - np.tanh(z) ** 2),
               "relu": (relu, lambda z: (z > 0).astype(float))}


class MLP:
    """多层感知机。

    layer_sizes 形如 [2, 8, 8, 3]：输入维 → 隐藏层 → 类别数。
    参数按 He 初始化（relu）/ Xavier（tanh）。
    """

    def __init__(self, layer_sizes, activation="tanh", seed=0):
        rng = np.random.default_rng(seed)
        self.activation_name = activation
        self.W, self.b = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.W.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        self._f, self._df = ACTIVATIONS[activation]

    # ---- 前向 ----
    def forward(self, X):
        """返回 (probabilities, cache)。cache 保存各层线性输出与激活值。"""
        A = np.asarray(X, dtype=float)
        linear_outs, activations = [], [A]
        L = len(self.W)
        for i in range(L):
            Z = A @ self.W[i] + self.b[i]
            linear_outs.append(Z)
            A = softmax(Z) if i == L - 1 else self._f(Z)
            activations.append(A)
        return A, (linear_outs, activations)

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # ---- 损失与梯度 ----
    def loss(self, X, y):
        """平均交叉熵。y 为整数类别标签。"""
        probs = self.predict_proba(X)
        n = len(y)
        return float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))

    def backward(self, X, y):
        """解析梯度（反向传播），返回 (grad_W, grad_b) 列表。"""
        linear_outs, activations = self.forward(X)[1]
        n = len(y)
        onehot = np.zeros_like(activations[-1])
        onehot[np.arange(n), y] = 1.0
        delta = (activations[-1] - onehot) / n  # softmax+CE 的联合导数
        grad_W = [None] * len(self.W)
        grad_b = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            grad_W[i] = activations[i].T @ delta
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._df(linear_outs[i - 1])
        return grad_W, grad_b

    def fit(self, X, y, lr=0.5, epochs=500, seed=None):
        """全批量梯度下降。返回 loss 历史。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        history = []
        for _ in range(epochs):
            grad_W, grad_b = self.backward(X, y)
            for i in range(len(self.W)):
                self.W[i] -= lr * grad_W[i]
                self.b[i] -= lr * grad_b[i]
            history.append(self.loss(X, y))
        return history


def numerical_gradient(model, X, y, eps=1e-5):
    """中心差分数值梯度，用于与 backward() 互相验证。"""
    num_W = [np.zeros_like(w) for w in model.W]
    num_b = [np.zeros_like(b) for b in model.b]
    for i in range(len(model.W)):
        it = np.nditer(model.W[i], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = model.W[i][idx]
            model.W[i][idx] = orig + eps
            lp = model.loss(X, y)
            model.W[i][idx] = orig - eps
            lm = model.loss(X, y)
            model.W[i][idx] = orig
            num_W[i][idx] = (lp - lm) / (2 * eps)
            it.iternext()
    for i in range(len(model.b)):
        it = np.nditer(model.b[i], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = model.b[i][idx]
            model.b[i][idx] = orig + eps
            lp = model.loss(X, y)
            model.b[i][idx] = orig - eps
            lm = model.loss(X, y)
            model.b[i][idx] = orig
            num_b[i][idx] = (lp - lm) / (2 * eps)
            it.iternext()
    return num_W, num_b
