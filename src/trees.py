"""CART 分类树：基尼不纯度、二叉递归划分、多数投票预测。

只实现分类树的核心机制（划分准则 + 递归 + 剪枝超参数），
不追求 sklearn 的工程完整度；配合测试验证划分正确性。
"""
import numpy as np


def gini(y):
    """基尼不纯度：1 - Σ p_k²。空数组返回 0。"""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(1.0 - np.sum(p ** 2))


def weighted_gini(y_left, y_right):
    n = len(y_left) + len(y_right)
    return (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / n


def best_split(X, y):
    """穷举所有特征的所有中点阈值，返回 (feature, threshold, gain) 或 None。"""
    best = None
    parent = gini(y)
    n, d = X.shape
    for f in range(d):
        values = np.unique(X[:, f])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for t in thresholds:
            mask = X[:, f] <= t
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            wg = weighted_gini(y[mask], y[~mask])
            gain = parent - wg
            if best is None or gain > best[2]:
                best = (f, float(t), float(gain))
    return best


def majority(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "label")


def _build(X, y, depth, max_depth, min_samples_split):
    node = _Node()
    node.label = None
    if depth >= max_depth or len(y) < min_samples_split or gini(y) == 0.0:
        node.label = majority(y)
        return node
    split = best_split(X, y)
    if split is None or split[2] <= 1e-12:
        node.label = majority(y)
        return node
    f, t, _ = split
    mask = X[:, f] <= t
    node.feature = f
    node.threshold = t
    node.left = _build(X[mask], y[mask], depth + 1, max_depth, min_samples_split)
    node.right = _build(X[~mask], y[~mask], depth + 1, max_depth, min_samples_split)
    return node


def _predict_one(node, x):
    while node.label is None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.label


class DecisionTreeClassifier:
    def __init__(self, max_depth=4, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root_ = _build(X, y, 0, self.max_depth, self.min_samples_split)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([_predict_one(self.root_, x) for x in X])
