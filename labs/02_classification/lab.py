"""实验 2：二分类——逻辑回归 vs 决策树（双月牙数据）。

双月牙用 numpy 手工生成；对比线性模型与树模型的决策边界形状。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import logistic  # noqa: E402
import trees  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_moons(n=300, noise=0.15, seed=0):
    """sklearn make_moons 的 numpy 复刻（半圆 + 半圆下移错开）。"""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    t1 = rng.uniform(0, np.pi, n_half)
    t2 = rng.uniform(0, np.pi, n - n_half)
    upper = np.column_stack([np.cos(t1), np.sin(t1)])
    lower = np.column_stack([1 - np.cos(t2), 0.5 - np.sin(t2)])
    X = np.vstack([upper, lower]) + rng.normal(0, noise, (n, 2))
    y = np.array([0] * n_half + [1] * (n - n_half))
    return X, y


X, y = make_moons()
w, hist = logistic.fit(X, y, lr=1.0, epochs=3000)
acc_lr = logistic.accuracy(y, logistic.predict(X, w))
clf = trees.DecisionTreeClassifier(max_depth=4).fit(X, y)
acc_tree = float(np.mean(clf.predict(X) == y))

grid = 200
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, grid),
                     np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, grid))
pts = np.c_[xx.ravel(), yy.ravel()]
p_lr = logistic.predict_proba(pts, w).reshape(xx.shape)
p_tree = clf.predict(pts).reshape(xx.shape).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, p, name, acc in [(axes[0], p_lr, f"逻辑回归 acc={acc_lr:.3f}", acc_lr),
                         (axes[1], p_tree, f"决策树(max_depth=4) acc={acc_tree:.3f}", acc_tree)]:
    ax.contourf(xx, yy, p, levels=20, cmap="coolwarm", alpha=0.7)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="b", s=12, label="类 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="r", s=12, label="类 1")
    ax.set_title(name)
    ax.legend()
fig.suptitle("线性模型 vs 树模型：决策边界形状")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "decision_boundaries.png"), dpi=120)

metrics = {"acc_logistic": acc_lr, "acc_tree_depth4": acc_tree,
           "logistic_w": w.tolist(), "tree_max_depth": 4}
with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"\n图已保存: {os.path.join(HERE, 'decision_boundaries.png')}")
