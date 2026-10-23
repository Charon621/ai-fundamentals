import numpy as np
import pytest

import trees


def test_gini_pure_and_balanced():
    assert trees.gini(np.array([1, 1, 1])) == pytest.approx(0.0)
    assert trees.gini(np.array([0, 1])) == pytest.approx(0.5)
    assert trees.gini(np.array([0, 1, 2, 3])) == pytest.approx(0.75)
    assert trees.gini(np.array([], dtype=int)) == 0.0


def test_best_split_finds_decision_boundary():
    # 单特征：x<=2 为类 0，x>2 为类 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    f, t, gain = trees.best_split(X, y)
    assert f == 0
    assert t == pytest.approx(2.5)
    assert gain == pytest.approx(0.5)  # 1.0 -> 0.5


def make_xor_points(n_per_corner=40, seed=1):
    """四角聚类的 XOR 数据。注意：均匀随机分布会让贪心 CART 卡在
    局部最优划分（见 docs/实验记录.md 的记录），聚类版本是标准教学设定。"""
    rng = np.random.default_rng(seed)
    corners = [(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)]
    X = np.vstack([rng.normal(c, 0.05, size=(n_per_corner, 2)) for c in corners])
    X = np.clip(X, 0, 1)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    return X, y


def test_tree_fits_xor_at_depth4():
    """贪心 CART 需要深度 4 才能干净拟合 XOR：任何 0.5 处的单次划分
    收益为 0，贪心只能逐层剥出纯区域（实测 depth=2 只有 ~0.75，
    见 docs/实验记录.md）。这不是 bug，是贪心搜索的固有局限。"""
    X, y = make_xor_points()
    clf = trees.DecisionTreeClassifier(max_depth=4).fit(X, y)
    assert np.mean(clf.predict(X) == y) > 0.95


def test_shallow_tree_underfits():
    X, y = make_xor_points()
    clf = trees.DecisionTreeClassifier(max_depth=1).fit(X, y)
    # 深度 1 无法表达 XOR
    assert np.mean(clf.predict(X) == y) < 0.9


def test_predict_consistent_with_majority_labels():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    y = np.array([1, 1, 0, 0])
    clf = trees.DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert list(clf.predict(np.array([[0.05], [5.05]]))) == [1, 0]
