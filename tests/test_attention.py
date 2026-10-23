import numpy as np
import pytest

import attention


def test_softmax_rows_sum_to_one():
    z = np.random.default_rng(0).normal(size=(4, 7))
    s = attention.softmax(z)
    assert np.allclose(s.sum(axis=-1), 1.0)
    assert np.all(s >= 0)


def test_softmax_numerically_stable():
    z = np.array([[10000.0, 10001.0, 9999.0]])
    s = attention.softmax(z)
    assert np.all(np.isfinite(s))
    assert np.allclose(s.sum(), 1.0)


def test_causal_mask_is_lower_triangular():
    m = attention.causal_mask(4)
    expected = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=bool)
    assert np.array_equal(m, expected)


def test_causal_attention_blocks_future():
    """位置 0 的输出必须与位置 >0 的内容无关。"""
    rng = np.random.default_rng(1)
    T, d = 5, 3
    Q = rng.normal(size=(T, d))
    K = rng.normal(size=(T, d))
    V = rng.normal(size=(T, d))
    mask = attention.causal_mask(T)
    out1, w1 = attention.attention(Q, K, V, mask=mask)

    V2 = V.copy()
    V2[1:] = 999.0  # 篡改位置 1 之后（对查询 0 是未来）
    out2, _ = attention.attention(Q, K, V2, mask=mask)
    assert np.allclose(out1[0], out2[0])              # 查询 0 输出不变
    assert np.all(w1[0, 1:] == 0.0)                   # 其未来权重为 0
    assert np.all(w1[np.triu_indices(T, 1)] == 0.0)   # 整体上三角为 0


def test_attention_uniform_when_scores_equal():
    Q = np.zeros((2, 4))
    K = np.ones((3, 4))
    V = np.arange(12, dtype=float).reshape(3, 4)
    out, w = attention.attention(Q, K, V)
    assert np.allclose(w, 1.0 / 3.0)
    assert np.allclose(out, V.mean(axis=0))


def test_multi_head_shape_and_single_head_equivalence():
    rng = np.random.default_rng(2)
    T, d_model = 6, 8
    X = rng.normal(size=(T, d_model))
    Wq, Wk, Wv, Wo = [rng.normal(size=(d_model, d_model)) * 0.3 for _ in range(4)]

    out_2h, w_2h = attention.multi_head_attention(X, Wq, Wk, Wv, Wo, 2)
    assert out_2h.shape == (T, d_model)
    assert w_2h.shape == (2, T, T)

    # 单头时与直接 attention 结果一致
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    ref, ref_w = attention.attention(Q, K, V)
    out_1h, _ = attention.multi_head_attention(X, Wq, Wk, Wv, Wo, 1)
    assert np.allclose(out_1h, (ref @ Wo))


def test_multi_head_heads_differ():
    """不同头应产生不同的注意力模式（切分确实生效）。"""
    rng = np.random.default_rng(3)
    T, d_model, heads = 5, 8, 2
    X = rng.normal(size=(T, d_model))
    Wq, Wk, Wv, Wo = [rng.normal(size=(d_model, d_model)) for _ in range(4)]
    _, w = attention.multi_head_attention(X, Wq, Wk, Wv, Wo, heads)
    assert not np.allclose(w[0], w[1])
