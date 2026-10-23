"""注意力机制：scaled dot-product、因果掩码、多头注意力。

覆盖 Transformer 注意力的核心数学；位置编码与前馈层不在本模块范围。
"""
import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(seq_len):
    """下三角布尔掩码：位置 i 可见 j ≤ i。"""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def attention(Q, K, V, mask=None):
    """softmax(QKᵀ/√d_k) V。

    mask: 布尔数组，True 表示允许 attend。
    返回 (输出, 注意力权重矩阵)。
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e18)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """多头注意力。

    X: (T, d_model)；W_q/W_k/W_v: (d_model, d_model)；W_o: (d_model, d_model)。
    头维度切分按 d_head = d_model // num_heads。
    """
    T, d_model = X.shape
    assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
    d_head = d_model // num_heads

    def split(M):
        # (T, d_model) -> (num_heads, T, d_head)
        return M.reshape(T, num_heads, d_head).transpose(1, 0, 2)

    Q = split(X @ W_q)
    K = split(X @ W_k)
    V = split(X @ W_v)

    if mask is not None:
        head_mask = mask[None, :, :]  # 广播到所有头
    else:
        head_mask = None

    out, weights = attention(Q, K, V, mask=head_mask)
    # (num_heads, T, d_head) -> (T, d_model)
    concat = out.transpose(1, 0, 2).reshape(T, d_model)
    return concat @ W_o, weights
