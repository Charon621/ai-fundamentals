"""实验 4：注意力可视化——因果掩码与注意力热图。

构造"最后一个词是代词"的最小场景，观察注意力把权重放在哪些词上。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import attention  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

tokens = ["小猫", "追", "毛线球", "，", "它", "很", "开心"]
T, d = len(tokens), 16

# 手工设定"语义方向"：代词「它」的查询与名词有较强对齐，
# 其余 token 用随机小值——用受控夹具说明机制，不声称模拟真实语言模型。
rng = np.random.default_rng(0)
X = rng.normal(0, 0.3, size=(T, d))
noun_idx = [0, 2]
pron_idx = 4
X[pron_idx] = 0.0
X[noun_idx] = 1.0                      # 名词共享一个方向
X[pron_idx, :8] = 1.0                  # 代词的查询前半维与名词同向

Wq = Wk = Wv = Wo = np.eye(d) * 0.5

mask = attention.causal_mask(T)
out, w = attention.multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=1, mask=mask)

fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(w[0], cmap="viridis")
ax.set_xticks(range(T), tokens)
ax.set_yticks(range(T), tokens)
ax.set_title("因果掩码下的注意力权重（行=查询，列=键）")
ax.set_xlabel("被关注的 token（键）")
ax.set_ylabel("当前 token（查询）")
for i in range(T):
    for j in range(T):
        ax.text(j, i, f"{w[0, i, j]:.2f}", ha="center", va="center",
                color="white" if w[0, i, j] < 0.5 else "black", fontsize=8)
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "attention_heatmap.png"), dpi=120)

checks = {
    "causal_upper_triangle_zero": bool(np.all(w[0][np.triu_indices(T, 1)] == 0.0)),
    "rows_sum_to_one": bool(np.allclose(w[0].sum(axis=1), 1.0)),
    "pronoun_top1_key": tokens[int(np.argmax(w[0, pron_idx]))],
    "pronoun_weight_on_nouns": float(w[0, pron_idx, noun_idx].sum()),
}
with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(checks, f, ensure_ascii=False, indent=2)
print(json.dumps(checks, ensure_ascii=False, indent=2))
print(f"\n图已保存: {os.path.join(HERE, 'attention_heatmap.png')}")
