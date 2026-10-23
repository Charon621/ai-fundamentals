"""实验 3：MLP 解决 XOR 与环形两分类。

XOR 是单层线性模型不可分、需要隐藏层的最小算例。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import mlp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# --- XOR ---
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_xor = np.array([0, 1, 1, 0])
net_xor = mlp.MLP([2, 8, 2], activation="tanh", seed=42)
hist_xor = net_xor.fit(X_xor, y_xor, lr=1.0, epochs=3000)
xor_ok = net_xor.predict(X_xor).tolist() == y_xor.tolist()

# --- 环形数据（内圈 vs 外圈）---
rng = np.random.default_rng(3)
n = 400
r_in = rng.uniform(0, 1.0, n // 2)
r_out = rng.uniform(1.6, 2.4, n - n // 2)
theta = rng.uniform(0, 2 * np.pi, n)
X_ring = np.column_stack([np.concatenate([r_in, r_out]) * np.cos(theta),
                          np.concatenate([r_in, r_out]) * np.sin(theta)])
y_ring = np.array([0] * (n // 2) + [1] * (n - n // 2))
net = mlp.MLP([2, 16, 16, 2], activation="tanh", seed=7)
hist = net.fit(X_ring, y_ring, lr=0.5, epochs=1500)
acc = float(np.mean(net.predict(X_ring) == y_ring))

xx, yy = np.meshgrid(np.linspace(-3, 3, 150), np.linspace(-3, 3, 150))
p = net.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(hist_xor)
axes[0].set(title=f"XOR 训练损失（预测{'正确' if xor_ok else '错误'}）", xlabel="epoch", ylabel="loss")
axes[1].plot(hist)
axes[1].set(title=f"环形两分类损失 (acc={acc:.3f})", xlabel="epoch", ylabel="loss")
axes[2].contourf(xx, yy, p, levels=20, cmap="coolwarm", alpha=0.8)
axes[2].scatter(X_ring[y_ring == 0, 0], X_ring[y_ring == 0, 1], c="b", s=8, label="内圈")
axes[2].scatter(X_ring[y_ring == 1, 0], X_ring[y_ring == 1, 1], c="r", s=8, label="外圈")
axes[2].set_title("MLP 决策概率")
axes[2].legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "mlp_xor_ring.png"), dpi=120)

metrics = {"xor_solved": xor_ok, "xor_final_loss": hist_xor[-1],
           "ring_acc": acc, "ring_final_loss": hist[-1],
           "ring_arch": [2, 16, 16, 2]}
with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"\n图已保存: {os.path.join(HERE, 'mlp_xor_ring.png')}")
