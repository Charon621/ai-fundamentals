"""实验 1：线性回归——闭式解 vs 梯度下降。

输出：loss 曲线与拟合图（PNG）、指标（stdout 与 results.json）。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import linear  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(42)
X = rng.uniform(-3, 3, size=(120, 1))
y = 3.0 + 1.8 * X[:, 0] + rng.normal(0, 0.8, 120)

w_cf = linear.closed_form(X, y)
w_gd, hist = linear.grad_descent(X, y, lr=0.05, epochs=3000)

pred_cf = linear.predict(X, w_cf)
pred_gd = linear.predict(X, w_gd)
metrics = {
    "closed_form_w": w_cf.tolist(),
    "gd_w": w_gd.tolist(),
    "mse_closed_form": linear.mse(y, pred_cf),
    "mse_gd": linear.mse(y, pred_gd),
    "weight_gap_l2": float(np.linalg.norm(w_cf - w_gd)),
    "loss_first": hist[0],
    "loss_last": hist[-1],
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X, y, s=12, alpha=0.6, label="samples")
xs = np.linspace(-3, 3, 50)
axes[0].plot(xs, linear.predict(xs[:, None], w_cf), "r-", lw=2, label="closed form")
axes[0].plot(xs, linear.predict(xs[:, None], w_gd), "g--", lw=2, label="gradient descent")
axes[0].set(title="线性回归拟合", xlabel="x", ylabel="y")
axes[0].legend()
axes[1].plot(hist)
axes[1].set(title="梯度下降损失曲线", xlabel="epoch", ylabel="MSE")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "linear_regression.png"), dpi=120)

with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"\n图已保存: {os.path.join(HERE, 'linear_regression.png')}")
