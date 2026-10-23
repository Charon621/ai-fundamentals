"""实验 5：卷积前向传播——垂直边缘检测与最大池化。

用 Prewitt 核对合成图像做边缘检测，再对响应图做 2x2 最大池化，
展示卷积输出的空间分辨率变化。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import conv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# 合成 32x32 图像：左暗右亮 + 圆形亮斑
n = 32
img = np.zeros((n, n))
img[:, 16:] = 0.6
yy, xx = np.mgrid[:n, :n]
blob = (xx - 22) ** 2 + (yy - 8) ** 2 < 25
img[blob] = 1.0
img += np.random.default_rng(0).normal(0, 0.02, (n, n))
X = img[None, None]

response = conv.conv2d(X, conv.edge_kernel())[0, 0]
pooled = conv.maxpool2d(np.abs(response)[None, None], size=2)[0, 0]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("输入图像（32×32）")
axes[1].imshow(response, cmap="bwr", vmin=-1.5, vmax=1.5)
axes[1].set_title("垂直边缘响应（3×3 卷积 → 30×30）")
axes[2].imshow(pooled, cmap="bwr", vmin=-1.5, vmax=1.5)
axes[2].set_title("|响应| 2×2 最大池化 → 15×15")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "conv_edge_detection.png"), dpi=120)

metrics = {
    "input_shape": [32, 32],
    "conv_output_shape": list(response.shape),
    "pooled_shape": list(pooled.shape),
    "max_abs_response": float(np.abs(response).max()),
    "flat_region_response_max": float(np.abs(conv.conv2d(
        np.full((1, 1, 8, 8), 0.5), conv.edge_kernel())).max()),
}
with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"\n图已保存: {os.path.join(HERE, 'conv_edge_detection.png')}")
