# AI Fundamentals · numpy 从零实现的基础算法实验室

不调库，把人工智能专业的地基算法亲手写一遍：线性模型、决策树、多层感知机、注意力机制、卷积——每个模块都配**数值梯度检验**和**手算对照用例**，每个实验都留下**真实运行的数据与图**。

> 属于 [Charon621](https://github.com/Charon621) 的基础层练习；应用层项目见 [agentlab](https://github.com/Charon621/agentlab)（多智能体）与 [LLM Benchmark Studio](https://github.com/Charon621/llm-benchmark-studio)（评测）。

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests -q     # 30 项测试，约 0.4 秒
python labs/01_linear_regression/lab.py   # 任一实验，CPU 几秒内出图
```

## 模块与验证方式

| 模块 | 实现内容 | 正确性验证 |
|---|---|---|
| `src/linear.py` | 正规方程、岭回归（截距不正则）、批量梯度下降 | 闭式解与梯度下降收敛到同一解（权重差 < 1e-14） |
| `src/logistic.py` | 数值稳定 sigmoid、交叉熵 + L2、二分类 | 解析梯度 vs 中心差分（误差 < 1e-6）；线性可分数据 100% 拟合 |
| `src/trees.py` | CART 基尼不纯度、穷举最优划分、递归二叉树 | 手工算例的划分阈值与增益；XOR 拟合与欠拟合对照 |
| `src/mlp.py` | 前向、反向传播、softmax+CE、He/Xavier 初始化 | tanh/relu 两种激活的数值梯度检验；XOR 最小算例 |
| `src/attention.py` | scaled dot-product、因果掩码、多头切分 | 掩码上三角严格为 0；篡改未来位置输出不变；单头与直接公式等价 |
| `src/conv.py` | im2col 卷积、最大池化、六重循环参考实现 | im2col 与参考实现逐元素一致（atol=1e-10）；卷积核数值梯度检验 |

梯度检验是本仓库的正确性金标准：**每个含梯度的模块都必须先通过中心差分对照，再看训练效果**。

## 实验（全部真实运行，结果见 `labs/*/results.json` 与 `docs/实验记录.md`）

| 实验 | 问题 | 关键结果（2026-10-24 实测） |
|---|---|---|
| [01 线性回归](labs/01_linear_regression/) | 闭式解 vs 梯度下降是否一致 | 权重 L2 差 **1.8e-15**（机器精度），MSE 0.650 |
| [02 分类边界](labs/02_classification/) | 线性模型与树在双月牙上的差异 | 逻辑回归 **86.7%** vs 决策树 **91.3%** |
| [03 MLP](labs/03_mlp_xor/) | XOR 与环形两分类 | XOR loss 收敛到 **2.2e-4**；环形分类 **100%** |
| [04 注意力](labs/04_attention/) | 因果掩码下注意力关注什么 | 上三角权重全 0；代词"它"最强关注的键是"小猫" |
| [05 卷积](labs/05_conv/) | 边缘检测与池化的形状变化 | 32×32 → 卷积 30×30 → 池化 15×15；平坦区响应恰为 0 |

## 验证边界（诚实声明）

- 本仓库验证的是**算法机制**：数学正确性（梯度、收敛、划分、掩码语义）。
- **不能**由此得出：与 sklearn/torch 生产实现的性能对比、大规模数据上的泛化结论、任何真实模型能力。性能对照需要框架级工程与标准数据集协议，超出本仓库范围。
- 贪心 CART 在均匀分布 XOR 上会卡在局部最优（深度 2 仅 ~64% 准确率，深度 4 才 99%+）——这不是 bug，是贪心搜索的固有性质，详见[实验记录](docs/实验记录.md#贪心-cart-与-xor)。
- 实验固定随机种子，同环境重跑结果一致；GitHub Actions 在 Python 3.11 / 3.12 上跑全部测试。

## 目录结构

```text
src/     六个基础模块（仅依赖 numpy）
tests/   pytest 测试，含数值梯度检验
labs/    五个可复现实验（图 + 指标 JSON）
docs/    实验记录（真实运行数据）、验证边界说明
```

## License

MIT
