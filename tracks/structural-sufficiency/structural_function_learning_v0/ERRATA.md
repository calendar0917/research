# ERRATA — COUNT-ALL 局部贡献统计的重复标准化

本文件修正 `interpretable_structural_rule_discovery_v1`（ISRD-v1，冻结 commit
`16d7a002`）中 `run_phaseC.local_contributions()` 的贡献量计算错误。ISRD-v1 的
历史文件只读保留、不覆盖；本 ERRATA 明确哪些数字被本轮替代、哪些结论不受影响。

## E1 — 贡献再次除以标准差

**位置**：`interpretable_structural_rule_discovery_v1/code/run_phaseC.py`,
`local_contributions()`。

**错误**：代码先由标准化系数得到原始单位系数

```
w_raw = w_std / sd
```

随后又用

```
contrib = (x - mu) / sd * w_raw
```

计算贡献，等价于 `w_std * (x - mu) / sd**2`：标准化设计的中心化项被**第二次**
除以 `sd`。因此贡献大小、绝对值总量、top-k 占比和结构排名都被错误缩放，且各列
被缩放的比例不同（`1/sd`），所以不只是整体常数倍。

**未受影响**：

* 模型本身与 AUC/AP：预测闭合使用的是正确的原始单位参数化
  `z = b_raw + x @ w_raw`（历史 `closure_raw_units = 1.07e-14`）；错误只发生在
  报告贡献的数组上。
* `COUNT-ALL` 的系数、截距、标准化参数、逐分子 logit 与所有指标。
* 历史模型没有被重训。

## E2 — 重算方法与闭合核验

读取冻结 `analysis/phaseC.json` 的 `count_all` 参数与冻结输入，逐分子重算两种
精确分解（不重训、不重新选择 C）：

1. **零结构基准**（本轮主要定义）

   ```
   contrib_s = w_raw_s * x_s
   z = b_raw + sum_size w_raw*x + sum_s contrib_s
   ```

2. **均值中心化基准**

   ```
   contrib_s = w_std_s * (x_s - mu_s) / sd_s
   z = b_std + sum_size ((x-mu)/sd)*w_std + sum_s contrib_s
   ```

结果（`analysis/phase0_contributions.json`）：

| 基准 | max abs 重建误差 |
|---|---:|
| 零结构基准 | **1.42e-14** |
| 均值中心化基准 | **1.95e-14** |
| 历史错误公式（对照） | **8.40e+03** |

因此两种正确定义都能重建原始 logit；历史公式不能。

## E3 — 被替代的集中度与绝对贡献总量

ISRD-v1 REPORT §5 的原值（Test 划分、逐分子平均）与修正后值：

| 统计量 | 旧（重复除以 sd） | 新（零结构基准） | 说明 |
|---|---:|---:|---|
| top-5 绝对贡献占比 | 0.2915 | **0.3505** | 旧值低估集中度 |
| top-10 绝对贡献占比 | 0.3639 | **0.5151** | |
| top-20 绝对贡献占比 | 0.4268 | **0.7026** | |
| 全部结构项 \|贡献\| 总量 | 2200.29 | **33.11** | 旧值约高 66 倍 |
| 前 20 项之外 \|贡献\| 剩余 | 1177.67 | **10.85** | |
| 未展示项的**有符号**剩余 | — | **-0.86** | 本轮新增 |
| 平均激活结构数 | 89.0 | 89.0 | 未受影响 |

Valid 与 Train 的同向数值见 `analysis/phase0_contributions.json` 的
`old_buggy` / `zero_baseline` 两个块。

新值还给出按 COARSE 父结构汇总 FULL 子型后的组级集中度（组贡献 = 组内实际模型项
之和，不增删项；全部 3 652 个 FULL 都有保留的 COARSE 父结构，因此共 754 组，
无孤儿 FULL）：

| 组级统计（Test 平均） | 值 |
|---|---:|
| top-5 家族绝对贡献占比 | 0.4960 |
| top-10 家族绝对贡献占比 | 0.7018 |
| top-20 家族绝对贡献占比 | 0.8912 |
| 家族 \|贡献\| 总量 | 24.61 |
| 平均激活家族数 | 36.9 |

## E4 — 结论层面的影响范围

**被替代**：ISRD-v1 关于“局部解释不集中”的定量陈述——原文的
“top-5/10/20 = 29.1% / 36.4% / 42.7%、其余 1177.7 个 logit 单位”全部作废，
以本 ERRATA 的 35.0% / 51.5% / 70.3%、剩余 10.85 为准。修正后结构项集中度明显
更高，家族级 top-20 约 89%，因此“可重建但不稀疏”仍成立（平均 89 个结构项激活），
但“前 20 项只占 42.7%”这一支持该判断的数字不再成立。

**不被替代**：所有模型指标、AUC/AP、预测闭合、结构排名之外的模型选择、
`COUNT-ALL` 与 `XGB-ALL` 的对比、以及本轮所有新模型结果。

**未检查**：仓库中其他轮次若复制了同一公式，不在本轮范围内；本轮只核验并使用
ISRD-v1 `COUNT-ALL` 的冻结参数。

## E5 — 可复现命令

```
uv run python tracks/structural-sufficiency/structural_function_learning_v0/code/phase0_contributions.py
uv run pytest -q tracks/structural-sufficiency/structural_function_learning_v0/tests -m "not slow"
```

对应核验：`tests/test_real_models.py::test_hiv_input_identity_against_historical_predictions`
与 `test_phaseE_tree_and_nn_closures`；数值记录在
`analysis/phase0_contributions.json` 和 `analysis/provenance.json`。

## E6 — 实现偏差：加性神经的偏置初始化

PROTOCOL §6 将 `H-NN-ADD` / `Z-NN-ADD` 的初始化写为 “a、v、b 为确定性
N(0,0.01) 初始化”。实际代码中 `a`、`v` 由 NumPy `default_rng(0)` 生成
N(0,0.01)（确定性，但不是 torch RNG），而 `bias` 初始化为 **0**，不是
N(0,0.01)。这是冻结协议与实现之间的偏差，记录于此。

影响评估：偏置只影响初始输出平移，不改变 `f_j(0)=0`、加性结构或闭包；统一
50 epochs、AdamW 训练下该常数初始化会被快速吸收。没有因此重训或改选模型。
所有报告数字来自实际代码；不能把 `bias` 初始化描述为已按协议执行。
