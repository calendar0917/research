# DECISION — LCRC 稀疏局部模式身份保留 v1

Round: `lcrc-sparse-identity-v1`
Pre-registration: `PROTOCOL.md`（正式训练前冻结，阈值此后未改动）
Verdict produced by `code/si_analysis.py` → `analysis/decision.json`
Decision date: 2026-11-03（Test 一次性评分后）

---

## 判定

# `NO_PERFORMANCE_SIGNAL`

**理由（预注册判据，逐字对应 `PROTOCOL.md §8`）：**
`WIDE_TRUE − A0` 的三 seed 平均 ROC-AUC 在官方 Validation **和** Test 上都不大于零
（Validation **−0.004436**，Test **−0.000752**），因此命中
`NO_PERFORMANCE_SIGNAL` 的定义句「在 Validation 或 Test 上不大于零」。

---

## 预注册条件逐条核验（阈值未改动）

`PREDICTIVE_IDENTITY_SIGNAL` 要求 1–5 全部满足：

| # | 条件 | 阈值 | 实测 | 通过 |
|---|---|---|---|---|
| 1a | `WIDE_TRUE − A0`，Validation，3 seed 平均 ROC-AUC | ≥ +0.010 | −0.004436 | ✗ |
| 1b | `WIDE_TRUE − A0`，Test，3 seed 平均 ROC-AUC | ≥ +0.010 | −0.000752 | ✗ |
| 2 | 在 Validation **与** Test 上都优于配对 A0 的 seed 数 | ≥ 2 | 0（Valid 0/3，Test 2/3，交集 0） | ✗ |
| 3a | `WIDE_TRUE − WIDE_NULL`，Validation | ≥ +0.005 | −0.002169 | ✗ |
| 3b | `WIDE_TRUE − WIDE_NULL`，Test | ≥ +0.005 | +0.006190 | ✓ |
| 4 | `WIDE_TRUE` 平均 Test AP 不低于 A0 | ≥ A0 | 0.293951 ≥ 0.285930 | ✓ |
| 5 | 所有正确性门禁通过 | 全 PASS | 14/14 PASS | ✓ |

不满足条件 1a、1b、2、3a → `PREDICTIVE_IDENTITY_SIGNAL` 不成立。

按协议给出的降级顺序判定：命中 `NO_PERFORMANCE_SIGNAL` 后，不再考察
`NO_IDENTITY_SPECIFIC_SIGNAL`（该判定要求先存在正向预测增量）与
`MIXED_OR_WEAK_SIGNAL`（同样要求部分正向证据）；`INCOMPLETE` 不适用
（12 个正式训练、SPARSE_ONLY、冻结、一次性 Test、全部门禁均已完成，无失败、无中断）。

---

## 支撑判定的数字

三 seed 平均 ROC-AUC（官方 OGB scaffold split）：

| arm | Validation | Test |
|---|---|---|
| A0 | **0.834969** ± 0.003156 | **0.771597** ± 0.005534 |
| WIDE_TRUE | 0.830532 ± 0.001790 | 0.770845 ± 0.007958 |
| WIDE_NULL | 0.832702 ± 0.003663 | 0.764655 ± 0.005031 |
| FOLD64 | 0.833031 ± 0.004124 | 0.765002 ± 0.006816 |
| SPARSE_ONLY | 0.663770 | 0.673443 |

配对差值（3 seed 平均，含 95% 配对 bootstrap 区间；Test 为
`EXPOSED_TEST_EXPLORATORY`）：

| 对照 | Validation | Test |
|---|---|---|
| TRUE − A0 | −0.004436 [−0.016671, +0.006648] | −0.000752 [−0.011238, +0.010982] |
| TRUE − NULL | −0.002169 [−0.011544, +0.006114] | +0.006190 [−0.002842, +0.015657] |
| TRUE − FOLD64 | −0.002499 [−0.007515, +0.001912] | +0.005843 [−0.000806, +0.012484] |
| NULL − A0 | −0.002267 [−0.009139, +0.004837] | −0.006942 [−0.016456, +0.002605] |
| FOLD64 − A0 | −0.001938 [−0.012639, +0.008065] | −0.006595 [−0.016808, +0.004253] |

平均 Test AP：A0 0.285930 ± 0.034463，WIDE_TRUE 0.293951 ± 0.007894（+0.008021，1/3 seed 为正）。

---

## 这个判定意味着什么

1. 在当前配置下（A0 主干冻结结构不变、512 维冻结 S2 模式、零初始化无 bias 线性残差、
   R3 配方 100 epoch、seed 0/1/2），**保留 S2 模式的离散身份没有带来可重复的预测增量**。
2. 判定依据是预注册的三 seed 平均 ROC-AUC 差值，且两个 split 的均值都为负；
   这不是「区间包含 0」的弱证据，而是点估计落在错误方向。
3. 唯一与「有信号」同向的观察（Test AP 均值 +0.008、Test `TRUE − NULL` +0.0062）
   未达到预注册的多条件联合要求，且 `TRUE − NULL` 在 Validation 为负，
   因此不构成身份特异性证据。

## 这个判定不意味着什么

1. **不**意味着 512 个 S2 模式没有信息：`SPARSE_ONLY` 在 Validation/Test 上达到
   0.663770 / 0.673443 ROC-AUC，明显高于随机；WIDE_TRUE 的 512 个分支权重全部非零
   （L2 范数 4.6–5.3，平均 |logit 贡献| ≈ 0.17），说明分支确实被训练并起作用。
2. **不**意味着压缩是唯一瓶颈，也不意味着「保留身份」在其他读取方式（非线性、交互、
   更大容量、其他正则）下也无效——本轮只检验了这一个预注册读取方式。
3. **不**意味着 A0 已经充分利用了这些模式（未做移除式消融，协议未包含）。
4. **不**构成对全新盲测数据的确认性统计证据：Test 在历史轮次已暴露，本轮 Test 结论
   一律为 `EXPOSED_TEST_EXPLORATORY`；且 3 seed × 4,113 分子使区间半宽约 ±0.011，
   无法排除 ±0.01 量级的真实效应。
5. **不**对远距离/非局部表达性作任何断言（18 环分子对在 512 维表示下仍然完全碰撞，
   Jaccard = 1.0；本轮不研究该问题）。

---

## 执行事实（判定所依赖的前提）

- 12 次正式神经训练（4 arm × seed 0/1/2）+ 1 次 CPU `SPARSE_ONLY`，全部 100 epoch 完成，
  无失败、无重跑、无方案修改。
- 14 项正确性门禁全部 PASS（`analysis/correctness_gates.json`）；
  零初始化等价误差 0.0（≤1e-6），重载复现误差 ≤ 5.72e-6，NULL 边际严格守恒。
- 冻结在 Test 之前：`analysis/checkpoint_manifest.json`（自哈希
  `604e25b028597b52…a37fa4`，`frozen_before_test_access=true`），
  随后一次性评分 13 个模型（`analysis/test_access_ledger.json`：13 条事件，冻结前 0 条）。
- 官方 OGB Evaluator 与 sklearn 在 13 个模型上完全一致（max|Δ| = 0.0）。
- GPU 用量 **3.208851 A100 GPU·h**（正式 3.186494 + 预检 0.022356），
  低于软预算 4.0 与硬上限 5.0。
- A0 为同轮重训并与历史 PTB A0 逐 seed 对齐（best epoch 53/32/29 完全一致，
  Valid ROC-AUC 差 ≤ 7.96e-5），因此比较是配对的，不是与旧 checkpoint 比较。

## 后续

按任务边界，本 agent 不提出下一代架构或下一轮实验方案。
可供下一轮独立设计参考的**事实**（不是建议）：`SPARSE_ONLY` 与分支权重显示模式有信息；
本轮唯一的读取方式（A0 + 零初始化线性残差）在两个 split 上都未产生正增量，
而在 Test 上对结构被破坏的 `WIDE_NULL` 有 +0.0062 的差距、在 Validation 上为负。
