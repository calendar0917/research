# KSVD E1-T2 初始化与多启动审计协议

> 日期：2026-07-31  
> 状态：**冻结 v0**  
> 前置结果：E0-T1、E0-T2、E1-T1 通过；E1-T2 在 10 个 single-start seeds 中严格恢复 5 次。

## 1. 本轮唯一问题

> E1-T2 的失败主要是“正确解存在但 single-start 容易落入局部最优”，还是“训练重构目标无法从多个候选中识别语义正确的字典”？

本轮只审计初始化和多启动选择。数据、结构基、稀疏度、OMP、KSVD 更新和测试指标全部保持不变。

## 2. 固定不变的实验对象

- 数据：E1 固定 6 节点槽位、4 个边支持互不重叠的真实结构基；
- 条件：T2，每个 patch 含 1 或 2 个结构基，singleton probability 为 0.5；
- data seed：`20260731`；
- train/test：1000/300；
- KSVD：`K=4`、`T=2`、`T_min=1`、25 iterations；
- learner initialization：现有 `ksvd.py` 的 random training columns + `1e-3` Gaussian jitter；
- 候选池：learner seeds `0..49`；
- 测试集只做冻结评估，**不得用于模型选择**。

本轮禁止改变 atom 设计、数据分布、迭代次数、初始化算法或 success threshold。否则属于下一轮实验，不得混入本轮结论。

## 3. 单次初始化记录

每个 learner seed 记录两类信息。

### 3.1 初始化前状态

按照现有 KSVD 的随机数调用精确重建初始字典，并记录：

- 被选中的训练列索引及其真实 atom support；
- 4 个初始列中 singleton 列数量；
- singleton 覆盖到的不同真实 atom 数；
- 所有初始列联合覆盖到的真实 atom 数；
- 初始字典对真实字典的 matched mean atom cosine。

这些指标只用于解释 basin，不参与多启动选择。

### 3.2 训练后状态

记录：

- train relative reconstruction error；
- test relative reconstruction error；
- matched mean/minimum atom cosine；
- code support precision/recall/F1；
- atom edge-support F1；
- reconstructed edge F1；
- exact patch recovery。

严格语义恢复定义冻结为：

\[
\text{strict-success}
= [\text{mean atom cosine}\ge 0.99]
\land [\text{test reconstruction}\le 0.01].
\]

## 4. 多启动模拟

不重复训练。先得到 50 个候选，再用固定 audit RNG 从候选池中无放回抽取 restart group。

- restart budgets：`R = 1, 2, 3, 5, 10`；
- 每个 budget：5000 个随机 group；
- audit seed：`20260731`；
- group 内唯一合法 selector：**最低 train relative reconstruction error**；
- train error 完全相同时以较小 learner seed 打破平局；
- 禁止用 test reconstruction、atom cosine、support F1 或 edge 指标选择。

每个 budget 报告：

1. group 中至少存在一个 strict-success 候选的概率；
2. train-error selector 最终选中 strict-success 的概率；
3. selection efficiency：`P(selected success) / P(group contains success)`；
4. 被选候选的 atom cosine、support F1、test reconstruction 均值；
5. atom-cosine regret：group 内最大 atom cosine 减去被选候选 atom cosine；
6. strict-success containment 的超几何精确概率，作为 Monte Carlo 校验。

## 5. 目标可辨识性诊断

全候选池额外报告：

- train reconstruction 与 atom cosine 的 Pearson / Spearman 相关；
- train reconstruction 与 support F1 的 Pearson / Spearman 相关；
- initial atom cosine 与 final atom cosine 的相关；
- train error 最低的 strict-success 与 non-success 候选；
- 全局最低 train error 候选是否 strict-success；
- strict-success 和 non-success 的 train error 分布范围。

注意：train error 越低越好而 atom cosine 越高越好，因此“目标一致”通常表现为负相关。

## 6. 预注册判断规则

以 `R=5` 为主判断点：

### A. 可管理的局部最优问题

同时满足：

- group contains strict success `>= 0.90`；
- selector selects strict success `>= 0.90`；
- selection efficiency `>= 0.90`。

解释：正确 basin 并不罕见，而且纯训练目标能够识别它；后续可把多启动作为普通优化策略。

### B. 选择目标 / 可辨识性问题

满足：

- group contains strict success `>= 0.90`；
- 但 selector selects strict success `< 0.80`，或 selection efficiency `< 0.80`。

解释：候选池里经常已有语义正确解，但最低训练重构仍会选错。此时不能只靠增加 restart，应先研究 objective、约束或 atom 可辨识性。

### C. 正确 basin 稀少

- group contains strict success `< 0.90`。

解释：当前初始化下正确解仍不够容易出现。下一步先审计初始化策略，不讨论真实图任务。

0.80–0.90 的中间区间记为 **INCONCLUSIVE**，扩大候选池或 trial 数后再判断。

## 7. 本轮边界

本实验通过也只说明：在 E1-T2 的 oracle synthetic setting 下，多启动是否能解决 ordinary KSVD 的优化稳定性。

它不回答：

- 随机游走能否采到结构；
- 节点置换如何处理；
- atom 是否为合法子图；
- atom occurrence/incidence 如何形成图表征；
- CIN、MolHIV 或其他真实数据上的分类效果。

## 8. 复现入口

```bash
UV_INDEX_URL=https://pypi.org/simple \
uv run --with numpy python -m tracks.ksvd.code.test_e1_t2_initialization_audit

UV_INDEX_URL=https://pypi.org/simple \
uv run --with numpy python -m tracks.ksvd.code.run_from_scratch_e1_t2_initialization_audit
```

默认输出：

- `tracks/ksvd/results/from_scratch/e1_t2_initialization_audit_20260731.json`
- `tracks/ksvd/results/from_scratch/E1_T2_INITIALIZATION_AUDIT_20260731.md`

JSON 是数字唯一源；Markdown 只做可读汇总。
