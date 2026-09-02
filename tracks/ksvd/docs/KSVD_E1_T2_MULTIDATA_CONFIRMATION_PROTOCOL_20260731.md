# KSVD E1-T2 跨数据种子五启动确认协议

> 日期：2026-07-31  
> 状态：**冻结 v0**  
> 前置结论：固定 data seed `20260731` 上，E1-T2 的 50 个 single-start 候选有 29 个严格恢复；五启动按最低训练重构选择的模拟成功率约为 99%。

## 1. 本轮唯一问题

> “五次随机初始化 + 最低训练重构误差选择”能否跨不同 E1-T2 synthetic 数据实现，稳定选出语义正确的结构字典？

上一轮只改变 learner seed，数据集固定。本轮只改变 data seed，用来排除五启动结论只是某一个训练样本实现上的偶然现象。

## 2. 固定配置

- 数据族：E1 固定 6 节点槽位、4 个边支持互不重叠的真实 atoms；
- 条件：T2，每个 patch 含 1 或 2 个 atoms；
- singleton probability：0.5；
- train/test：1000/300；
- data seeds：`20260731..20260740`，共 10 个连续、预先指定的 seeds；
- 每个 data seed 的 learner seeds：`0,1,2,3,4`；
- KSVD：`K=4`、`T=2`、`T_min=1`、25 iterations；
- 每个 data seed 内，只按最低 train relative reconstruction error 选择 1 个模型；
- train error 完全相同时，以较小 learner seed 打破平局；
- 测试集和真实字典指标只用于冻结评估，不参与选择。

本轮不改变 atom 设计、样本数、singleton probability、KSVD 迭代、初始化算法和 success threshold。

## 3. 每个数据种子的判定

先运行 oracle dictionary control，要求：

- test reconstruction `<= 1e-8`；
- support F1 `>= 0.999`；
- edge F1 和 exact patch recovery 均为 1。

然后运行 5 个 learner seeds。严格语义恢复沿用上一轮定义：

\[
\text{strict-success}
= [\text{mean atom cosine}\ge 0.99]
\land [\text{test reconstruction}\le 0.01].
\]

每个 data seed 记录：

1. 五个候选中是否至少存在一个 strict-success；
2. 最低 train reconstruction selector 是否选中 strict-success；
3. 若 group 中存在正确候选但 selector 选错，记为一次 selection miss；
4. selected model 的 atom cosine、support F1、edge F1、exact patch 和 test reconstruction。

## 4. 预注册确认门槛

五启动规则确认为 **PASS**，必须同时满足：

1. 10/10 oracle controls 通过；
2. 至少 9/10 data seeds 的五候选 group 中包含 strict-success；
3. 至少 9/10 data seeds 的 train-error selector 选中 strict-success；
4. selection miss 数为 0；
5. selected models 的 mean atom cosine `>= 0.99`；
6. selected models 的 mean test reconstruction `<= 0.01`。

结果解释：

- **PASS**：后续 E1 系列实验冻结使用 `5 restarts + minimum train reconstruction selector`；
- **FAIL_NO_SUCCESS_BASIN**：主要失败来自五个候选里没有正确解，需要先改进初始化或增加 restart；
- **FAIL_SELECTOR**：候选中已有正确解，但训练重构选错，需要重新检查 objective/可辨识性；
- **FAIL_ORACLE_OR_METRIC**：生成或 evaluator 出现问题，不得继续；
- 其他未通过情况记为 **FAIL_CONFIRMATION_GATE**。

## 5. 边界

通过本轮只允许冻结一个优化操作规则，不证明更困难的数据仍然可恢复，也不涉及：

- singleton 稀缺；
- atom 边支持重叠；
- 噪声；
- 节点置换；
- 随机游走和 patch sampling；
- 图级 readout 或真实数据。

## 6. 通过后的下一阶梯

若本轮通过，下一轮只改变一个生成因素：降低 singleton probability，测试 KSVD 是否能从组合观测而非大量直接 atom 观测中恢复结构基。

## 7. 复现入口

```bash
UV_INDEX_URL=https://pypi.org/simple \
uv run --with numpy python -m tracks.ksvd.code.test_e1_t2_multidata_confirmation

UV_INDEX_URL=https://pypi.org/simple \
uv run --with numpy python -m tracks.ksvd.code.run_from_scratch_e1_t2_multidata_confirmation
```

默认输出：

- `tracks/ksvd/results/from_scratch/e1_t2_multidata_confirmation_20260731.json`
- `tracks/ksvd/results/from_scratch/E1_T2_MULTIDATA_CONFIRMATION_20260731.md`

JSON 是数字唯一源；Markdown 只做可读汇总。
