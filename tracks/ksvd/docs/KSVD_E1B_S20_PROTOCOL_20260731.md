# KSVD E1B-S20：降低 singleton frequency 的结构基恢复协议

> 日期：2026-07-31  
> 状态：**冻结 v0**  
> 前置结论：E1-T2 在 singleton probability 0.5 时，`5 restarts + minimum train reconstruction selector` 跨 10 个 data seeds 全部严格恢复。

## 1. 本轮唯一问题

> 当直接展示单个 atom 的 patch 从约 50% 降低到约 20% 时，KSVD 是否仍能从以双 atom 组合为主的观测中恢复真实结构基？

本轮只改变 singleton probability：

```text
0.5 → 0.2
```

其余数据结构、训练规模、模型参数、restart 数、selector 和成功标准全部冻结。

## 2. 数据生成

沿用 E1 固定槽位图结构基：

- 6 个有语义的节点槽位；
- 邻接矩阵上三角 15 维；
- 4 个真实 atoms；
- atom 边支持互不重叠；
- patch 由 1 或 2 个 atoms 的 binary union 生成；
- 无节点置换、无噪声、无边支持重叠。

唯一变化：

- 普通随机样本的 singleton probability 为 `0.2`；
- 其余约 80% 样本为双 atom 组合；
- 训练集和测试集开头仍分别放置每个 atom 的一个 guaranteed singleton。

保留 guaranteed singleton 是为了只研究“直接观测频率下降”，暂不混入“某些 atoms 从未单独出现”的覆盖问题。

## 3. 冻结运行配置

- data seeds：`20260731..20260740`，共 10 个；
- train/test：1000/300；
- 每个 data seed 的 learner seeds：`0,1,2,3,4`；
- KSVD：`K=4`、`T=2`、`T_min=1`、25 iterations；
- 每组 5 个候选中，仅按最低 train relative reconstruction error 选择；
- train error 相同时选择较小 learner seed；
- test 和真实字典指标不参与选择。

每个数据集必须记录实际 singleton count/rate，确认生成器确实产生接近 20% 的单 atom patch。

## 4. Oracle control

每个 data seed 的真实字典必须满足：

- test reconstruction `<= 1e-8`；
- support F1 `>= 0.999`；
- edge F1 = 1；
- exact patch recovery = 1。

Oracle 未通过时停止解释 learner 结果。

## 5. 严格恢复标准

沿用前两轮定义：

\[
\text{strict-success}
= [\text{mean atom cosine}\ge 0.99]
\land [\text{test reconstruction}\le 0.01].
\]

同时继续报告：

- minimum atom cosine；
- code support F1；
- atom edge-support F1；
- edge F1；
- exact patch recovery。

## 6. 预注册 Gate

E1B-S20 记为 **PASS**，必须同时满足：

1. 10/10 oracle controls 通过；
2. 至少 9/10 五启动 groups 包含 strict-success；
3. 至少 9/10 train-error selectors 选中 strict-success；
4. selection miss 数为 0；
5. selected mean atom cosine `>= 0.99`；
6. selected mean test reconstruction `<= 0.01`；
7. 所有训练集实际 singleton rate 均位于 `[0.15, 0.25]`。

结果解释：

- **PASS**：继续只把 singleton probability 降到 0.05；
- **FAIL_NO_SUCCESS_BASIN**：五启动下正确 basin 不够常见，先研究 restart/初始化；
- **FAIL_SELECTOR**：已有正确候选但 train reconstruction 选错，研究 objective；
- **FAIL_ORACLE_OR_DATA**：oracle 或数据频率检查失败；
- 其他情况为 **FAIL_RECOVERY_GATE**。

## 7. 本轮不回答什么

本轮仍不涉及：

- 完全没有 singleton；
- 不保证每个 atom 被单独观察；
- atom 边支持重叠；
- edge noise；
- 节点置换；
- 随机游走、patch sampling；
- patch 间关联、图级 readout 或真实数据。

## 8. 复现入口

```bash
UV_INDEX_URL=https://pypi.org/simple \
uv run --with numpy python -m tracks.ksvd.code.test_e1b_singleton_frequency

UV_INDEX_URL=https://pypi.org/simple \
uv run --with numpy python -m tracks.ksvd.code.run_from_scratch_e1b_s20
```

默认输出：

- `tracks/ksvd/results/from_scratch/e1b_s20_20260731.json`
- `tracks/ksvd/results/from_scratch/E1B_S20_20260731.md`

JSON 是数字唯一源；Markdown 只做可读汇总。
