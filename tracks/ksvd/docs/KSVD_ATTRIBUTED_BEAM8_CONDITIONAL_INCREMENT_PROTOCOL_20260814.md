# Beam8 在全图统计条件后的冻结增量协议

> 日期：2026-08-14  
> 状态：BAG specificity controls 结果可见后冻结；本轮属于机制性探索，若通过仍需未见
> split 或外部数据集确认。

## 1. 研究问题

Matched controls 已显示 `GLOBAL_STATS` 在 split 3/4 × model seed 0/1/2 的 18 个单元中
全部超过 `BEAM8_FULL`。但两个单独 residual 的比较不能回答 Beam8 是否仍包含较弱的独立信息。
本轮改为顺序条件检验：

> 在 `GINE + GLOBAL_STATS` 已经训练并冻结之后，Beam8 是否还能提供稳定的额外分类增量？

## 2. 冻结网格与训练顺序

- 数据：TU Mutagenicity；不构造数据集；
- split seeds：3/4；model seeds：0/1/2；每个 cell 3 folds，共 18 units；
- GINE base epoch、划分、dictionary recipe 与 specificity controls 完全相同；
- 第一阶段只训练 rank-16、zero-init 的 `GLOBAL_STATS` residual；
- 冻结 GINE 与第一阶段 residual，将其 logits 作为新的固定 base logits；
- 第二阶段再训练一个 rank-16、zero-init residual；
- 两阶段各自只用 inner validation 选择 epoch，outer test 只用于最终评价；
- 第一阶段 GINE/`GLOBAL_STATS` 分数必须逐 unit 精确复现已有 specificity controls。

## 3. 第二阶段 matched controls

第二阶段所有输入都使用相同维度、相同 rank、优化器、loader seed 与 checkpoint 流程：

- `GLOBAL_PLUS_BEAM8_FULL`：deterministic INIT code + Beam8 patch node histogram；
- `GLOBAL_PLUS_BEAM8_CODE_ONLY`：只保留 deterministic INIT code；
- `GLOBAL_PLUS_BEAM8_HIST_ONLY`：只保留 Beam8 patch 节点属性均值；
- `GLOBAL_PLUS_BEAM8_RANDOM_DICTIONARY`：相同 patches、K24/T3 与 histogram，但使用从同一
  outer-train patch pool 抽取的随机归一化 atoms；
- `GLOBAL_STATS_STAGE1`：不训练第二阶段 head，作为条件基线。

所有 Beam8 rows 都先按 outer-train nodes 归一化，再做 BAG 广播；dictionary 与归一化统计均不
读取 outer-test labels 或特征分布。

## 4. 冻结 gate

只有全部满足，才认为当前 Beam8 BAG 在普通统计之外含有可转化的独立信息：

1. FULL−GLOBAL mean `>= +0.5pt`，至少 12/18 units 为正；
2. FULL−HIST_ONLY mean `>= +0.25pt`，至少 11/18 为正；
3. FULL−RANDOM_DICTIONARY mean `>= +0.25pt`，至少 11/18 为正；
4. FULL−GLOBAL 的 split3/4 means 均为正；
5. FULL−GLOBAL 至少 2/3 model-seed means 为正；
6. GINE 与第一阶段 `GLOBAL_STATS` parity 全部精确通过。

通过时判定为 `BEAM8_ADDS_INFORMATION_BEYOND_GLOBAL_STATS_EXPLORATORY`，随后才允许在未见
split 或新的 attributed TU dataset 上做确认。失败时判定为
`BEAM8_INCREMENT_BEYOND_GLOBAL_STATS_NOT_ESTABLISHED`，停止在 Mutagenicity 上扩大 BAG、
gate、cross-attention 或监督 dictionary；当前分类收益归入通用图统计校准，而非 Beam8-specific
贡献。
