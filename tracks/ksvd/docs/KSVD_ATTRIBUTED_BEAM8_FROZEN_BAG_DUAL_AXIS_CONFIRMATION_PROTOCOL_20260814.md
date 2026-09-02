# Attributed Beam8 graph-conditioned frozen BAG dual-axis confirmation

> 日期：2026-08-14  
> 状态：localized unseen-split confirmation 失败后冻结；新增 model-seed 结果不可见。

## 1. 研究问题

未见 split seeds 3/4 上，TRUE localized residual 未通过 localization/binding gate，但 BAG
residual 达到 75.56%，高于 TRUE 的 74.66% 与 GINE 的 73.68%。本轮不再测试局部对应，只回答：

> graph-specific Beam8 summary 对 frozen GINE 的低容量条件校准，能否同时跨数据 split 与
> neural model seed 保持收益？

## 2. 固定设计

- 数据：TU Mutagenicity；
- split seeds：3/4，各 3 outer folds；
- model seeds：0/1/2；
- model seed 0 复用已经冻结的 split3/4 结果；新增 model seeds 1/2；
- attributed Beam8 INIT dictionary：每 fold 只在 outer-train 上拟合，K24/T3；
- 先构造 orbit-safe node incidence 并按 outer-train normalization，再对每图取 node mean，
  广播成 BAG field；
- GINE 参数与 BatchNorm 完全冻结；
- residual 仍为 rank-16、zero-init conditional adapter；
- strict inner validation 分别选择 base epoch 与 BAG residual epoch，outer-test 只评估一次；
- 只比较 `GINE_FROZEN` 与 `BAG_RESIDUAL`，不恢复 TRUE、SHUFFLED、KSVD updates 或 relation。

## 3. Confirmatory gate

在 2 splits × 3 model seeds × 3 folds = 18 个单元上，要求全部满足：

1. BAG−GINE mean `≥+0.5pt`；
2. 至少 12/18 folds 为正；
3. split seeds 3/4 的 aggregated means 均为正；
4. 至少 2/3 model-seed means 为正；
5. 最差 model-seed mean 不低于 `−0.5pt`；
6. 六个 split×model cells 中至少 4 个 cell means 为正。

通过后才扩展到 split seeds 1/2 × model seeds 1/2，形成完整双轴矩阵。失败则停止 Beam8
分类扩展，只保留已有结果作为条件校准诊断，不进入更复杂融合。
