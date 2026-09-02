# Attributed Beam8 frozen residual multi-split validation

> 日期：2026-08-14  
> 状态：multi-model-seed gate 通过后冻结；split seeds 1/2 结果不可见。

## 1. 研究问题

固定 split seed 0 时，frozen-GINE conditional residual 在三个 model seeds、九个
fold×model-seed 单元中稳定超过 GINE、SHUFFLED 与 BAG。本轮回答：

> 该增量能否跨 outer data split 保持，还是只依赖 split seed 0 的样本分配？

## 2. 固定设计

- 数据：TU Mutagenicity；
- split seeds：0/1/2，每个 seed 三个 stratified outer folds；
- model seed：固定为 0；
- 每个 outer fold 重新在 outer-train 上拟合 attributed Beam8 INIT dictionary；
- inner validation、dictionary sampling 与 SHUFFLED control 只确定性依赖 split seed/fold；
- GINE/residual initialization 只依赖 model seed/fold；
- frozen GINE、rank-16 zero-init conditional residual、TRUE/SHUFFLED/BAG 容量及 checkpoint
  流程保持不变；
- split seed 0 使用已经冻结的 model-seed 0 结果，不重新挑选或覆盖。

## 3. 汇总单元与预注册 gate

汇总 9 个 split-seed×outer-fold 单元，要求全部满足：

1. TRUE−GINE mean `≥+1.0pt`，至少 6/9 正；
2. TRUE−SHUFFLED mean `≥+0.5pt`，至少 6/9 正；
3. TRUE−BAG mean `≥+0.5pt`，至少 6/9 正；
4. 对上述三类 comparison，各自至少 2/3 split-seed means 为正；
5. 最差 split-seed TRUE−GINE mean 不低于 `−0.5pt`。

全部通过才将它视为跨 split 的 Beam8 分类收益机制，并允许进一步做更强外部数据集确认。
若失败，则只能声称 fixed split seed 0 下有条件残差信号，不进入 atom gate、监督字典或
cross-attention。
