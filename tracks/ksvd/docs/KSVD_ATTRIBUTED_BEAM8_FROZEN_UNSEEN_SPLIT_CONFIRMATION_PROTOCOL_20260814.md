# Attributed Beam8 frozen residual unseen-split confirmation

> 日期：2026-08-14  
> 状态：split seeds 0/1/2 repeated-SHUFFLED diagnostic 通过后冻结；split seeds 3/4
> 结果不可见。

## 1. 研究问题

现有证据分别支持：frozen residual 的净分类增量、相对 BAG 的 localization，以及在 72 个
错误绑定上的小幅 TRUE binding 优势。但 repeated-SHUFFLED protocol 是看到 split seeds 0/1/2
的 single-shuffle 结果后设计的。本轮在未见 split seeds 3/4 上确认：

> frozen Beam8 conditional residual 的 classification、localization 与 exact binding 是否能
> 同时外推到新的数据划分？

## 2. 固定设计

- 数据：TU Mutagenicity；
- confirmatory split seeds：3/4，各 3 stratified outer folds；
- model seed：固定为 0；
- 每个 fold 重新在 outer-train 上拟合 attributed Beam8 INIT dictionary；
- strict inner validation 选择 base/residual epochs，outer-test 只评估一次；
- GINE 完全冻结，residual 保持 rank 16、zero-init 与同一优化配置；
- 变体：GINE_FROZEN、TRUE_RESIDUAL、BAG_RESIDUAL；
- binding：每个 fold 运行 8 个独立 SHUFFLED realizations，共 48 个 comparisons；
- repeat 0 必须精确复现对应 frozen residual run 的 SHUFFLED score。

不调整 dictionary、rank、学习率、patience、shuffle 约束或 Beam8 参数。

## 3. Confirmatory gate

六个 unseen split×fold 单元必须同时满足：

1. classification：TRUE−GINE mean `≥+1.0pt`，至少 4/6 folds 正；
2. localization：TRUE−BAG mean `≥+0.5pt`，至少 4/6 folds 正；
3. split consistency：split 3/4 的 classification means 均为正；
4. split consistency：split 3/4 的 localization means 均为正；
5. repeated binding audit：
   - TRUE−SHUFFLED mean `≥+0.5pt`；
   - 至少 29/48 comparisons 正；
   - 至少 4/6 fold means 正；
   - split 3/4 means 均为正；
   - 最差 split mean 不低于 `−0.5pt`；
6. 所有 GINE/repeat0 parity checks 精确通过。

全部通过才允许进入冻结 dictionary 的低容量 atom/channel gate。任一项失败，则保持当前
frozen residual 作为最终 Beam8 分类机制，不增加监督 dictionary、cross-attention 或普通
KSVD updates。
