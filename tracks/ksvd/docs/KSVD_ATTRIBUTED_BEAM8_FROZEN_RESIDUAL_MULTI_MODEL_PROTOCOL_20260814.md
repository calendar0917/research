# Attributed Beam8 frozen residual multi-model-seed validation

> 日期：2026-08-14  
> 状态：frozen-GINE residual seed0 三项 gate 全通过后冻结；seeds1/2 结果不可见。

固定 split seed 0、三个 outer folds、同一 Beam8 items/dictionaries/features/shuffle，只扩展
GINE/residual model seeds 1/2。seed0 使用冻结结果；所有 model seed 都运行完整 strict
two-stage base/residual checkpoint。

汇总 9 个 fold×model-seed 单元，要求：

1. TRUE−GINE mean `≥+1pt`，至少 6/9 正；
2. TRUE−SHUFFLED mean `≥+0.5pt`，至少 6/9 正；
3. TRUE−BAG mean `≥+0.5pt`，至少 6/9 正；
4. 三个 model seeds 中，以上三类 seed-level mean 各至少 2/3 为正；
5. 最差 model-seed TRUE−GINE 不低于 `-0.5pt`。

通过后才允许扩展 split seeds 1/2；失败则 seed0 结果只作为优化隔离诊断，不声称稳定分类收益。
