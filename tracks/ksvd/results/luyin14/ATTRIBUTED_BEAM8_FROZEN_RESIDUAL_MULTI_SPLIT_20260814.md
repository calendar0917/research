# Attributed Beam8 frozen residual multi-split validation

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_RESIDUAL_MULTI_SPLIT_PROTOCOL_20260814.md`  
> 判定：`FROZEN_RESIDUAL_NOT_STABLE_ACROSS_SPLITS`

| variant | balanced accuracy over 9 units |
|---|---:|
| GINE_FROZEN | 0.7332 ± 0.0341 |
| TRUE_RESIDUAL | 0.7524 ± 0.0257 |
| SHUFFLED_RESIDUAL | 0.7495 ± 0.0239 |
| BAG_RESIDUAL | 0.7404 ± 0.0308 |

## Paired attribution

| comparison | mean | W/T/L | split0/1/2 |
|---|---:|---:|---:|
| increment | +0.0192 | 8/0/1 | +0.0296 / +0.0113 / +0.0167 |
| binding | +0.0029 | 6/0/3 | +0.0083 / +0.0046 / -0.0042 |
| localization | +0.0120 | 9/0/0 | +0.0118 / +0.0153 / +0.0089 |

## Frozen checks

- increment：`True`；
- binding：`False`；
- localization：`True`；
- split_majority：`True`；
- worst_split_increment：`True`；

## Boundary

- model seed 固定为0；本报告只回答跨数据划分稳定性。
- 每个 split/fold 的 dictionary 只在对应 outer-train 上拟合。
- TRUE/SHUFFLED/BAG 在每个 split/fold 内共用完全相同的 frozen GINE。
