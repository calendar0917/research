# Attributed Beam8 frozen binding repeated-SHUFFLED audit

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BINDING_SHUFFLE_AUDIT_PROTOCOL_20260814.md`  
> 判定：`EXACT_BINDING_SUPPORTED_AGAINST_SHUFFLE_DISTRIBUTION`

## Aggregate

| metric | value |
|---|---:|
| comparisons | 72 |
| TRUE−SHUFFLED mean | +0.0061 |
| delta std | 0.0113 |
| W/T/L | 51/0/21 |
| split0/1/2 mean | +0.0055 / +0.0087 / +0.0040 |

## Fold means over shuffle realizations

| split:fold | TRUE−mean(SHUFFLED) |
|---|---:|
| 0:0 | +0.0015 |
| 0:1 | +0.0049 |
| 0:2 | +0.0102 |
| 1:0 | -0.0049 |
| 1:1 | +0.0088 |
| 1:2 | +0.0221 |
| 2:0 | +0.0076 |
| 2:1 | -0.0028 |
| 2:2 | +0.0071 |

## Frozen checks

- repeat0_parity：`True`；
- mean_binding：`True`；
- comparison_win_rate：`True`；
- fold_majority：`True`；
- split_majority：`True`；
- worst_split：`True`；

## Boundary

- repeat 0 与原 frozen-GINE SHUFFLED 结果逐 fold 一致。
- 该审计在 single-shuffle 结果可见后设计，只用于机制诊断。
- TRUE score、base epoch、model seed、dictionary recipe 与 residual capacity 均未重选。
