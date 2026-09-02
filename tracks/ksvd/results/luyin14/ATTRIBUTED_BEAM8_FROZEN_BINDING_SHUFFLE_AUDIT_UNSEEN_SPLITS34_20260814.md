# Attributed Beam8 frozen binding repeated-SHUFFLED audit

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BINDING_SHUFFLE_AUDIT_PROTOCOL_20260814.md`  
> 判定：`EXACT_BINDING_NOT_SUPPORTED_AGAINST_SHUFFLE_DISTRIBUTION`

## Aggregate

| metric | value |
|---|---:|
| comparisons | 48 |
| TRUE−SHUFFLED mean | -0.0047 |
| delta std | 0.0128 |
| W/T/L | 18/0/30 |
| split means | s3:-0.0129 / s4:+0.0035 |

## Fold means over shuffle realizations

| split:fold | TRUE−mean(SHUFFLED) |
|---|---:|
| 3:0 | -0.0055 |
| 3:1 | -0.0255 |
| 3:2 | -0.0077 |
| 4:0 | +0.0001 |
| 4:1 | +0.0020 |
| 4:2 | +0.0084 |

## Frozen checks

- repeat0_parity：`True`；
- mean_binding：`False`；
- comparison_win_rate：`False`；
- fold_majority：`False`；
- split_majority：`False`；
- worst_split：`False`；

## Boundary

- repeat 0 与原 frozen-GINE SHUFFLED 结果逐 fold 一致。
- 该审计在 single-shuffle 结果可见后设计，只用于机制诊断。
- TRUE score、base epoch、model seed、dictionary recipe 与 residual capacity 均未重选。
