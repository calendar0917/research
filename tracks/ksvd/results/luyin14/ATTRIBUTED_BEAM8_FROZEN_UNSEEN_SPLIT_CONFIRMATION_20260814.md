# Attributed Beam8 frozen residual unseen-split confirmation

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_UNSEEN_SPLIT_CONFIRMATION_PROTOCOL_20260814.md`  
> 判定：`UNSEEN_SPLITS_DO_NOT_CONFIRM_FROZEN_BEAM8_ROUTE_STOP_EXPANSION`

## Classification variants

| variant | balanced accuracy over 6 units |
|---|---:|
| GINE_FROZEN | 0.7368 ± 0.0242 |
| TRUE_RESIDUAL | 0.7466 ± 0.0150 |
| SHUFFLED_RESIDUAL | 0.7514 ± 0.0134 |
| BAG_RESIDUAL | 0.7556 ± 0.0131 |

## Paired classification

| comparison | mean | W/T/L | split3/4 |
|---|---:|---:|---:|
| increment | +0.0098 | 5/0/1 | +0.0102 / +0.0093 |
| localization | -0.0091 | 0/0/6 | -0.0129 / -0.0052 |

## Repeated binding

- TRUE−SHUFFLED mean：`-0.0047`；
- W/T/L：`18/0/30`；
- split3/4：`-0.0129 / +0.0035`；

## Frozen checks

- classification：`False`；
- localization：`False`；
- classification_both_splits：`True`；
- localization_both_splits：`False`；
- binding_audit：`False`；
- parity：`True`；

## Boundary

- split seeds 3/4 在协议冻结前未运行。
- model seed 固定为0；本轮只确认新的数据划分。
- 全部通过只授权下一步低容量 frozen-dictionary gate，不授权 cross-attention 或普通 KSVD updates。
