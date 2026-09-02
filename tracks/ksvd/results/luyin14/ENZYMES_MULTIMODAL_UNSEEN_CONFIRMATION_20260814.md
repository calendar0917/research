# ENZYMES multimodal baseline unseen-split confirmation

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_MULTIMODAL_UNSEEN_CONFIRMATION_PROTOCOL_20260814.md`  
> 判定：`ENZYMES_MULTIMODAL_BASE_CONFIRMED_ON_UNSEEN_SPLITS`

| variant | balanced accuracy over 18 units |
|---|---:|
| GLOBAL_STATS_LINEAR | 0.4860 ± 0.0321 |
| GIN_FULL_ATTRIBUTES | 0.4942 ± 0.0637 |
| GIN_LABEL_ONLY_PLUS_GLOBAL | 0.4829 ± 0.0529 |
| GIN_FULL_PLUS_GLOBAL | 0.5817 ± 0.0481 |

## Paired confirmation

| comparison | mean | W/T/L | split3/4 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_global_vs_global | +0.0957 | 17/0/1 | +0.0845 / +0.1070 | +0.1100 / +0.0749 / +0.1023 |
| full_global_vs_label_global | +0.0988 | 16/0/2 | +0.0890 / +0.1086 | +0.1315 / +0.0791 / +0.0858 |
| full_vs_global | +0.0082 | 10/0/8 | +0.0022 / +0.0143 | +0.0273 / -0.0040 / +0.0014 |

## Frozen checks

- combined_increment：`True`；
- attribute_organization：`True`；
- both_splits_increment：`True`；
- both_splits_organization：`True`；
- model_majority_increment：`True`；
- model_majority_organization：`True`；
- dimensions：`True`；

## Boundary

- split3/4 在协议冻结前未用于 ENZYMES 模型选择或阈值调整。
- 通过只确认强多模态基线；Beam8 必须在新的 split5/6 上做条件增量 matched controls。
- 失败时停止 ENZYMES Beam8 分类路线。
