# ENZYMES continuous-attribute organization follow-up

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_ATTRIBUTE_ORGANIZATION_FOLLOWUP_PROTOCOL_20260814.md`  
> 判定：`CONTINUOUS_ATTRIBUTE_ORGANIZATION_ADDS_BEYOND_GLOBAL_STATS`

| variant | balanced accuracy over 27 units |
|---|---:|
| GLOBAL_STATS_LINEAR | 0.4829 ± 0.0215 |
| GIN_LABEL_ONLY | 0.2386 ± 0.0357 |
| GIN_LABEL_ONLY_PLUS_GLOBAL | 0.4507 ± 0.0725 |
| GIN_FULL_PLUS_GLOBAL | 0.5597 ± 0.0434 |

## Paired attribution

| comparison | mean | W/T/L | split0/1/2 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_global_vs_label_global | +0.1090 | 26/0/1 | +0.1270 / +0.0730 / +0.1270 | +0.1076 / +0.0977 / +0.1217 |
| label_global_vs_global | -0.0322 | 10/0/17 | -0.0247 / -0.0175 / -0.0544 | -0.0307 / -0.0185 / -0.0474 |
| full_global_vs_global | +0.0768 | 25/0/2 | +0.1023 / +0.0554 / +0.0726 | +0.0768 / +0.0791 / +0.0743 |

## Checks

- mean：`True`；
- wins：`True`；
- all_splits：`True`；
- model_majority：`True`；
- parity：`True`；

## Boundary

- label-only 与 full-attribute 分支使用相同统计 residual 容量和训练协议。
- 本轮不事后修改原 ENZYMES Beam8 晋级 gate。
- 通过只说明连续属性的图内组织是独立融合信号，不证明 Beam8 能读取该信号。
