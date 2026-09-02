# BZR Beam8 patch-graph conditional increment

> 协议：`tracks/ksvd/docs/KSVD_BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_PROTOCOL_20260814.md`  
> 判定：`BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_INCREMENT_NOT_ESTABLISHED`

| variant | balanced accuracy over 6 unseen units |
|---|---:|
| GLOBAL_ONLY | 0.6943 ± 0.0618 |
| GLOBAL_PLUS_BAG | 0.6270 ± 0.0333 |
| GLOBAL_PLUS_PATCH_GRAPH_TRUE | 0.6395 ± 0.0351 |
| GLOBAL_PLUS_PATCH_GRAPH_TOKEN_SHUFFLED | 0.6445 ± 0.0391 |

| comparison | mean | W/T/L | split3/4 |
|---|---:|---:|---:|
| true_vs_global | -0.0548 | 2/0/4 | -0.0348 / -0.0748 |
| true_vs_bag | +0.0124 | 4/0/2 | +0.0214 / +0.0034 |
| true_vs_token_shuffled | -0.0050 | 2/1/3 | -0.0068 / -0.0032 |

## Frozen checks

- conditional_increment：`False`；
- relation_increment：`True`；
- binding：`False`；
- both_splits_increment：`False`；
- both_splits_relation：`True`；
- both_splits_binding：`False`；
- invariance：`True`；

## Boundary

- 原 BZR prescreen 失败判定保持不变；本轮只检查 GLOBAL_STATS 条件后的独立增量。
- 四个 variants 输入维度一致，BAG 与 GLOBAL_ONLY 使用末尾零 padding。
- split3/4 在协议冻结前未用于选择表示、阈值或分类器。
- 失败时不训练 patch-GNN；通过时也只能在新 split5/6 做一层 matched-control 验证。
