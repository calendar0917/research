# Beam8 small attributed-TU patch-graph prescreen

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_SMALL_TU_PATCH_GRAPH_PRESCREEN_PROTOCOL_20260814.md`  
> 晋级数据集：`[]`

## BZR

> 判定：`BZR_BEAM8_PATCH_GRAPH_PRESCREEN_NOT_ESTABLISHED`

| variant | balanced accuracy over 9 units |
|---|---:|
| GLOBAL_STATS | 0.6854 ± 0.0520 |
| RAW_BAG | 0.6176 ± 0.0384 |
| RAW_PATCH_GRAPH_TRUE | 0.6345 ± 0.0471 |
| RAW_PATCH_GRAPH_TOKEN_SHUFFLED | 0.6136 ± 0.0477 |

| comparison | mean | W/T/L | split0/1/2 |
|---|---:|---:|---:|
| true_vs_token_shuffled | +0.0209 | 7/1/1 | +0.0322 / +0.0018 / +0.0287 |
| true_vs_bag | +0.0169 | 7/0/2 | +0.0227 / +0.0019 / +0.0260 |
| true_vs_global_stats | -0.0509 | 3/0/6 | -0.1024 / -0.0563 / +0.0059 |

### Frozen checks

- binding：`True`；
- relation_increment：`True`；
- beats_global_stats：`False`；
- all_split_binding_positive：`True`；
- all_split_increment_positive：`True`；
- invariance：`True`；

### Relabel audit

- chain_exact_match：`0.625000`（diagnostic）；
- token_row_match：`1.000000`（required）；
- relation_matrix_match：`1.000000`（required）；
- true_feature_match：`1.000000`（required）；
- shuffled_feature_match：`1.000000`（required）；

## COX2

> 判定：`COX2_BEAM8_PATCH_GRAPH_PRESCREEN_NOT_ESTABLISHED`

| variant | balanced accuracy over 9 units |
|---|---:|
| GLOBAL_STATS | 0.6678 ± 0.0543 |
| RAW_BAG | 0.5702 ± 0.0474 |
| RAW_PATCH_GRAPH_TRUE | 0.5724 ± 0.0415 |
| RAW_PATCH_GRAPH_TOKEN_SHUFFLED | 0.5584 ± 0.0399 |

| comparison | mean | W/T/L | split0/1/2 |
|---|---:|---:|---:|
| true_vs_token_shuffled | +0.0140 | 6/0/3 | +0.0197 / +0.0325 / -0.0102 |
| true_vs_bag | +0.0022 | 5/0/4 | -0.0280 / +0.0257 / +0.0090 |
| true_vs_global_stats | -0.0954 | 0/0/9 | -0.0966 / -0.0533 / -0.1362 |

### Frozen checks

- binding：`True`；
- relation_increment：`False`；
- beats_global_stats：`False`；
- all_split_binding_positive：`False`；
- all_split_increment_positive：`False`；
- invariance：`True`；

### Relabel audit

- chain_exact_match：`0.343750`（diagnostic）；
- token_row_match：`1.000000`（required）；
- relation_matrix_match：`1.000000`（required）；
- true_feature_match：`1.000000`（required）；
- shuffled_feature_match：`1.000000`（required）；

## DHFR

> 判定：`DHFR_BEAM8_PATCH_GRAPH_PRESCREEN_NOT_ESTABLISHED`

| variant | balanced accuracy over 9 units |
|---|---:|
| GLOBAL_STATS | 0.6760 ± 0.0237 |
| RAW_BAG | 0.6489 ± 0.0202 |
| RAW_PATCH_GRAPH_TRUE | 0.6493 ± 0.0242 |
| RAW_PATCH_GRAPH_TOKEN_SHUFFLED | 0.6407 ± 0.0096 |

| comparison | mean | W/T/L | split0/1/2 |
|---|---:|---:|---:|
| true_vs_token_shuffled | +0.0086 | 6/0/3 | +0.0240 / -0.0096 / +0.0114 |
| true_vs_bag | +0.0004 | 4/0/5 | +0.0276 / -0.0201 / -0.0064 |
| true_vs_global_stats | -0.0267 | 4/0/5 | -0.0030 / -0.0627 / -0.0143 |

### Frozen checks

- binding：`False`；
- relation_increment：`False`；
- beats_global_stats：`False`；
- all_split_binding_positive：`False`；
- all_split_increment_positive：`False`；
- invariance：`True`；

### Relabel audit

- chain_exact_match：`0.406250`（diagnostic）；
- token_row_match：`1.000000`（required）；
- relation_matrix_match：`1.000000`（required）；
- true_feature_match：`1.000000`（required）；
- shuffled_feature_match：`1.000000`（required）；

## Boundary

- 本轮只使用官方离散 node labels；3D coordinates 不进入模型。
- RAW token 不经过 KSVD，直接检查 Beam8 patch graph substrate。
- 未晋级的数据集不运行可学习 patch-GNN。
- 晋级者只能在新的 split3/4 上运行一层 patch-GNN，并保留 matched controls。
