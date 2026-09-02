# TU ENZYMES full-node-attribute prescreen

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_FULL_ATTRIBUTE_PRESCREEN_PROTOCOL_20260814.md`  
> 判定：`ENZYMES_DO_NOT_ADVANCE_TO_BEAM8`

| variant | balanced accuracy over 27 units |
|---|---:|
| GLOBAL_STATS_LINEAR | 0.4829 ± 0.0215 |
| GIN_LABEL_ONLY | 0.2386 ± 0.0357 |
| GIN_FULL_ATTRIBUTES | 0.4985 ± 0.0526 |
| GIN_FULL_PLUS_GLOBAL | 0.5597 ± 0.0434 |

## Paired screening

| comparison | mean | W/T/L | split0/1/2 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_vs_global | +0.0156 | 16/0/11 | +0.0472 / +0.0081 / -0.0085 | +0.0193 / +0.0191 / +0.0083 |
| full_vs_label_only | +0.2599 | 27/0/0 | +0.2939 / +0.2438 / +0.2419 | +0.2683 / +0.2461 / +0.2652 |
| combined_vs_global | +0.0768 | 25/0/2 | +0.1023 / +0.0554 / +0.0726 | +0.0768 / +0.0791 / +0.0743 |
| combined_vs_full | +0.0612 | 26/1/0 | +0.0551 / +0.0473 / +0.0811 | +0.0575 / +0.0600 / +0.0660 |

## Frozen checks

- full_beats_global：`False`；
- continuous_attributes_help：`True`；
- all_split_means_positive：`False`；
- model_majority_positive：`True`；
- combined_beats_global：`True`；
- dimensions：`True`；
- multi_patch_size_proxy：`True`；

## Boundary

- GIN 读取 18 continuous + 3 discrete；连续属性 train-only 标准化。
- 后续 canonicalization 只能读取 3 维离散 labels，不能把连续属性 argmax 当颜色。
- 本轮只决定是否进入低容量 Beam8 matched controls，不构成 Beam8 分类证据。
