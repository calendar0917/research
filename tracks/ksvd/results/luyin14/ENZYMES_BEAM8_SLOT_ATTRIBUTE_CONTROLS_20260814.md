# ENZYMES canonical-slot attribute Beam8 controls

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_SLOT_ATTRIBUTE_PROTOCOL_20260814.md`  
> 判定：`ENZYMES_CANONICAL_SLOT_ATTRIBUTE_BEAM8_INCREMENT_NOT_ESTABLISHED`

| variant | balanced accuracy over 18 units |
|---|---:|
| BASE_MULTIMODAL | 0.5671 ± 0.0351 |
| SLOT_FULL_TRUE | 0.5696 ± 0.0376 |
| SLOT_FULL_BAG | 0.5707 ± 0.0324 |
| SLOT_FULL_INCIDENCE_SHUFFLED | 0.5707 ± 0.0330 |
| SLOT_FULL_WITHIN_PATCH_SHUFFLED | 0.5693 ± 0.0369 |
| SLOT_CODE_ONLY_TRUE | 0.5665 ± 0.0348 |
| MEAN_FULL_TRUE | 0.5643 ± 0.0360 |
| SLOT_RANDOM_DICTIONARY_TRUE | 0.5699 ± 0.0355 |

## Paired attribution

| comparison | mean | W/T/L | split9/10 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_vs_base | +0.0025 | 6/8/4 | +0.0083 / -0.0033 | +0.0092 / -0.0008 / -0.0009 |
| full_vs_bag | -0.0011 | 4/9/5 | +0.0005 / -0.0027 | +0.0067 / -0.0016 / -0.0084 |
| full_vs_incidence_shuffled | -0.0011 | 6/6/6 | -0.0022 / +0.0001 | +0.0051 / +0.0035 / -0.0118 |
| full_vs_within_patch_shuffled | +0.0003 | 6/7/5 | +0.0012 / -0.0005 | +0.0059 / -0.0024 / -0.0025 |
| full_vs_code_only | +0.0031 | 8/6/4 | +0.0088 / -0.0027 | +0.0092 / +0.0000 / -0.0001 |
| full_vs_mean | +0.0053 | 7/6/5 | +0.0078 / +0.0028 | +0.0092 / +0.0009 / +0.0058 |
| full_vs_random_dictionary | -0.0003 | 2/9/7 | +0.0049 / -0.0055 | +0.0017 / -0.0025 / -0.0001 |

## Frozen checks

- increment：`False`；
- localization：`False`；
- incidence_binding：`False`；
- slot_beats_mean：`False`；
- within_patch_binding：`False`；
- beats_random_dictionary：`False`；
- both_splits_increment：`False`；
- both_splits_slot_mean：`True`；
- both_splits_within_binding：`False`；
- model_majority_increment：`False`；
- model_majority_slot_mean：`True`；
- model_majority_within_binding：`False`；

## Boundary

- FULL patch token = 24-d train-only INIT code + canonical 8×18 continuous attributes + 8-d mask。
- BASE、dictionary、normalization、rank-16 frozen residual 与上一轮一致；只改变属性表示。
- split9/10 在协议冻结前未用于表示、阈值或超参数选择。
- 分类结果只有在独立 slot-token invariance audit 100%通过后才有效。
- 失败时停止 ENZYMES Beam8 分类，不扩大模型补救。
