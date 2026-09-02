# ENZYMES full-feature-canonical Beam8 conditional controls

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_FEATURE_CANONICAL_CONTROLS_PROTOCOL_20260814.md`  
> 判定：`ENZYMES_FEATURE_CANONICAL_BEAM8_SPECIFIC_INCREMENT_NOT_ESTABLISHED`

| variant | balanced accuracy over 18 units |
|---|---:|
| BASE_MULTIMODAL | 0.5419 ± 0.0591 |
| BEAM8_FULL_TRUE | 0.5383 ± 0.0554 |
| BEAM8_FULL_BAG | 0.5397 ± 0.0584 |
| BEAM8_FULL_SHUFFLED | 0.5394 ± 0.0543 |
| BEAM8_CODE_ONLY_TRUE | 0.5411 ± 0.0591 |
| BEAM8_HIST_ONLY_TRUE | 0.5489 ± 0.0581 |
| BEAM8_RANDOM_DICTIONARY_TRUE | 0.5461 ± 0.0588 |

## Paired attribution

| comparison | mean | W/T/L | split7/8 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_vs_base | -0.0036 | 1/11/6 | -0.0067 / -0.0005 | -0.0017 / -0.0059 / -0.0033 |
| full_vs_bag | -0.0014 | 4/9/5 | -0.0044 / +0.0017 | -0.0025 / -0.0017 / +0.0001 |
| full_vs_shuffled | -0.0011 | 3/9/6 | +0.0005 / -0.0027 | +0.0058 / -0.0051 / -0.0041 |
| full_vs_code_only | -0.0028 | 4/7/7 | -0.0067 / +0.0011 | -0.0017 / -0.0059 / -0.0008 |
| full_vs_hist_only | -0.0106 | 1/8/9 | -0.0183 / -0.0028 | -0.0058 / -0.0124 / -0.0134 |
| full_vs_random_dictionary | -0.0078 | 3/8/7 | -0.0183 / +0.0028 | -0.0100 / -0.0074 / -0.0059 |

## Frozen checks

- increment：`False`；
- localization：`False`；
- binding：`False`；
- beats_hist_only：`False`；
- beats_random_dictionary：`False`；
- both_splits_increment：`False`；
- both_splits_binding：`False`；
- model_majority_increment：`False`；
- model_majority_binding：`False`；

## Boundary

- BASE_MULTIMODAL 是先冻结 full-attribute GIN 与 GLOBAL_STATS residual 后的强基线。
- canonicalization/orbits 使用完整21维 feature-row colors；structural vector 仍只编码3维离散 labels。
- classification 只在 fixed feature-canonical invariance 达到100%后有效。
- 失败时不允许通过更大融合器或监督 dictionary 补救。
