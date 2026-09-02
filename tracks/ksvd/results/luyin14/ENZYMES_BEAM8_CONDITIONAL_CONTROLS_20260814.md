# ENZYMES Beam8 conditional matched controls（作废）

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_CONDITIONAL_CONTROLS_PROTOCOL_20260814.md`  
> 原判定：`ENZYMES_BEAM8_SPECIFIC_INCREMENT_NOT_ESTABLISHED`  
> 最终状态：`INVALIDATED_BY_RELABEL_INVARIANCE_FAILURE`

本结果生成后进行的 64 graphs × 3 permutations 审计显示，token rows、multiset、readout 与
node-incidence equivariance 的匹配率仅约 `0.71`。离散 canonical automorphism 内的节点可能
具有不同连续属性，重标号后 cover 选择不同等价节点会改变 patch content。因此下表只能用于
定位实现问题，不得作为 Beam8 分类正负证据；修复后必须使用新的未见 split seeds 重跑。

| variant | balanced accuracy over 18 units |
|---|---:|
| BASE_MULTIMODAL | 0.5518 ± 0.0401 |
| BEAM8_FULL_TRUE | 0.5544 ± 0.0392 |
| BEAM8_FULL_BAG | 0.5496 ± 0.0382 |
| BEAM8_FULL_SHUFFLED | 0.5497 ± 0.0375 |
| BEAM8_CODE_ONLY_TRUE | 0.5515 ± 0.0422 |
| BEAM8_HIST_ONLY_TRUE | 0.5546 ± 0.0387 |
| BEAM8_RANDOM_DICTIONARY_TRUE | 0.5533 ± 0.0401 |

## Paired attribution

| comparison | mean | W/T/L | split5/6 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_vs_base | +0.0025 | 5/11/2 | +0.0056 / -0.0005 | -0.0025 / +0.0084 / +0.0016 |
| full_vs_bag | +0.0047 | 7/8/3 | +0.0083 / +0.0011 | +0.0008 / +0.0058 / +0.0075 |
| full_vs_shuffled | +0.0047 | 6/9/3 | +0.0089 / +0.0005 | +0.0050 / +0.0025 / +0.0066 |
| full_vs_code_only | +0.0029 | 7/6/5 | +0.0045 / +0.0012 | +0.0026 / +0.0068 / -0.0008 |
| full_vs_hist_only | -0.0003 | 4/9/5 | +0.0050 / -0.0055 | -0.0050 / +0.0093 / -0.0051 |
| full_vs_random_dictionary | +0.0011 | 6/7/5 | +0.0033 / -0.0011 | +0.0009 / +0.0041 / -0.0017 |

## Frozen checks

- increment：`False`；
- localization：`False`；
- binding：`False`；
- beats_hist_only：`False`；
- beats_random_dictionary：`False`；
- both_splits_increment：`False`；
- both_splits_binding：`True`；
- model_majority_increment：`True`；
- model_majority_binding：`True`；

## Boundary

- BASE_MULTIMODAL 是先冻结 full-attribute GIN 与 GLOBAL_STATS residual 后的强基线。
- canonicalization/orbits 只读取离散 labels；连续属性只进入 patch content。
- 失败时不允许通过更大融合器或监督 dictionary 补救。
