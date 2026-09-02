# Beam8/Mutagenicity typed-anchor classification

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_CLASSIFICATION_PROTOCOL_20260813.md`  
> 判定：`TYPED_BEAM8_CLASSIFICATION_BELOW_GATE`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| FEATURE_STATS | 0.7052 ± 0.0048 | 0.7129 |
| BASE_RAW_BAG | 0.6764 ± 0.0123 | 0.6843 |
| BASE_RAW_TRUE | 0.6748 ± 0.0174 | 0.6825 |
| BASE_RAW_SHUFFLED | 0.6720 ± 0.0122 | 0.6802 |
| BASE_INIT_TRUE | 0.6763 ± 0.0108 | 0.6839 |
| BASE_FINAL_TRUE | 0.6883 ± 0.0115 | 0.6959 |
| ANCHOR_RAW_BAG | 0.6671 ± 0.0160 | 0.6751 |
| ANCHOR_RAW_TRUE | 0.6645 ± 0.0151 | 0.6724 |
| ANCHOR_RAW_SHUFFLED | 0.6599 ± 0.0133 | 0.6682 |
| ANCHOR_INIT_BAG | 0.6731 ± 0.0020 | 0.6813 |
| ANCHOR_INIT_TRUE | 0.6654 ± 0.0065 | 0.6733 |
| ANCHOR_FINAL_BAG | 0.6834 ± 0.0042 | 0.6910 |
| ANCHOR_FINAL_TRUE | 0.6799 ± 0.0075 | 0.6873 |
| ANCHOR_FINAL_SHUFFLED | 0.6818 ± 0.0075 | 0.6892 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| base_relation | +0.0027 | 2/0/1 |
| anchor_increment | -0.0109 | 0/0/3 |
| anchor_relation | -0.0019 | 0/0/3 |
| anchor_over_bag | -0.0035 | 1/0/2 |
| ksvd_update | +0.0145 | 2/0/1 |
| attribute_complement | -0.0398 | 0/0/3 |

## Checks

- base_relation：`False`；
- anchor_increment：`False`；
- anchor_relation：`False`；
- anchor_over_bag：`False`；
- ksvd_update：`True`；
- attribute_complement：`False`；

## Boundary

- BASE/ANCHOR 共用 outer-train ANCHOR dictionary 与 normalization。
- anchor 是独立 segment；其收益不能解释为连续 Beam8 chain 收益。
