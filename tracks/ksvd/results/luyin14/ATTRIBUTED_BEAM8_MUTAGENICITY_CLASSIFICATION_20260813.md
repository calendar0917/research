# Attributed Beam8/Mutagenicity classification

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_CLASSIFICATION_PROTOCOL_20260813.md`  
> 判定：`TYPED_BEAM8_CLASSIFICATION_BELOW_GATE`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| FEATURE_STATS | 0.7052 ± 0.0048 | 0.7129 |
| BASE_RAW_BAG | 0.6944 ± 0.0041 | 0.7009 |
| BASE_RAW_TRUE | 0.6878 ± 0.0087 | 0.6943 |
| BASE_RAW_SHUFFLED | 0.6934 ± 0.0104 | 0.6996 |
| BASE_INIT_TRUE | 0.6795 ± 0.0111 | 0.6855 |
| BASE_FINAL_TRUE | 0.6859 ± 0.0051 | 0.6936 |
| ANCHOR_RAW_BAG | 0.6920 ± 0.0022 | 0.6982 |
| ANCHOR_RAW_TRUE | 0.6878 ± 0.0072 | 0.6936 |
| ANCHOR_RAW_SHUFFLED | 0.6921 ± 0.0063 | 0.6984 |
| ANCHOR_INIT_BAG | 0.6786 ± 0.0014 | 0.6867 |
| ANCHOR_INIT_TRUE | 0.6788 ± 0.0038 | 0.6857 |
| ANCHOR_FINAL_BAG | 0.6838 ± 0.0019 | 0.6922 |
| ANCHOR_FINAL_TRUE | 0.6779 ± 0.0046 | 0.6862 |
| ANCHOR_FINAL_SHUFFLED | 0.6877 ± 0.0055 | 0.6959 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| base_relation | -0.0056 | 0/0/3 |
| anchor_increment | -0.0006 | 2/0/1 |
| anchor_relation | -0.0098 | 0/0/3 |
| anchor_over_bag | -0.0059 | 0/0/3 |
| ksvd_update | -0.0009 | 1/0/2 |
| attribute_complement | -0.0264 | 0/0/3 |

## Frozen checks

- base_relation：`False`；
- anchor_increment：`False`；
- anchor_relation：`False`；
- anchor_over_bag：`False`；
- ksvd_update：`False`；
- attribute_complement：`False`；

## Boundary

- 本结果只使用通过 relabel-invariance gate 的 atom-colored/bond-typed canonical representation。
- BASE/ANCHOR 共用 outer-train ANCHOR dictionary 与 normalization。
- anchor 为独立 segment；其收益与连续 Beam8 relation、KSVD updates 分开解释。
- 旧 binary-order typed classification 是 invalidated diagnostic，不参与证据汇总。
