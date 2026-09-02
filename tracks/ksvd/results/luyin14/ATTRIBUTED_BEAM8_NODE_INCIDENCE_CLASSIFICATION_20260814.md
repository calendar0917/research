# Attributed Beam8 node-incidence classification

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_NODE_INCIDENCE_CLASSIFICATION_PROTOCOL_20260814.md`  
> 判定：`BEAM8_NODE_INCIDENCE_CLASSIFICATION_BELOW_GATE`

| variant | balanced accuracy | accuracy | epoch |
|---|---:|---:|---:|
| GINE_ONLY | 0.7268 ± 0.0239 | 0.7247 | 40.3 |
| RAW_FULL_TRUE | 0.6969 ± 0.0116 | 0.7019 | 19.3 |
| INIT_FULL_TRUE | 0.7354 ± 0.0113 | 0.7381 | 39.7 |
| FINAL_FULL_TRUE | 0.7240 ± 0.0114 | 0.7240 | 28.7 |
| FINAL_FULL_SHUFFLED | 0.6963 ± 0.0112 | 0.6963 | 37.3 |
| FINAL_BAG_BROADCAST | 0.7302 ± 0.0081 | 0.7240 | 38.7 |
| FINAL_NO_RELATION | 0.7122 ± 0.0316 | 0.7217 | 35.0 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| classification_increment | -0.0028 | 2/0/1 |
| binding | +0.0276 | 3/0/0 |
| localization | -0.0062 | 1/0/2 |
| relation_metadata | +0.0118 | 1/0/2 |
| ksvd_update | -0.0114 | 0/0/3 |
| compression_vs_raw | +0.0270 | 3/0/0 |

## Frozen checks

- classification_increment：`False`；
- binding：`True`；
- localization：`False`；
- relation_metadata：`False`；
- ksvd_update：`False`；

## Boundary

- 所有结构通道均为 attributed-orbit-safe node-equivariant features。
- outer-test 不参与字典、normalization、epoch 或模型选择。
- SHUFFLED 保留同图 node incidence row multiset；BAG 保留图级均值但删除 localization。
