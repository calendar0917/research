# Attributed Beam8 content-aware relation Stage B1

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_CONTENT_RELATION_STAGE_B1_PROTOCOL_20260814.md`  
> 判定：`LOCALIZED_INIT_ONLY_STOP_PATCH_RELATION`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_ONLY | 0.7268 ± 0.0239 | 0.7247 |
| INIT_LOCAL_CONTENT | 0.7490 ± 0.0126 | 0.7526 |
| INIT_RELATION_TRUE | 0.6649 ± 0.0156 | 0.6608 |
| INIT_RELATION_SHUFFLED | 0.6952 ± 0.0089 | 0.6924 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| classification | -0.0619 | 0/0/3 |
| relation_increment | -0.0842 | 0/0/3 |
| relation_binding | -0.0303 | 0/0/3 |
| local_classification | +0.0223 | 2/0/1 |

## Frozen checks

- classification：`False`；
- relation_increment：`False`；
- relation_binding：`False`；

## Boundary

- 本轮只使用 INIT vocabulary；没有普通 KSVD updates。
- SHUFFLED 只打乱邻居 message codes，local patch content、patch graph 与 node incidence 保持不变。
- 通过 Stage B1 才允许扩展 model seeds 或学习 atom gate。
