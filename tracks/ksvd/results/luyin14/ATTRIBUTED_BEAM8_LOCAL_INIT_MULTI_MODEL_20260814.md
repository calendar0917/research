# Attributed Beam8 localized INIT multi-model-seed validation

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_LOCAL_INIT_MULTI_MODEL_PROTOCOL_20260814.md`  
> 判定：`LOCALIZED_INIT_NOT_STABLE_ACROSS_MODEL_SEEDS`

| variant | balanced accuracy over 9 units |
|---|---:|
| GINE_ONLY | 0.7391 ± 0.0362 |
| INIT_LOCAL_CONTENT | 0.7158 ± 0.0347 |

- LOCAL−GINE：`-0.0233`，W/T/L `2/0/7`；

## Model-seed means

- seed 0：`+0.0223`；
- seed 1：`-0.0486`；
- seed 2：`-0.0436`；

## Frozen checks

- mean_increment：`False`；
- unit_majority：`False`；
- seed_majority：`False`；
- worst_seed：`False`；

## Boundary

- split seed 固定为 0；本轮只验证 neural initialization，不是跨数据划分确认。
- seed0 读取冻结 Stage-B1 folds；seed1/2 使用完全相同的数据、dictionary 和 checkpoint protocol。
