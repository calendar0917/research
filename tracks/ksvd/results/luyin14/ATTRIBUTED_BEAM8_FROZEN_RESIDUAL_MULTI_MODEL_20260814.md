# Attributed Beam8 frozen residual multi-model-seed validation

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_RESIDUAL_MULTI_MODEL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_RESIDUAL_ADVANCES_TO_MULTI_SPLIT`

| variant | balanced accuracy over 9 units |
|---|---:|
| GINE_FROZEN | 0.7328 ± 0.0544 |
| TRUE_RESIDUAL | 0.7672 ± 0.0205 |
| SHUFFLED_RESIDUAL | 0.7562 ± 0.0196 |
| BAG_RESIDUAL | 0.7503 ± 0.0262 |

## Paired attribution

| comparison | mean | W/T/L | seed0/1/2 |
|---|---:|---:|---:|
| increment | +0.0344 | 9/0/0 | +0.0296 / +0.0092 / +0.0643 |
| binding | +0.0110 | 8/0/1 | +0.0083 / +0.0130 / +0.0118 |
| localization | +0.0169 | 8/0/1 | +0.0118 / +0.0220 / +0.0169 |

## Frozen checks

- increment：`True`；
- binding：`True`；
- localization：`True`；
- seed_majority：`True`；
- worst_seed_increment：`True`；

## Boundary

- split seed 固定为0；通过后仍需跨 split 验证。
- 每个 residual variant 使用同 model-seed 下完全相同的 frozen GINE。
