# Attributed Beam8 frozen-GINE residual calibration

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_GINE_RESIDUAL_ADVANCES_TO_MULTI_MODEL_SEED`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7268 ± 0.0239 | 0.7247 |
| TRUE_RESIDUAL | 0.7563 ± 0.0146 | 0.7542 |
| SHUFFLED_RESIDUAL | 0.7480 ± 0.0152 | 0.7475 |
| BAG_RESIDUAL | 0.7445 ± 0.0179 | 0.7420 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| increment | +0.0296 | 3/0/0 |
| binding | +0.0083 | 2/0/1 |
| localization | +0.0118 | 3/0/0 |

## Frozen checks

- increment：`True`；
- binding：`True`；
- localization：`True`；

## Boundary

- residual training never updates GINE parameters or BatchNorm state.
- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity.
