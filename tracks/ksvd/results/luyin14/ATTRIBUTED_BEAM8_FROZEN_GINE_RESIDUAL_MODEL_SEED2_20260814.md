# Attributed Beam8 frozen-GINE residual calibration

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_GINE_RESIDUAL_ADVANCES_TO_MULTI_MODEL_SEED`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7069 ± 0.0737 | 0.6968 |
| TRUE_RESIDUAL | 0.7712 ± 0.0048 | 0.7713 |
| SHUFFLED_RESIDUAL | 0.7594 ± 0.0155 | 0.7600 |
| BAG_RESIDUAL | 0.7543 ± 0.0215 | 0.7521 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| increment | +0.0643 | 3/0/0 |
| binding | +0.0118 | 3/0/0 |
| localization | +0.0169 | 2/0/1 |

## Frozen checks

- increment：`True`；
- binding：`True`；
- localization：`True`；

## Boundary

- residual training never updates GINE parameters or BatchNorm state.
- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity.
