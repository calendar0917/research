# Attributed Beam8 frozen-GINE residual calibration

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_GINE_RESIDUAL_ADVANCES_TO_MULTI_MODEL_SEED`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7648 ± 0.0341 | 0.7690 |
| TRUE_RESIDUAL | 0.7741 ± 0.0290 | 0.7777 |
| SHUFFLED_RESIDUAL | 0.7611 ± 0.0240 | 0.7641 |
| BAG_RESIDUAL | 0.7521 ± 0.0349 | 0.7533 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| increment | +0.0092 | 3/0/0 |
| binding | +0.0130 | 3/0/0 |
| localization | +0.0220 | 3/0/0 |

## Frozen checks

- increment：`True`；
- binding：`True`；
- localization：`True`；

## Boundary

- residual training never updates GINE parameters or BatchNorm state.
- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity.
