# Attributed Beam8 frozen-GINE residual calibration

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_GINE_RESIDUAL_BELOW_GATE_STOP_MUTAGENICITY_CLASSIFICATION`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7484 ± 0.0253 | 0.7344 |
| TRUE_RESIDUAL | 0.7577 ± 0.0123 | 0.7544 |
| SHUFFLED_RESIDUAL | 0.7558 ± 0.0165 | 0.7554 |
| BAG_RESIDUAL | 0.7629 ± 0.0143 | 0.7563 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| increment | +0.0093 | 2/0/1 |
| binding | +0.0019 | 2/0/1 |
| localization | -0.0052 | 0/0/3 |

## Frozen checks

- increment：`True`；
- binding：`False`；
- localization：`False`；

## Boundary

- residual training never updates GINE parameters or BatchNorm state.
- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity.
