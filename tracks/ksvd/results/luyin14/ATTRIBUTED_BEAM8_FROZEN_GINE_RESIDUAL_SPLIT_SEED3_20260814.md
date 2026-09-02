# Attributed Beam8 frozen-GINE residual calibration

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_GINE_RESIDUAL_BELOW_GATE_STOP_MUTAGENICITY_CLASSIFICATION`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7252 ± 0.0162 | 0.7270 |
| TRUE_RESIDUAL | 0.7354 ± 0.0071 | 0.7450 |
| SHUFFLED_RESIDUAL | 0.7470 ± 0.0069 | 0.7517 |
| BAG_RESIDUAL | 0.7483 ± 0.0058 | 0.7519 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| increment | +0.0102 | 3/0/0 |
| binding | -0.0116 | 0/0/3 |
| localization | -0.0129 | 0/0/3 |

## Frozen checks

- increment：`True`；
- binding：`False`；
- localization：`False`；

## Boundary

- residual training never updates GINE parameters or BatchNorm state.
- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity.
