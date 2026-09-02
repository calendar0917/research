# Attributed Beam8 frozen-GINE residual calibration

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_GINE_RESIDUAL_BELOW_GATE_STOP_MUTAGENICITY_CLASSIFICATION`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7255 ± 0.0460 | 0.7413 |
| TRUE_RESIDUAL | 0.7423 ± 0.0365 | 0.7507 |
| SHUFFLED_RESIDUAL | 0.7465 ± 0.0330 | 0.7528 |
| BAG_RESIDUAL | 0.7334 ± 0.0439 | 0.7438 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| increment | +0.0167 | 3/0/0 |
| binding | -0.0042 | 1/0/2 |
| localization | +0.0089 | 3/0/0 |

## Frozen checks

- increment：`True`；
- binding：`False`；
- localization：`True`；

## Boundary

- residual training never updates GINE parameters or BatchNorm state.
- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity.
