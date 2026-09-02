# Attributed Beam8 frozen-GINE residual calibration

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_GINE_RESIDUAL_PROTOCOL_20260814.md`  
> 判定：`FROZEN_GINE_RESIDUAL_BELOW_GATE_STOP_MUTAGENICITY_CLASSIFICATION`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7474 ± 0.0225 | 0.7533 |
| TRUE_RESIDUAL | 0.7586 ± 0.0167 | 0.7609 |
| SHUFFLED_RESIDUAL | 0.7541 ± 0.0190 | 0.7565 |
| BAG_RESIDUAL | 0.7434 ± 0.0229 | 0.7489 |

## Paired attribution

| comparison | mean | W/T/L |
|---|---:|---:|
| increment | +0.0113 | 2/0/1 |
| binding | +0.0046 | 3/0/0 |
| localization | +0.0153 | 3/0/0 |

## Frozen checks

- increment：`True`；
- binding：`False`；
- localization：`True`；

## Boundary

- residual training never updates GINE parameters or BatchNorm state.
- TRUE/SHUFFLED/BAG share the same frozen base model, checkpoint epochs, seeds and residual capacity.
