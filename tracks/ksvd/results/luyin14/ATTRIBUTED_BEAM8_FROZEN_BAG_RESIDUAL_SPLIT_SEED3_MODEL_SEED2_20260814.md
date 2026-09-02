# Attributed Beam8 graph-conditioned frozen BAG residual

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md`  
> split/model seed：`3/2`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.6883 ± 0.0896 | 0.7106 |
| BAG_RESIDUAL | 0.7159 ± 0.0510 | 0.7291 |

## Paired BAG−GINE

- mean：`+0.0277`；
- W/T/L：`2/0/1`；

## Boundary

- GINE parameters and BatchNorm state remain frozen during residual training.
- No localized TRUE/SHUFFLED field, relation message or KSVD update is used.
