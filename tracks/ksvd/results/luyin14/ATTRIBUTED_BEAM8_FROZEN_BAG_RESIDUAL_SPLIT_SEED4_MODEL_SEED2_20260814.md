# Attributed Beam8 graph-conditioned frozen BAG residual

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md`  
> split/model seed：`4/2`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7492 ± 0.0178 | 0.7448 |
| BAG_RESIDUAL | 0.7623 ± 0.0142 | 0.7595 |

## Paired BAG−GINE

- mean：`+0.0131`；
- W/T/L：`3/0/0`；

## Boundary

- GINE parameters and BatchNorm state remain frozen during residual training.
- No localized TRUE/SHUFFLED field, relation message or KSVD update is used.
