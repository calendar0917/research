# Attributed Beam8 graph-conditioned frozen BAG residual

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md`  
> split/model seed：`2/1`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7542 ± 0.0224 | 0.7422 |
| BAG_RESIDUAL | 0.7620 ± 0.0235 | 0.7604 |

## Paired BAG−GINE

- mean：`+0.0078`；
- W/T/L：`2/0/1`；

## Boundary

- GINE parameters and BatchNorm state remain frozen during residual training.
- No localized TRUE/SHUFFLED field, relation message or KSVD update is used.
