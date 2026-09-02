# Attributed Beam8 graph-conditioned frozen BAG residual

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md`  
> split/model seed：`1/1`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7397 ± 0.0466 | 0.7454 |
| BAG_RESIDUAL | 0.7571 ± 0.0320 | 0.7602 |

## Paired BAG−GINE

- mean：`+0.0174`；
- W/T/L：`2/0/1`；

## Boundary

- GINE parameters and BatchNorm state remain frozen during residual training.
- No localized TRUE/SHUFFLED field, relation message or KSVD update is used.
