# Attributed Beam8 graph-conditioned frozen BAG residual

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md`  
> split/model seed：`2/2`

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| GINE_FROZEN | 0.7484 ± 0.0150 | 0.7574 |
| BAG_RESIDUAL | 0.7641 ± 0.0091 | 0.7694 |

## Paired BAG−GINE

- mean：`+0.0157`；
- W/T/L：`3/0/0`；

## Boundary

- GINE parameters and BatchNorm state remain frozen during residual training.
- No localized TRUE/SHUFFLED field, relation message or KSVD update is used.
