# Attributed Beam8 graph-conditioned frozen BAG dual-axis confirmation

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md`  
> 判定：`FROZEN_BAG_ADVANCES_TO_FULL_DUAL_AXIS_MATRIX`

| metric | value |
|---|---:|
| GINE mean | 0.7331 |
| BAG mean | 0.7500 |
| BAG−GINE | +0.0168 ± 0.0249 |
| W/T/L | 15/0/3 |

## Axis means

| axis | deltas |
|---|---:|
| split3/4 | +0.0259 / +0.0078 |
| model0/1/2 | +0.0188 / +0.0114 / +0.0204 |

## Cell means

| split:model | BAG−GINE |
|---|---:|
| 3:0 | +0.0231 |
| 3:1 | +0.0268 |
| 3:2 | +0.0277 |
| 4:0 | +0.0145 |
| 4:1 | -0.0041 |
| 4:2 | +0.0131 |

## Frozen checks

- mean：`True`；
- fold_wins：`True`；
- both_splits：`True`；
- model_majority：`True`；
- worst_model：`True`；
- cell_majority：`True`；

## Boundary

- This route uses only graph-conditioned BAG fields and a frozen GINE backbone.
- Passing only authorizes completion of the remaining split/model grid.
