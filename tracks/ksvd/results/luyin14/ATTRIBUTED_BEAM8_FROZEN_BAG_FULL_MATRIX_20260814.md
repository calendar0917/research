# Attributed Beam8 graph-conditioned frozen BAG full matrix

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_FULL_MATRIX_PROTOCOL_20260814.md`  
> 判定：`FROZEN_BAG_CONFIRMED_AS_STABLE_BEAM8_CLASSIFICATION_INTERFACE`

| metric | value |
|---|---:|
| GINE mean | 0.7358 |
| GINE std | 0.0457 |
| BAG mean | 0.7510 |
| BAG std | 0.0283 |
| BAG−GINE | +0.0152 ± 0.0293 |
| median BAG−GINE | +0.0079 |
| W/T/L | 36/0/9 |

## Split means

| split seed | BAG−GINE |
|---:|---:|
| 0 | +0.0175 |
| 1 | +0.0142 |
| 2 | +0.0104 |
| 3 | +0.0259 |
| 4 | +0.0078 |

## Model means

| model seed | BAG−GINE |
|---:|---:|
| 0 | +0.0118 |
| 1 | +0.0070 |
| 2 | +0.0266 |

## Cell means

| split:model | BAG−GINE |
|---|---:|
| 0:0 | +0.0178 |
| 0:1 | -0.0127 |
| 0:2 | +0.0474 |
| 1:0 | -0.0040 |
| 1:1 | +0.0174 |
| 1:2 | +0.0292 |
| 2:0 | +0.0078 |
| 2:1 | +0.0078 |
| 2:2 | +0.0157 |
| 3:0 | +0.0231 |
| 3:1 | +0.0268 |
| 3:2 | +0.0277 |
| 4:0 | +0.0145 |
| 4:1 | -0.0041 |
| 4:2 | +0.0131 |

## Post-hoc stability diagnosis

- corr(GINE, BAG−GINE)：`-0.8010`；
- GINE≥0.74：24 units，mean delta `-0.0003`，W/L `16/8`；
- weakest 5 GINE units：GINE `0.6387` → BAG `0.7084`，delta `+0.0697`；

## Frozen checks

- mean：`True`；
- fold_wins：`True`；
- split_majority：`True`；
- model_majority：`True`；
- worst_split：`True`；
- worst_model：`True`；
- cell_majority：`True`；

## Boundary

- The stable interface is graph-conditioned BAG calibration on a frozen GINE.
- The post-hoc diagnosis characterizes variance reduction and does not alter the frozen gate.
- This result does not revive localized binding, atom gates, cross-attention or ordinary KSVD updates.
