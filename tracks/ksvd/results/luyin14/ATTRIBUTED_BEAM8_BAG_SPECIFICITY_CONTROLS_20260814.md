# Attributed Beam8 frozen BAG specificity matched controls

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_BAG_SPECIFICITY_CONTROLS_PROTOCOL_20260814.md`  
> 判定：`BAG_GAIN_NOT_ESTABLISHED_AS_BEAM8_SPECIFIC`

| variant | balanced accuracy over 18 units |
|---|---:|
| GINE_FROZEN | 0.7331 ± 0.0482 |
| BEAM8_FULL | 0.7500 ± 0.0283 |
| BEAM8_CODE_ONLY | 0.7447 ± 0.0240 |
| BEAM8_HIST_ONLY | 0.7533 ± 0.0458 |
| BEAM8_RANDOM_DICTIONARY | 0.7493 ± 0.0288 |
| GLOBAL_STATS | 0.7867 ± 0.0121 |

## Paired attribution

| comparison | mean | W/T/L | split3/4 | model0/1/2 |
|---|---:|---:|---:|---:|
| full_vs_gine | +0.0168 | 15/0/3 | +0.0259 / +0.0078 | +0.0188 / +0.0114 / +0.0204 |
| full_vs_global_stats | -0.0367 | 0/0/18 | -0.0454 / -0.0280 | -0.0281 / -0.0367 / -0.0453 |
| full_vs_hist_only | -0.0033 | 5/0/13 | +0.0057 / -0.0123 | +0.0018 / -0.0114 / -0.0003 |
| full_vs_random_dictionary | +0.0007 | 9/0/9 | +0.0014 / -0.0000 | +0.0118 / -0.0054 / -0.0043 |
| code_only_vs_gine | +0.0115 | 9/0/9 | +0.0224 / +0.0007 | +0.0083 / +0.0080 / +0.0183 |

## Frozen checks

- full_gain：`True`；
- beats_global_stats：`False`；
- beats_hist_only：`False`；
- beats_random_dictionary：`False`；
- global_both_splits：`False`；
- global_model_majority：`False`；
- parity：`True`；

## Boundary

- All controls use identical frozen GINE states, residual rank and padded input dimension.
- Failure means calibration may be useful but cannot be attributed specifically to Beam8.
