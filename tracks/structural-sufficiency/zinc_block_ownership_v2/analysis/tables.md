# zbo-v2 — tables

generated 2026-10-27T12:38:34+0800 · official test never loaded

## Run-level metrics

| run | best@ep | W20 | T100 | final | GPU·h | mech active |
|---|---:|---:|---:|---:|---:|---|
| N_s0 | 0.081592@359 | 0.083953 | 0.084129 | 0.084134 | 0.586 | True |
| R_s0 | 0.076707@387 | 0.077748 | 0.077827 | 0.077741 | 0.493 | True |

### Reference — R vs N, seed 0 (v1 numbers, re-derived)

| quantity | value |
|---|---:|
| ref / other | R_s0 / N_s0 |
| best-valid ref / other | 0.076707 / 0.081592 |
| **d_best** | **+0.004884** |
| d_W20 / d_T100 / d_final | +0.006204 / +0.006302 / +0.006394 |
| paired ΔMAE (mean) | +0.004884 [-0.000306, +0.011554] |
| win rate | 0.531 |
| train@best ref / other | 0.013868 / 0.016376 |

| group | n | ΔMAE | 95% CI |
|---|---:|---:|---|
| ordinary_y_gt_-3 | 935 | +0.002089 | [-0.001847, +0.006091] |
| tail_y_le_-3 | 65 | +0.045089 | [-0.010717, +0.124434] |
| support_all_supported | 384 | +0.003108 | [-0.001242, +0.008045] |
| support_low_support | 365 | +0.010286 | [+0.000958, +0.024114] |
| support_any_unseen | 251 | -0.000252 | [-0.013395, +0.012873] |
| rings_ge_4 | 997 | +0.004961 | [-0.000204, +0.011587] |
| n_multi_1 | 110 | +0.010359 | [-0.018833, +0.054490] |
| n_multi_2 | 476 | +0.003915 | [-0.002029, +0.009418] |
| n_multi_3plus | 413 | +0.004882 | [-0.001812, +0.011748] |
| fixed_worst10 | 10 | +0.124489 | [-0.226936, +0.636787] |
| block_sig_combo_novel | 370 | +0.005636 | [-0.007059, +0.021336] |
| block_sig_combo_seen | 630 | +0.004443 | [+0.001092, +0.007853] |

| drop variant | n | ΔMAE | 95% CI |
|---|---:|---:|---|
| full | 1000 | +0.004884 | [-0.000306, +0.011554] |
| drop_row172 | 999 | +0.002840 | [-0.001283, +0.006905] |
| drop_fixed_worst10 | 990 | +0.003676 | [+0.000125, +0.007408] |
| drop_row172_and_fixed_worst10 | 990 | +0.003676 | [+0.000125, +0.007408] |
| drop_top1_abs_delta | 999 | +0.002840 | [-0.001283, +0.006905] |
| drop_top5_abs_delta | 995 | +0.002746 | [-0.000658, +0.006103] |
| drop_top10_abs_delta | 990 | +0.002592 | [-0.000429, +0.005664] |
| drop_top10_ae_ref | 990 | +0.004997 | [+0.001486, +0.008772] |

secondary (non-gating) DiD vs the n_multi<=1 control: Δ(n_multi>=2) +0.004364 − Δ(n_multi<=1) +0.009050 = **-0.004686** [-0.050005, +0.023653], n=889/111

single-row check: top-1 share 0.419, not_single_row_driven=True, LOO [+0.002840, +0.005560]

