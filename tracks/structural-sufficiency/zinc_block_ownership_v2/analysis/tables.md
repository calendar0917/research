# zbo-v2 — tables

generated 2026-10-27T13:15:12+0800 · official test never loaded

## Run-level metrics

| run | best@ep | W20 | T100 | final | GPU·h | mech active |
|---|---:|---:|---:|---:|---:|---|
| N_s0 | 0.081592@359 | 0.083953 | 0.084129 | 0.084134 | 0.586 | True |
| R_s0 | 0.076707@387 | 0.077748 | 0.077827 | 0.077741 | 0.493 | True |
| S_s0 | 0.075762@405 | 0.076588 | 0.076637 | 0.076661 | 0.536 | True |

### Phase B — R vs S, seed 0 (positive = R better)

| quantity | value |
|---|---:|
| ref / other | R_s0 / S_s0 |
| best-valid ref / other | 0.076707 / 0.075762 |
| **d_best** | **-0.000945** |
| d_W20 / d_T100 / d_final | -0.001160 / -0.001190 / -0.001080 |
| paired ΔMAE (mean) | -0.000945 [-0.005963, +0.003847] |
| win rate | 0.482 |
| train@best ref / other | 0.013868 / 0.010859 |

| group | n | ΔMAE | 95% CI |
|---|---:|---:|---|
| ordinary_y_gt_-3 | 935 | +0.000355 | [-0.003049, +0.003873] |
| tail_y_le_-3 | 65 | -0.019643 | [-0.078807, +0.027649] |
| support_all_supported | 384 | -0.001515 | [-0.004591, +0.001752] |
| support_low_support | 365 | -0.004256 | [-0.014848, +0.004290] |
| support_any_unseen | 251 | +0.004742 | [-0.006512, +0.015949] |
| rings_ge_4 | 997 | -0.000774 | [-0.005907, +0.003757] |
| n_multi_1 | 110 | -0.008831 | [-0.039053, +0.013895] |
| n_multi_2 | 476 | -0.002229 | [-0.008422, +0.003151] |
| n_multi_3plus | 413 | +0.003010 | [-0.002589, +0.008423] |
| fixed_worst10 | 10 | -0.137877 | [-0.432580, +0.070576] |
| block_sig_combo_novel | 370 | -0.005984 | [-0.017964, +0.004113] |
| block_sig_combo_seen | 630 | +0.002015 | [-0.001729, +0.005874] |

| drop variant | n | ΔMAE | 95% CI |
|---|---:|---:|---|
| full | 1000 | -0.000945 | [-0.005963, +0.003847] |
| drop_row172 | 999 | +0.000311 | [-0.003846, +0.004316] |
| drop_fixed_worst10 | 990 | +0.000438 | [-0.003750, +0.004254] |
| drop_row172_and_fixed_worst10 | 990 | +0.000438 | [-0.003750, +0.004254] |
| drop_top1_abs_delta | 999 | +0.000311 | [-0.003846, +0.004316] |
| drop_top5_abs_delta | 995 | +0.002046 | [-0.001187, +0.005387] |
| drop_top10_abs_delta | 990 | +0.001216 | [-0.001750, +0.004210] |
| drop_top10_ae_ref | 990 | +0.002521 | [-0.000610, +0.005490] |

secondary (non-gating) DiD vs the n_multi<=1 control: Δ(n_multi>=2) +0.000205 − Δ(n_multi<=1) -0.010153 = **+0.010358** [-0.013416, +0.039782], n=889/111

single-row check: top-1 share 1.329, not_single_row_driven=False, LOO [-0.001247, +0.000311]

**decision**

* checks: {'d_best_ge_0.003': False, 'd_W20_ge_-0.002': True, 'd_T100_ge_-0.002': True, 'not_single_row_driven': False, 'mechanism_active_S': True}
* label: `SINGLETON_CONTENT_EXPLAINS_THE_GAIN`
* proceed_to_phase_c: **False**

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

