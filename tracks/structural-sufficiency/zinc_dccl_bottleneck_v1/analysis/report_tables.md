# Phase A — machine-generated tables (DCC-L, official valid 1,000)

Full-sample MAE (seed mean) = **0.07599** (per seed 0.07854, 0.07384, 0.07468, 0.07691); median AE 0.0407; p90 AE 0.1381; top-10 rows = 18.5% of total AE, top-100 = 46.0%.

## Table 1 (MECE, 12 cells) — radius-2 support × extreme label × ring complexity

`U` = molecule has ≥1 atom whose exact typed radius-2 signature never occurs in train 10k; `T` = y ≤ −3; `R` = ring count. MAE is the seed mean; shares are of total absolute error.

| U | T | rings | n | n% | MAE | AE share | signed | seeds same sign | per-seed MAE |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| seen | ordinary | r0-1 | 35 | 3.5% | 0.0571 | 2.6% | -0.0115 | no | 0.054 / 0.052 / 0.056 / 0.067 |
| seen | ordinary | r2-3 | 521 | 52.1% | 0.0429 | 29.4% | -0.0059 | yes | 0.043 / 0.043 / 0.043 / 0.043 |
| seen | ordinary | r>=4 | 154 | 15.4% | 0.0518 | 10.5% | -0.0083 | yes | 0.048 / 0.048 / 0.058 / 0.053 |
| seen | y≤-3 | r0-1 | 2 | 0.2% | 0.0503 | 0.1% | +0.0345 | no | 0.046 / 0.068 / 0.037 / 0.050 |
| seen | y≤-3 | r2-3 | 25 | 2.5% | 0.2653 | 8.7% | +0.2289 | yes | 0.291 / 0.272 / 0.180 / 0.318 |
| seen | y≤-3 | r>=4 | 12 | 1.2% | 0.1170 | 1.8% | +0.0182 | no | 0.119 / 0.112 / 0.124 / 0.113 |
| unseen | ordinary | r0-1 | 35 | 3.5% | 0.0994 | 4.6% | +0.0378 | yes | 0.101 / 0.097 / 0.099 / 0.102 |
| unseen | ordinary | r2-3 | 149 | 14.9% | 0.1020 | 20.0% | +0.0303 | yes | 0.106 / 0.105 / 0.099 / 0.098 |
| unseen | ordinary | r>=4 | 41 | 4.1% | 0.2275 | 12.3% | -0.1184 | yes | 0.245 / 0.205 / 0.226 / 0.233 |
| unseen | y≤-3 | r0-1 | 4 | 0.4% | 0.1532 | 0.8% | -0.0072 | no | 0.220 / 0.149 / 0.149 / 0.095 |
| unseen | y≤-3 | r2-3 | 16 | 1.6% | 0.1693 | 3.6% | +0.0324 | yes | 0.169 / 0.123 / 0.218 / 0.168 |
| unseen | y≤-3 | r>=4 | 6 | 0.6% | 0.6957 | 5.5% | +0.6759 | yes | 0.814 / 0.671 / 0.650 / 0.648 |

## Table 2 (MECE, 9 cells) — atom-count tertile (train cuts) × environment support tier

support tier: `full` = every atom's radius-2 signature has ≥5 train molecules; `low4` = none unseen but ≥1 atom with 1–4 train molecules; `unseen` = ≥1 atom with 0.

| size | support | n | MAE | AE share | signed |
|---|---|---:|---:|---:|---:|
| size_q1 | seen_full | 61 | 0.0468 | 3.8% | -0.0148 |
| size_q1 | seen_low4 | 95 | 0.0599 | 7.5% | -0.0005 |
| size_q1 | unseen | 110 | 0.1212 | 17.6% | +0.0450 |
| size_q2 | seen_full | 157 | 0.0321 | 6.6% | -0.0040 |
| size_q2 | seen_low4 | 128 | 0.0965 | 16.2% | +0.0359 |
| size_q2 | unseen | 68 | 0.1103 | 9.9% | +0.0018 |
| size_q3 | seen_full | 165 | 0.0369 | 8.0% | -0.0004 |
| size_q3 | seen_low4 | 143 | 0.0592 | 11.1% | -0.0118 |
| size_q3 | unseen | 73 | 0.2008 | 19.3% | +0.0062 |

## Table 3 (MECE, 8 cells) — radius-2 support × extreme label × size

size split at the train median n_atoms = 25.

| U | T | large | n | MAE | AE share | signed |
|---|---|---|---:|---:|---:|---:|
| seen | ordinary | False | 338 | 0.0467 | 20.8% | -0.0079 |
| seen | ordinary | True | 372 | 0.0445 | 21.8% | -0.0056 |
| seen | y≤-3 | False | 27 | 0.0622 | 2.2% | +0.0246 |
| seen | y≤-3 | True | 12 | 0.5383 | 8.5% | +0.4454 |
| unseen | ordinary | False | 148 | 0.1120 | 21.8% | +0.0296 |
| unseen | ordinary | True | 77 | 0.1485 | 15.0% | -0.0441 |
| unseen | y≤-3 | False | 18 | 0.1889 | 4.5% | +0.0323 |
| unseen | y≤-3 | True | 8 | 0.5119 | 5.4% | +0.4955 |

## Dose response

| mean per-atom unseen fraction | n | MAE | AE share | signed | bias²/MSE |
|---|---:|---:|---:|---:|---:|
| 0 | 749 | 0.0541 | 53.3% | +0.0017 | 0.0001 |
| (0,5%] | 76 | 0.0871 | 8.7% | +0.0336 | 0.0561 |
| (5,10%] | 93 | 0.1020 | 12.5% | +0.0298 | 0.0440 |
| >10% | 82 | 0.2365 | 25.5% | +0.0024 | 0.0000 |

| fraction of atoms with ≥5 train examples | n | MAE | AE share |
|---|---:|---:|---:|
| all_supported | 383 | 0.0365 | 18.4% |
| >=95% | 183 | 0.0530 | 12.8% |
| 80-95% | 322 | 0.0939 | 39.8% |
| <80% | 112 | 0.1971 | 29.0% |

## Confounder control (radius-2 novelty vs absolute error)

| statistic | value |
|---|---:|
| spearman_raw | +0.4370 |
| spearman_partial_n_atoms | +0.4193 |
| spearman_partial_natoms_cyclerank | +0.4153 |
| spearman_partial_natoms_cyclerank_absy | +0.4130 |
| delta_standardised_by_size_ring | +0.0835 |
| per-seed ΔMAE (unseen − seen) | +0.0967 / +0.0823 / +0.0890 / +0.0815 |
| ΔMAE after dropping rows with AE>1 | +0.0736 (n=996) |

### Size × ring strata (unseen − seen), cells with ≥10 molecules on both sides

| size bin | ring bin | n unseen | n seen | MAE unseen | MAE seen | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 37 | 32 | 0.1075 | 0.0599 | +0.0476 |
| 1 | 1 | 73 | 123 | 0.1282 | 0.0533 | +0.0749 |
| 2 | 1 | 57 | 255 | 0.0906 | 0.0614 | +0.0292 |
| 3 | 1 | 35 | 168 | 0.0967 | 0.0405 | +0.0562 |
| 3 | 2 | 38 | 140 | 0.2968 | 0.0554 | +0.2414 |

## Seed structure of the signed error

Per-seed signed-error Pearson correlations (s0..s3): min 0.820, mean 0.891. Fraction of rows where all four seeds share the error sign: 55.9%.

| cell | n | mean signed error per seed | 4/4 same sign | per-row sign agreement |
|---|---:|---|---:|---:|
| seen/ord/r0-1 | 35 | -0.006 / +0.002 / -0.020 / -0.022 | no | 0.63 |
| seen/ord/r2-3 | 521 | -0.000 / -0.004 / -0.009 / -0.011 | yes | 0.51 |
| seen/ord/r>=4 | 154 | -0.010 / -0.004 / -0.014 / -0.005 | yes | 0.60 |
| seen/y≤-3/r0-1 | 2 | +0.043 / +0.068 / +0.037 / -0.010 | no | 0.50 |
| seen/y≤-3/r2-3 | 25 | +0.257 / +0.235 / +0.137 / +0.286 | yes | 0.44 |
| seen/y≤-3/r>=4 | 12 | -0.030 / +0.061 / -0.012 / +0.054 | no | 0.50 |
| unseen/ord/r0-1 | 35 | +0.052 / +0.035 / +0.022 / +0.042 | yes | 0.57 |
| unseen/ord/r2-3 | 149 | +0.033 / +0.039 / +0.033 / +0.016 | yes | 0.62 |
| unseen/ord/r>=4 | 41 | -0.124 / -0.092 / -0.128 / -0.130 | yes | 0.68 |
| unseen/y≤-3/r0-1 | 4 | -0.026 / -0.042 / -0.011 / +0.050 | no | 0.50 |
| unseen/y≤-3/r2-3 | 16 | +0.027 / +0.047 / +0.023 / +0.033 | yes | 0.81 |
| unseen/y≤-3/r>=4 | 6 | +0.814 / +0.648 / +0.650 / +0.592 | yes | 0.83 |

## Pre-fixed target-free support (whole-molecule similarity)

**wl2_idf_cosine** — IDF-weighted WL subtree kernel h=2, cosine, k=10

| quantity | value |
|---|---:|
| knn_reference_mae | +1.1987 |
| median_abs_global | +1.4381 |
| spearman_top1_vs_nn_label_gap | -0.3128 |
| spearman_simtopk_ae | -0.3507 |
| spearman_simtopk_ae_partial | -0.3191 |
| spearman_nnystd_ae | +0.1592 |
| spearman_nnystd_ae_partial | +0.1298 |
| top-1 similarity quantiles (0/10/50/90/100%) | 0.491, 0.685, 0.789, 0.868, 1.000 |

**r2_tanimoto** — Tanimoto on exact typed radius-2 signature sets, k=10

| quantity | value |
|---|---:|
| knn_reference_mae | +0.9911 |
| median_abs_global | +1.4381 |
| spearman_top1_vs_nn_label_gap | -0.1264 |
| spearman_simtopk_ae | -0.4337 |
| spearman_simtopk_ae_partial | -0.3946 |
| spearman_nnystd_ae | +0.1928 |
| spearman_nnystd_ae_partial | +0.1596 |
| top-1 similarity quantiles (0/10/50/90/100%) | 0.107, 0.250, 0.357, 0.485, 1.000 |

### Support × neighbour-label dispersion (terciles, primary similarity)

| support tercile | dispersion tercile | n | MAE | AE share | kNN MAE | mean sim | mean nn y-std | unseen share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 56 | 0.1259 | 9.3% | 1.687 | 0.655 | 0.612 | 43% |
| 1 | 2 | 119 | 0.0888 | 13.9% | 1.601 | 0.628 | 1.061 | 45% |
| 1 | 3 | 158 | 0.1047 | 21.8% | 1.560 | 0.634 | 1.902 | 37% |
| 2 | 1 | 124 | 0.0624 | 10.2% | 0.973 | 0.730 | 0.612 | 29% |
| 2 | 2 | 113 | 0.0577 | 8.6% | 1.041 | 0.733 | 1.026 | 19% |
| 2 | 3 | 96 | 0.1182 | 14.9% | 1.567 | 0.731 | 1.812 | 21% |
| 3 | 1 | 153 | 0.0441 | 8.9% | 0.620 | 0.806 | 0.599 | 13% |
| 3 | 2 | 101 | 0.0483 | 6.4% | 0.986 | 0.798 | 1.011 | 13% |
| 3 | 3 | 80 | 0.0573 | 6.0% | 1.051 | 0.794 | 1.903 | 6% |

### kNN reference vs DCC-L

| group | n | DCC-L MAE | kNN MAE (WL-IDF) | kNN MAE (Tanimoto) |
|---|---:|---:|---:|---:|
| seen+ordinary | 710 | 0.0456 | 0.9184 | 0.8138 |
| unseen+ordinary | 225 | 0.1245 | 1.4132 | 0.9338 |
| seen+tail | 39 | 0.2086 | 3.5483 | 3.1652 |
| unseen+tail | 26 | 0.2883 | 3.4727 | 3.0667 |

## Headroom arithmetic

- total MAE 0.07599; seen (749) 0.05406; unseen (251) 0.14143; unseen share of AE 46.7%
- if the unseen group matched the seen group, total MAE would be 0.05406
- holding the seen group at 0.05406, reaching 0.060 needs unseen = 0.07772

## Coverage projection (zero-GPU, structure only)

- fit=fit4k (4000 molecules): official-valid atom tiers unseen 4.49% / low4 8.63% / supported 86.87%; molecules with any unseen environment 41.6%; locked-dev atom unseen 4.29%
- fit=fit8k (8000 molecules): official-valid atom tiers unseen 2.99% / low4 5.97% / supported 91.04%; molecules with any unseen environment 29.7%; locked-dev atom unseen 2.61%
- fit=train10k (10000 molecules): official-valid atom tiers unseen 2.46% / low4 5.28% / supported 92.26%; molecules with any unseen environment 25.1%; locked-dev atom unseen 0.00%
- fixed-tier-error projection 4k→8k on official valid: 0.09466 → 0.08150 (Δ -0.01317)
