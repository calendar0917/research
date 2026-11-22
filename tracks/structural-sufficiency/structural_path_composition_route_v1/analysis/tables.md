# PSC result tables (auto-generated)

## D0 held-out whole-molecule accuracy

| arm | input | readout | target | seeds | RMSE mean ± sd | MAE | RMSE ACRA_REF − arm [95% CI] |
|---|---|---|---|---|---|---|---|
| PSC_SUM | ORDER | SUM | MOLECULE | 1 | 0.7865 ± 0.0000 | 0.5999 |  |
| PSC_SUM_PAIR | ORDER | SUM | PAIR | 1 | 0.7698 ± 0.0000 | 0.6143 |  |
| PATH_COUNT_RIDGE | – | – | – | 1 | 0.6167 ± 0.0000 | 0.4647 |  |
| PATH_COUNT_KRR | – | – | – | 1 | 0.6550 ± 0.0000 | 0.4952 |  |
| MORGAN_COUNT_KRR | – | – | – | 1 | 0.6593 ± 0.0000 | 0.5204 |  |
| COUNT_ONLY_RIDGE | – | – | – | 1 | 1.1614 ± 0.0000 | 0.9016 |  |

Reference for the last column: ACRA_REF. Positive Δ means the arm is better than the reference.

## Pre-specified contrasts, D0 (paired molecule bootstrap)

| question | candidate vs reference | Δ RMSE [95% CI] | p |
|---|---|---|---|
| Q4 PAIR coupling vs MOLECULE (SUM, ORDER) | PSC_SUM_PAIR vs PSC_SUM | 0.0138 [-0.0008, 0.0309] | 0.070 |

Scaffold-cluster bootstrap (D0, one resample per Bemis–Murcko scaffold cluster, same statistic):

| contrast | Δ RMSE [95% CI] | clusters |
|---|---|---|
| PSC_SUM_PAIR_vs_PSC_SUM | 0.0138 [-0.0174, 0.0458] | 231 |


## Pair-level metrics, D0

### By matched-pair scope (unique unordered pairs, unweighted)

| scope | model | pairs scored | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|
| fit_fit | MORGAN_COUNT_KRR | 0 | – | – | – |
| fit_fit | PATH_COUNT_KRR | 0 | – | – | – |
| fit_fit | PSC_SUM | 2456 | 0.4568 | 0.6152 | 0.775 |
| fit_heldout | MORGAN_COUNT_KRR | 0 | – | – | – |
| fit_heldout | PATH_COUNT_KRR | 0 | – | – | – |
| fit_heldout | PSC_SUM | 350 | 0.5643 | 0.7217 | 0.756 |
| heldout_heldout | MORGAN_COUNT_KRR | 413 | 0.4762 | 0.6329 | 0.751 |
| heldout_heldout | PATH_COUNT_KRR | 413 | 0.4620 | 0.6179 | 0.756 |
| heldout_heldout | PSC_SUM | 413 | 0.5173 | 0.7023 | 0.768 |

### By support class (heldout × heldout, cut-row weighted)

| support class | pairs | weight | model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|---|
| T_seen | 195 | 173.0 | MORGAN_COUNT_KRR | 0.4649 | 0.6385 | 0.742 |
| T_seen | 195 | 173.0 | PATH_COUNT_KRR | 0.4306 | 0.5956 | 0.799 |
| T_seen | 195 | 173.0 | PSC_SUM | 0.4795 | 0.6687 | 0.784 |
| T_unseen_both_frags_unseen | 24 | 19.4 | MORGAN_COUNT_KRR | 0.4068 | 0.4889 | 0.625 |
| T_unseen_both_frags_unseen | 24 | 19.4 | PATH_COUNT_KRR | 0.4258 | 0.5416 | 0.583 |
| T_unseen_both_frags_unseen | 24 | 19.4 | PSC_SUM | 0.3906 | 0.5231 | 0.625 |
| T_unseen_frag_both_seen | 156 | 135.8 | MORGAN_COUNT_KRR | 0.4924 | 0.6557 | 0.799 |
| T_unseen_frag_both_seen | 156 | 135.8 | PATH_COUNT_KRR | 0.4666 | 0.6260 | 0.818 |
| T_unseen_frag_both_seen | 156 | 135.8 | PSC_SUM | 0.5474 | 0.7452 | 0.831 |
| T_unseen_one_frag_unseen | 100 | 84.8 | MORGAN_COUNT_KRR | 0.4891 | 0.6133 | 0.750 |
| T_unseen_one_frag_unseen | 100 | 84.8 | PATH_COUNT_KRR | 0.5271 | 0.6639 | 0.650 |
| T_unseen_one_frag_unseen | 100 | 84.8 | PSC_SUM | 0.5753 | 0.7335 | 0.690 |

### Novelty (bag_full) — fit_fit, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2456 | 397 | PSC_SUM | 0.4568 | 0.6152 |

### Novelty (order_full) — fit_fit, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2456 | 397 | PSC_SUM | 0.4568 | 0.6152 |

### Novelty (order_len3_L) — fit_fit, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2456 | 397 | PSC_SUM | 0.4568 | 0.6152 |

### Novelty (bag_full) — heldout_heldout, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 21 | 11 | MORGAN_COUNT_KRR | 0.5476 | 0.7493 |
| (0.25,0.5] | 21 | 11 | PATH_COUNT_KRR | 0.5984 | 0.8471 |
| (0.25,0.5] | 21 | 11 | PSC_SUM | 0.4293 | 0.6162 |
| (0.5,1.0] | 1 | 1 | MORGAN_COUNT_KRR | 0.5346 | 0.5346 |
| (0.5,1.0] | 1 | 1 | PATH_COUNT_KRR | 0.8241 | 0.8241 |
| (0.5,1.0] | 1 | 1 | PSC_SUM | 0.6227 | 0.6227 |
| eta=0 | 212 | 44 | MORGAN_COUNT_KRR | 0.4287 | 0.5621 |
| eta=0 | 212 | 44 | PATH_COUNT_KRR | 0.3985 | 0.5272 |
| eta=0 | 212 | 44 | PSC_SUM | 0.4465 | 0.5892 |

### Novelty (order_full) — heldout_heldout, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 35 | 15 | MORGAN_COUNT_KRR | 0.5047 | 0.6977 |
| (0.25,0.5] | 35 | 15 | PATH_COUNT_KRR | 0.5421 | 0.7680 |
| (0.25,0.5] | 35 | 15 | PSC_SUM | 0.4496 | 0.6331 |
| (0.5,1.0] | 2 | 2 | MORGAN_COUNT_KRR | 0.3998 | 0.4219 |
| (0.5,1.0] | 2 | 2 | PATH_COUNT_KRR | 0.4605 | 0.5868 |
| (0.5,1.0] | 2 | 2 | PSC_SUM | 0.3374 | 0.4418 |
| eta=0 | 136 | 35 | MORGAN_COUNT_KRR | 0.4294 | 0.5774 |
| eta=0 | 136 | 35 | PATH_COUNT_KRR | 0.3814 | 0.5286 |
| eta=0 | 136 | 35 | PSC_SUM | 0.4313 | 0.5831 |

### Novelty (order_len3_L) — heldout_heldout, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 17 | 9 | MORGAN_COUNT_KRR | 0.4560 | 0.6961 |
| (0.25,0.5] | 17 | 9 | PATH_COUNT_KRR | 0.4960 | 0.7592 |
| (0.25,0.5] | 17 | 9 | PSC_SUM | 0.4870 | 0.6800 |
| (0.5,1.0] | 2 | 2 | MORGAN_COUNT_KRR | 0.5447 | 0.5448 |
| (0.5,1.0] | 2 | 2 | PATH_COUNT_KRR | 0.7897 | 0.7905 |
| (0.5,1.0] | 2 | 2 | PSC_SUM | 0.4352 | 0.4739 |
| eta=0 | 338 | 63 | MORGAN_COUNT_KRR | 0.4789 | 0.6352 |
| eta=0 | 338 | 63 | PATH_COUNT_KRR | 0.4502 | 0.6005 |
| eta=0 | 338 | 63 | PSC_SUM | 0.5112 | 0.6960 |

### |Δ| ≥ 0.25 sensitivity (D0 heldout pairs)

| model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|
| MORGAN_COUNT_KRR | 0.5425 | 0.7042 | 0.838 |
| PATH_COUNT_KRR | 0.5421 | 0.6970 | 0.835 |
| PSC_SUM | 0.6121 | 0.7865 | 0.880 |

Counts: {"components_all": 458, "direction_items_removed": 3, "scope_counts": {"fit_fit": 2456, "fit_heldout": 350, "heldout_heldout": 413}, "unique_pairs": 3219, "unknown_fraction": {"atoms": 0.0, "bonds": 0.0, "n_atoms": 90514.0, "n_bonds": 98561.0}}

## Pair-level metrics, D1

### By matched-pair scope (unique unordered pairs, unweighted)

| scope | model | pairs scored | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|

### By support class (heldout × heldout, cut-row weighted)

| support class | pairs | weight | model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|---|

### Novelty (bag_full) — fit_fit, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|

### Novelty (order_full) — fit_fit, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|

### Novelty (order_len3_L) — fit_fit, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|

### Novelty (bag_full) — heldout_heldout, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|

### Novelty (order_full) — heldout_heldout, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|

### Novelty (order_len3_L) — heldout_heldout, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|

### |Δ| ≥ 0.25 sensitivity (D1 heldout pairs)

| model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|

Counts: {"components_all": 458, "direction_items_removed": 5, "scope_counts": {"fit_fit": 2501, "fit_heldout": 444, "heldout_heldout": 274}, "unique_pairs": 3219, "unknown_fraction": {"atoms": 0.0, "bonds": 0.0, "n_atoms": 90514.0, "n_bonds": 98561.0}}

## Pair-residual decomposition (D0 heldout × heldout, component weights)

| model | E[(e_b−e_a)²] | E[e_a²] | E[e_b²] | E[e_a e_b] | corr(e_a,e_b) | identity err |
|---|---|---|---|---|---|---|
| PSC_SUM | 0.3818 | 0.8381 | 0.5890 | 0.5226 | 0.608 | 1.1e-16 |


## Molecule-size strata (D0 dev, seed 0) and worst-error sensitivity

| model | overall RMSE | MAE | Q1 | Q2 | Q3 | Q4 | top-5 removed |
|---|---|---|---|---|---|---|---|
| COUNT_ONLY_RIDGE | 1.1614 | 0.9016 | 1.1849 | 1.1977 | 0.9664 | 1.2865 | 1.1174 |
| PSC_SUM | 0.7865 | 0.5999 | 0.8244 | 0.8557 | 0.6484 | 0.8313 | 0.7556 |

Quartile edges (fit atom counts): [21.0, 27.0, 32.0]

| model (union of 8 worst removed) | RMSE |
|---|---|
| COUNT_ONLY_RIDGE | 1.1054 |
| PSC_SUM | 0.7441 |


## Compute budget

| item | value |
|---|---|
| GPU-hours (all launched jobs) | 0.144 |
| GPU-hours (completed runs) | 0.144 |
| train-loop hours | 0.134 |
| train-loop fraction of GPU wall | 0.927 |
| CPU-hours (closed-form baselines) | 0.039 |
| budget | 16 GPU-hours |
