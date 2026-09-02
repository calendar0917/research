# MolHIV task-aligned interaction screen

Protocol: `luyin16-molhiv-task-aligned-interaction-v1`

Official-train-only nested scaffold audit. Official validation/test are not encoded or evaluated.

## Equal-budget XGBoost tuning

| view | fixed outer AUC | tuned outer AUC | tuned - fixed |
|---|---:|---:|---:|
| `t_a` | 0.697717 | 0.712875 | +0.015158 |
| `f_raw_factorized` | 0.723534 | 0.708383 | -0.015152 |
| `f_centered` | 0.730605 | 0.732944 | +0.002339 |

XGBoost gates:

- `fixed_raw_vs_fixed_marginals`: +0.025817, wins 3/3 — **PASS**
- `fixed_centered_vs_fixed_marginals`: +0.032888, wins 3/3 — **PASS**
- `tuned_raw_vs_tuned_marginals`: -0.004493, wins 1/3 — **FAIL**
- `tuned_centered_vs_tuned_marginals`: +0.020069, wins 3/3 — **PASS**
- `raw_relative_gap_change_after_tuning`: -0.030310, wins 0/3 — **FAIL**
- `centered_relative_gap_change_after_tuning`: -0.012819, wins 0/3 — **FAIL**

## Cross-fitted sparse interaction correction

| source | outer AUC | delta vs tuned T+A |
|---|---:|---:|
| `tuned_t_a` | 0.712875 | +0.000000 |
| `raw` | 0.698604 | -0.014272 |
| `raw_shuffled` | 0.700720 | -0.012155 |
| `centered` | 0.707162 | -0.005713 |
| `centered_shuffled` | 0.677901 | -0.034974 |

Sparse gates:

- `raw_vs_base`: -0.014272, wins 2/3 — **FAIL**
- `raw_vs_shuffle`: -0.002117, wins 1/3 — **FAIL**
- `centered_vs_base`: -0.005713, wins 1/3 — **FAIL**
- `centered_vs_shuffle`: +0.029261, wins 2/3 — **PASS**

Decision: **DENSE_SIGNAL_PERSISTS_BUT_TUNING_IS_NOT_THE_RESCUE**
