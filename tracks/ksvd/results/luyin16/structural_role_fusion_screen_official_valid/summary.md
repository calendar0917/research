# MolHIV clean structural-role fusion screen

Protocol: `luyin16-molhiv-clean-structural-role-fusion-v1-official-valid`

Full all-centre radius-2 induced patches; strict atom attributes exclude OGB degree and is-in-ring. No K-SVD, attention, Optuna, or official test.

## Exact topology audit

- train/valid sampled patches: 1688 / 1875
- exact rooted topology train-to-valid patch coverage: 0.9483
- coarse ambiguous-key fraction: 0.0097
- rooted-WL ambiguous-key fraction: 0.0000

## rooted_wl

Relabel invariance: **PASS**

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `a_strict` | 0.773809 | 0.000000 |
| `t` | 0.758305 | 0.000000 |
| `t_a` | 0.786477 | 0.000000 |
| `f_raw_factorized` | 0.773050 | 0.000000 |
| `f_raw` | 0.768293 | 0.000000 |
| `f_centered` | 0.787179 | 0.000000 |
| `mixed_s` | 0.777459 | 0.000000 |
| `mixed_s_f_raw_factorized` | 0.785524 | 0.000000 |
| `mixed_s_f_raw` | 0.777640 | 0.000000 |
| `mixed_s_f_centered` | 0.786365 | 0.000000 |
| `f_raw_factorized_shuffled_0` | 0.769838 | 0.000000 |
| `f_raw_factorized_shuffled_1` | 0.770630 | 0.000000 |
| `f_raw_shuffled_0` | 0.779157 | 0.000000 |
| `f_raw_shuffled_1` | 0.772338 | 0.000000 |
| `f_centered_shuffled_0` | 0.774177 | 0.000000 |
| `f_centered_shuffled_1` | 0.768900 | 0.000000 |

Gates:

- `factorized_raw_vs_marginals`: -0.013427, wins 0/1 — **FAIL**
- `factorized_raw_vs_shuffle`: +0.002816, wins 1/1 — **FAIL**
- `raw_vs_marginals`: -0.018183, wins 0/1 — **FAIL**
- `raw_vs_shuffle`: -0.007454, wins 0/1 — **FAIL**
- `centered_vs_marginals`: +0.000703, wins 1/1 — **FAIL**
- `centered_vs_shuffle`: +0.015641, wins 1/1 — **PASS**
- `typed_edge_increment`: -0.004757, wins 0/1 — **FAIL**
- `mixed_factorized_increment`: +0.008065, wins 1/1 — **PASS**
- `mixed_increment`: +0.000181, wins 1/1 — **FAIL**
- `mixed_centered_increment`: +0.008906, wins 1/1 — **PASS**

## Decision

- Decision: **RICHER_STRUCTURE_DEPENDENCE_PASS_NO_STABLE_TASK_INCREMENT**
