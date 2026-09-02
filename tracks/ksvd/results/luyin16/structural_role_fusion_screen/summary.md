# MolHIV clean structural-role fusion screen

Protocol: `luyin16-molhiv-clean-structural-role-fusion-v1`

Full all-centre radius-2 induced patches; strict atom attributes exclude OGB degree and is-in-ring. No K-SVD, attention, Optuna, or official test.

## Exact topology audit

- train/valid sampled patches: 1701 / 1505
- exact rooted topology train-to-valid patch coverage: 0.9794
- coarse ambiguous-key fraction: 0.0000
- rooted-WL ambiguous-key fraction: 0.0000

## coarse

Relabel invariance: **PASS**

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `a_strict` | 0.658176 | 0.058267 |
| `t` | 0.660821 | 0.057513 |
| `t_a` | 0.675871 | 0.054017 |
| `f_raw_factorized` | 0.678091 | 0.049693 |
| `f_raw` | 0.680662 | 0.049265 |
| `f_centered` | 0.705013 | 0.057056 |
| `mixed_s` | 0.674877 | 0.061109 |
| `mixed_s_f_raw_factorized` | 0.683395 | 0.068768 |
| `mixed_s_f_raw` | 0.681941 | 0.055857 |
| `mixed_s_f_centered` | 0.706149 | 0.053684 |
| `f_raw_shuffled_0` | 0.663935 | 0.076985 |
| `f_raw_factorized_shuffled_0` | 0.671941 | 0.069851 |
| `f_centered_shuffled_0` | 0.635668 | 0.061273 |
| `f_raw_shuffled_1` | 0.676749 | 0.047483 |
| `f_raw_factorized_shuffled_1` | 0.679370 | 0.033209 |
| `f_centered_shuffled_1` | 0.628185 | 0.018757 |

Gates:

- `factorized_raw_vs_marginals`: +0.002220, wins 1/3 — **FAIL**
- `factorized_raw_vs_shuffle`: +0.002436, wins 1/3 — **FAIL**
- `raw_vs_marginals`: +0.004791, wins 1/3 — **FAIL**
- `raw_vs_shuffle`: +0.010320, wins 2/3 — **PASS**
- `centered_vs_marginals`: +0.029142, wins 3/3 — **PASS**
- `centered_vs_shuffle`: +0.073087, wins 3/3 — **PASS**
- `typed_edge_increment`: +0.002571, wins 1/3 — **FAIL**
- `mixed_factorized_increment`: +0.008519, wins 2/3 — **PASS**
- `mixed_increment`: +0.007065, wins 1/3 — **FAIL**
- `mixed_centered_increment`: +0.031272, wins 3/3 — **PASS**

## rooted_wl

Relabel invariance: **PASS**

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `a_strict` | 0.658176 | 0.058267 |
| `t` | 0.694088 | 0.036054 |
| `t_a` | 0.695939 | 0.046691 |
| `f_raw_factorized` | 0.732742 | 0.040457 |
| `f_raw` | 0.716389 | 0.041867 |
| `f_centered` | 0.725659 | 0.046851 |
| `mixed_s` | 0.674877 | 0.061109 |
| `mixed_s_f_raw_factorized` | 0.730956 | 0.042303 |
| `mixed_s_f_raw` | 0.710924 | 0.040465 |
| `mixed_s_f_centered` | 0.720394 | 0.050453 |
| `f_raw_shuffled_0` | 0.699470 | 0.058946 |
| `f_raw_factorized_shuffled_0` | 0.706683 | 0.060896 |
| `f_centered_shuffled_0` | 0.648409 | 0.069863 |
| `f_raw_shuffled_1` | 0.698714 | 0.055524 |
| `f_raw_factorized_shuffled_1` | 0.710804 | 0.061837 |
| `f_centered_shuffled_1` | 0.668268 | 0.050384 |

Gates:

- `factorized_raw_vs_marginals`: +0.036802, wins 3/3 — **PASS**
- `factorized_raw_vs_shuffle`: +0.023998, wins 2/3 — **PASS**
- `raw_vs_marginals`: +0.020450, wins 3/3 — **PASS**
- `raw_vs_shuffle`: +0.017297, wins 2/3 — **PASS**
- `centered_vs_marginals`: +0.029719, wins 3/3 — **PASS**
- `centered_vs_shuffle`: +0.067321, wins 3/3 — **PASS**
- `typed_edge_increment`: -0.016352, wins 0/3 — **FAIL**
- `mixed_factorized_increment`: +0.056079, wins 3/3 — **PASS**
- `mixed_increment`: +0.036047, wins 3/3 — **PASS**
- `mixed_centered_increment`: +0.045518, wins 3/3 — **PASS**

## Cross-schema decision

- rooted-WL factorized raw fusion minus coarse: +0.054651, wins 3/3 — **PASS**
- Decision: **ROOTED_WL_CLEAN_FUSION_PASS**
