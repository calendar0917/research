# Centre-level structure--attribute interaction screen

Protocol: `luyin16-molhiv-cross-center-interaction-screen-v1`

Official validation/test were not encoded or evaluated.

## Invariance audit

- pass: **True**
- maximum drift: 1.788e-07

## Fold scores

| fold | view | valid ROC-AUC |
|---:|---|---:|
| 0 | `s` | 0.660689 |
| 0 | `s_marginal` | 0.695501 |
| 0 | `s_cross_cov` | 0.698365 |
| 0 | `s_binding` | 0.695828 |
| 0 | `s_both` | 0.701253 |
| 0 | `s_cross_cov_shuffled_0` | 0.716927 |
| 0 | `s_binding_shuffled_0` | 0.696808 |
| 0 | `s_cross_cov_shuffled_1` | 0.692111 |
| 0 | `s_binding_shuffled_1` | 0.686459 |
| 0 | `late_marginal_binding` | 0.697913 |
| 1 | `s` | 0.606954 |
| 1 | `s_marginal` | 0.640833 |
| 1 | `s_cross_cov` | 0.660057 |
| 1 | `s_binding` | 0.641322 |
| 1 | `s_both` | 0.670000 |
| 1 | `s_cross_cov_shuffled_0` | 0.646178 |
| 1 | `s_binding_shuffled_0` | 0.634138 |
| 1 | `s_cross_cov_shuffled_1` | 0.635920 |
| 1 | `s_binding_shuffled_1` | 0.654799 |
| 1 | `late_marginal_binding` | 0.642500 |
| 2 | `s` | 0.621631 |
| 2 | `s_marginal` | 0.681511 |
| 2 | `s_cross_cov` | 0.682968 |
| 2 | `s_binding` | 0.685932 |
| 2 | `s_both` | 0.682013 |
| 2 | `s_cross_cov_shuffled_0` | 0.672594 |
| 2 | `s_binding_shuffled_0` | 0.695702 |
| 2 | `s_cross_cov_shuffled_1` | 0.689599 |
| 2 | `s_binding_shuffled_1` | 0.689976 |
| 2 | `late_marginal_binding` | 0.683772 |

## Aggregate

| view | mean AUC | fold std |
|---|---:|---:|
| `s` | 0.629758 | 0.022677 |
| `s_marginal` | 0.672615 | 0.023188 |
| `s_cross_cov` | 0.680463 | 0.015739 |
| `s_binding` | 0.674361 | 0.023709 |
| `s_both` | 0.684422 | 0.012872 |
| `s_cross_cov_shuffled_0` | 0.678566 | 0.029190 |
| `s_binding_shuffled_0` | 0.675549 | 0.029286 |
| `s_cross_cov_shuffled_1` | 0.672543 | 0.025917 |
| `s_binding_shuffled_1` | 0.677078 | 0.015819 |
| `late_marginal_binding` | 0.674728 | 0.023509 |

## Gates

- `cross_cov_vs_marginal`: +0.007848, wins 3/3
- `cross_cov_vs_center_shuffle`: +0.004909, wins 2/3
- `binding_vs_marginal`: +0.001745, wins 3/3
- `binding_vs_patch_shuffle`: -0.001953, wins 1/3
- `both_vs_marginal`: +0.011807, wins 3/3
- `late_vs_marginal`: +0.002113, wins 3/3

The interaction blocks are projected with fold-train-only PCA; shuffled controls preserve graph-local row marginals while breaking the tested correspondence.
