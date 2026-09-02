# Centre-level structure--attribute interaction screen

Protocol: `luyin16-molhiv-cross-center-interaction-confirmation-v1`

Official validation/test were not encoded or evaluated.

Discovery exclusion enabled: **True** (5121 dataset indices).
Selected/discovery overlap: **0**.

## Invariance audit

- pass: **True**
- maximum drift: 1.192e-07

## Fold scores

| fold | view | valid ROC-AUC |
|---:|---|---:|
| 0 | `s` | 0.687379 |
| 0 | `s_marginal` | 0.761574 |
| 0 | `s_cross_cov` | 0.759910 |
| 0 | `s_binding` | 0.760923 |
| 0 | `s_both` | 0.766372 |
| 0 | `s_cross_cov_shuffled_0` | 0.770592 |
| 0 | `s_binding_shuffled_0` | 0.759283 |
| 0 | `s_both_shuffled_0` | 0.772473 |
| 0 | `s_cross_cov_shuffled_1` | 0.760923 |
| 0 | `s_binding_shuffled_1` | 0.765022 |
| 0 | `s_both_shuffled_1` | 0.750145 |
| 0 | `late_marginal_binding` | 0.762563 |
| 1 | `s` | 0.679224 |
| 1 | `s_marginal` | 0.721236 |
| 1 | `s_cross_cov` | 0.713448 |
| 1 | `s_binding` | 0.718937 |
| 1 | `s_both` | 0.719885 |
| 1 | `s_cross_cov_shuffled_0` | 0.717500 |
| 1 | `s_binding_shuffled_0` | 0.707184 |
| 1 | `s_both_shuffled_0` | 0.701379 |
| 1 | `s_cross_cov_shuffled_1` | 0.704368 |
| 1 | `s_binding_shuffled_1` | 0.713678 |
| 1 | `s_both_shuffled_1` | 0.709425 |
| 1 | `late_marginal_binding` | 0.721236 |
| 2 | `s` | 0.694246 |
| 2 | `s_marginal` | 0.695326 |
| 2 | `s_cross_cov` | 0.713762 |
| 2 | `s_binding` | 0.703489 |
| 2 | `s_both` | 0.704996 |
| 2 | `s_cross_cov_shuffled_0` | 0.708437 |
| 2 | `s_binding_shuffled_0` | 0.702886 |
| 2 | `s_both_shuffled_0` | 0.708738 |
| 2 | `s_cross_cov_shuffled_1` | 0.686987 |
| 2 | `s_binding_shuffled_1` | 0.688896 |
| 2 | `s_both_shuffled_1` | 0.702836 |
| 2 | `late_marginal_binding` | 0.699520 |

## Aggregate

| view | mean AUC | fold std |
|---|---:|---:|
| `s` | 0.686950 | 0.006140 |
| `s_marginal` | 0.726045 | 0.027259 |
| `s_cross_cov` | 0.729040 | 0.021829 |
| `s_binding` | 0.727783 | 0.024267 |
| `s_both` | 0.730418 | 0.026140 |
| `s_cross_cov_shuffled_0` | 0.732176 | 0.027415 |
| `s_binding_shuffled_0` | 0.723118 | 0.025633 |
| `s_both_shuffled_0` | 0.727530 | 0.031921 |
| `s_cross_cov_shuffled_1` | 0.717426 | 0.031565 |
| `s_binding_shuffled_1` | 0.722532 | 0.031703 |
| `s_both_shuffled_1` | 0.720802 | 0.020922 |
| `late_marginal_binding` | 0.727773 | 0.026149 |

## Gates

- `cross_cov_vs_marginal`: +0.002995, wins 1/3
- `cross_cov_vs_center_shuffle`: +0.004239, wins 2/3
- `binding_vs_marginal`: +0.001738, wins 1/3
- `binding_vs_patch_shuffle`: +0.004958, wins 2/3
- `both_vs_marginal`: +0.004373, wins 2/3
- `both_vs_double_shuffle`: +0.006252, wins 2/3
- `late_vs_marginal`: +0.001728, wins 2/3

The interaction blocks are projected with fold-train-only PCA; shuffled controls preserve graph-local row marginals while breaking the tested correspondence. `both_vs_double_shuffle` pairs the centre and patch shuffles by repeat.
