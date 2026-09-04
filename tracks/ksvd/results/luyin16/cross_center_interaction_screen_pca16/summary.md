# Centre-level structure--attribute interaction screen

Protocol: `luyin16-molhiv-cross-center-interaction-screen-pca16-v1`

Official validation/test were not encoded or evaluated.

Discovery exclusion enabled: **False** (0 dataset indices).
Selected/discovery overlap: **0**.

## Invariance audit

- pass: **True**
- maximum drift: 1.788e-07

## Fold scores

| fold | view | valid ROC-AUC |
|---:|---|---:|
| 0 | `s` | 0.660689 |
| 0 | `s_marginal` | 0.695501 |
| 0 | `s_cross_cov` | 0.697712 |
| 0 | `s_binding` | 0.694371 |
| 0 | `s_both` | 0.701304 |
| 0 | `s_cross_cov_shuffled_0` | 0.726898 |
| 0 | `s_binding_shuffled_0` | 0.680230 |
| 0 | `s_both_shuffled_0` | 0.725868 |
| 0 | `s_cross_true_binding_shuffled_0` | 0.696431 |
| 0 | `s_cross_shuffled_binding_true_0` | 0.729862 |
| 0 | `s_cross_cov_shuffled_1` | 0.684776 |
| 0 | `s_binding_shuffled_1` | 0.698666 |
| 0 | `s_both_shuffled_1` | 0.681009 |
| 0 | `s_cross_true_binding_shuffled_1` | 0.700575 |
| 0 | `s_cross_shuffled_binding_true_1` | 0.681687 |
| 0 | `late_marginal_binding` | 0.697812 |
| 1 | `s` | 0.606954 |
| 1 | `s_marginal` | 0.640833 |
| 1 | `s_cross_cov` | 0.671810 |
| 1 | `s_binding` | 0.650086 |
| 1 | `s_both` | 0.662787 |
| 1 | `s_cross_cov_shuffled_0` | 0.674339 |
| 1 | `s_binding_shuffled_0` | 0.612989 |
| 1 | `s_both_shuffled_0` | 0.618477 |
| 1 | `s_cross_true_binding_shuffled_0` | 0.626983 |
| 1 | `s_cross_shuffled_binding_true_0` | 0.665517 |
| 1 | `s_cross_cov_shuffled_1` | 0.672500 |
| 1 | `s_binding_shuffled_1` | 0.609741 |
| 1 | `s_both_shuffled_1` | 0.628621 |
| 1 | `s_cross_true_binding_shuffled_1` | 0.636092 |
| 1 | `s_cross_shuffled_binding_true_1` | 0.693736 |
| 1 | `late_marginal_binding` | 0.645086 |
| 2 | `s` | 0.621631 |
| 2 | `s_marginal` | 0.681511 |
| 2 | `s_cross_cov` | 0.691407 |
| 2 | `s_binding` | 0.680732 |
| 2 | `s_both` | 0.693115 |
| 2 | `s_cross_cov_shuffled_0` | 0.687991 |
| 2 | `s_binding_shuffled_0` | 0.688192 |
| 2 | `s_both_shuffled_0` | 0.693090 |
| 2 | `s_cross_true_binding_shuffled_0` | 0.689724 |
| 2 | `s_cross_shuffled_binding_true_0` | 0.692111 |
| 2 | `s_cross_cov_shuffled_1` | 0.668425 |
| 2 | `s_binding_shuffled_1` | 0.689574 |
| 2 | `s_both_shuffled_1` | 0.672720 |
| 2 | `s_cross_true_binding_shuffled_1` | 0.696707 |
| 2 | `s_cross_shuffled_binding_true_1` | 0.667923 |
| 2 | `late_marginal_binding` | 0.681185 |

## Aggregate

| view | mean AUC | fold std |
|---|---:|---:|
| `s` | 0.629758 | 0.022677 |
| `s_marginal` | 0.672615 | 0.023188 |
| `s_cross_cov` | 0.686976 | 0.011029 |
| `s_binding` | 0.675063 | 0.018518 |
| `s_both` | 0.685735 | 0.016567 |
| `s_cross_cov_shuffled_0` | 0.696410 | 0.022268 |
| `s_binding_shuffled_0` | 0.660470 | 0.033732 |
| `s_both_shuffled_0` | 0.679145 | 0.044938 |
| `s_cross_true_binding_shuffled_0` | 0.671046 | 0.031277 |
| `s_cross_shuffled_binding_true_0` | 0.695830 | 0.026400 |
| `s_cross_cov_shuffled_1` | 0.675234 | 0.006950 |
| `s_binding_shuffled_1` | 0.665994 | 0.039949 |
| `s_both_shuffled_1` | 0.660783 | 0.022993 |
| `s_cross_true_binding_shuffled_1` | 0.677791 | 0.029528 |
| `s_cross_shuffled_binding_true_1` | 0.681115 | 0.010546 |
| `late_marginal_binding` | 0.674694 | 0.022009 |

## Gates

- `cross_cov_vs_marginal`: +0.014361, wins 3/3
- `cross_cov_vs_center_shuffle`: +0.001155, wins 1/3
- `binding_vs_marginal`: +0.002448, wins 1/3
- `binding_vs_patch_shuffle`: +0.011831, wins 2/3
- `both_vs_marginal`: +0.013120, wins 3/3
- `both_vs_cross_cov`: -0.001241, wins 2/3
- `both_vs_binding`: +0.010672, wins 3/3
- `both_vs_double_shuffle`: +0.015771, wins 2/3
- `both_vs_cross_true_binding_shuffle`: +0.011317, wins 2/3
- `both_vs_cross_shuffle_binding_true`: -0.002737, wins 1/3
- `late_vs_marginal`: +0.002079, wins 2/3

The interaction blocks are projected with fold-train-only PCA; shuffled controls preserve graph-local row marginals while breaking the tested correspondence. `both_vs_double_shuffle` pairs the centre and patch shuffles by repeat.
