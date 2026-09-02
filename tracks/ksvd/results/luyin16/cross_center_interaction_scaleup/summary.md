# Centre-level structure--attribute interaction screen

Protocol: `luyin16-molhiv-cross-center-interaction-scaleup-v1`

Official validation/test were not encoded or evaluated.

Discovery exclusion enabled: **True** (10194 dataset indices).
Selected/discovery overlap: **0**.

## Invariance audit

- pass: **True**
- maximum drift: 1.788e-07

## Fold scores

| fold | view | valid ROC-AUC |
|---:|---|---:|
| 0 | `s` | 0.699011 |
| 0 | `s_marginal` | 0.752883 |
| 0 | `s_cross_cov` | 0.737276 |
| 0 | `s_binding` | 0.757860 |
| 0 | `s_both` | 0.757117 |
| 0 | `s_cross_cov_shuffled_0` | 0.758989 |
| 0 | `s_binding_shuffled_0` | 0.757114 |
| 0 | `s_both_shuffled_0` | 0.758526 |
| 0 | `s_cross_true_binding_shuffled_0` | 0.738426 |
| 0 | `s_cross_shuffled_binding_true_0` | 0.777977 |
| 0 | `s_cross_cov_shuffled_1` | 0.738275 |
| 0 | `s_binding_shuffled_1` | 0.756954 |
| 0 | `s_both_shuffled_1` | 0.746336 |
| 0 | `s_cross_true_binding_shuffled_1` | 0.748900 |
| 0 | `s_cross_shuffled_binding_true_1` | 0.758974 |
| 0 | `s_norm_gate` | 0.749033 |
| 0 | `s_bilinear_gate` | 0.748964 |
| 0 | `s_norm_gate_cross_true_binding_shuffled_0` | 0.747217 |
| 0 | `s_norm_gate_cross_shuffled_binding_true_0` | 0.760231 |
| 0 | `s_norm_gate_both_shuffled_0` | 0.749262 |
| 0 | `s_bilinear_gate_cross_true_binding_shuffled_0` | 0.755321 |
| 0 | `s_bilinear_gate_cross_shuffled_binding_true_0` | 0.753107 |
| 0 | `s_bilinear_gate_both_shuffled_0` | 0.751865 |
| 0 | `s_norm_gate_cross_true_binding_shuffled_1` | 0.750225 |
| 0 | `s_norm_gate_cross_shuffled_binding_true_1` | 0.760239 |
| 0 | `s_norm_gate_both_shuffled_1` | 0.759567 |
| 0 | `s_bilinear_gate_cross_true_binding_shuffled_1` | 0.759487 |
| 0 | `s_bilinear_gate_cross_shuffled_binding_true_1` | 0.758351 |
| 0 | `s_bilinear_gate_both_shuffled_1` | 0.755872 |
| 0 | `late_marginal_binding` | 0.759738 |
| 1 | `s` | 0.675366 |
| 1 | `s_marginal` | 0.682640 |
| 1 | `s_cross_cov` | 0.701843 |
| 1 | `s_binding` | 0.691285 |
| 1 | `s_both` | 0.695734 |
| 1 | `s_cross_cov_shuffled_0` | 0.681979 |
| 1 | `s_binding_shuffled_0` | 0.687787 |
| 1 | `s_both_shuffled_0` | 0.682199 |
| 1 | `s_cross_true_binding_shuffled_0` | 0.694520 |
| 1 | `s_cross_shuffled_binding_true_0` | 0.673812 |
| 1 | `s_cross_cov_shuffled_1` | 0.669189 |
| 1 | `s_binding_shuffled_1` | 0.698374 |
| 1 | `s_both_shuffled_1` | 0.684360 |
| 1 | `s_cross_true_binding_shuffled_1` | 0.689607 |
| 1 | `s_cross_shuffled_binding_true_1` | 0.684585 |
| 1 | `s_norm_gate` | 0.703864 |
| 1 | `s_bilinear_gate` | 0.681946 |
| 1 | `s_norm_gate_cross_true_binding_shuffled_0` | 0.695021 |
| 1 | `s_norm_gate_cross_shuffled_binding_true_0` | 0.701219 |
| 1 | `s_norm_gate_both_shuffled_0` | 0.693667 |
| 1 | `s_bilinear_gate_cross_true_binding_shuffled_0` | 0.682297 |
| 1 | `s_bilinear_gate_cross_shuffled_binding_true_0` | 0.687244 |
| 1 | `s_bilinear_gate_both_shuffled_0` | 0.690460 |
| 1 | `s_norm_gate_cross_true_binding_shuffled_1` | 0.698922 |
| 1 | `s_norm_gate_cross_shuffled_binding_true_1` | 0.692579 |
| 1 | `s_norm_gate_both_shuffled_1` | 0.697811 |
| 1 | `s_bilinear_gate_cross_true_binding_shuffled_1` | 0.693545 |
| 1 | `s_bilinear_gate_cross_shuffled_binding_true_1` | 0.696109 |
| 1 | `s_bilinear_gate_both_shuffled_1` | 0.685687 |
| 1 | `late_marginal_binding` | 0.688753 |
| 2 | `s` | 0.690514 |
| 2 | `s_marginal` | 0.693812 |
| 2 | `s_cross_cov` | 0.700803 |
| 2 | `s_binding` | 0.695382 |
| 2 | `s_both` | 0.707213 |
| 2 | `s_cross_cov_shuffled_0` | 0.703353 |
| 2 | `s_binding_shuffled_0` | 0.699555 |
| 2 | `s_both_shuffled_0` | 0.698112 |
| 2 | `s_cross_true_binding_shuffled_0` | 0.702772 |
| 2 | `s_cross_shuffled_binding_true_0` | 0.696789 |
| 2 | `s_cross_cov_shuffled_1` | 0.696080 |
| 2 | `s_binding_shuffled_1` | 0.698778 |
| 2 | `s_both_shuffled_1` | 0.689237 |
| 2 | `s_cross_true_binding_shuffled_1` | 0.695410 |
| 2 | `s_cross_shuffled_binding_true_1` | 0.692166 |
| 2 | `s_norm_gate` | 0.706149 |
| 2 | `s_bilinear_gate` | 0.704109 |
| 2 | `s_norm_gate_cross_true_binding_shuffled_0` | 0.707897 |
| 2 | `s_norm_gate_cross_shuffled_binding_true_0` | 0.699786 |
| 2 | `s_norm_gate_both_shuffled_0` | 0.700886 |
| 2 | `s_bilinear_gate_cross_true_binding_shuffled_0` | 0.704972 |
| 2 | `s_bilinear_gate_cross_shuffled_binding_true_0` | 0.701173 |
| 2 | `s_bilinear_gate_both_shuffled_0` | 0.710346 |
| 2 | `s_norm_gate_cross_true_binding_shuffled_1` | 0.705348 |
| 2 | `s_norm_gate_cross_shuffled_binding_true_1` | 0.696957 |
| 2 | `s_norm_gate_both_shuffled_1` | 0.697495 |
| 2 | `s_bilinear_gate_cross_true_binding_shuffled_1` | 0.702949 |
| 2 | `s_bilinear_gate_cross_shuffled_binding_true_1` | 0.697718 |
| 2 | `s_bilinear_gate_both_shuffled_1` | 0.705703 |
| 2 | `late_marginal_binding` | 0.696036 |

## Aggregate

| view | mean AUC | fold std |
|---|---:|---:|
| `s` | 0.688297 | 0.009780 |
| `s_marginal` | 0.709778 | 0.030819 |
| `s_cross_cov` | 0.713307 | 0.016954 |
| `s_binding` | 0.714842 | 0.030464 |
| `s_both` | 0.720022 | 0.026646 |
| `s_cross_cov_shuffled_0` | 0.714774 | 0.032460 |
| `s_binding_shuffled_0` | 0.714819 | 0.030290 |
| `s_both_shuffled_0` | 0.712946 | 0.032878 |
| `s_cross_true_binding_shuffled_0` | 0.711906 | 0.019053 |
| `s_cross_shuffled_binding_true_0` | 0.716192 | 0.044684 |
| `s_cross_cov_shuffled_1` | 0.701181 | 0.028434 |
| `s_binding_shuffled_1` | 0.718035 | 0.027520 |
| `s_both_shuffled_1` | 0.706644 | 0.028137 |
| `s_cross_true_binding_shuffled_1` | 0.711305 | 0.026689 |
| `s_cross_shuffled_binding_true_1` | 0.711909 | 0.033424 |
| `s_norm_gate` | 0.719682 | 0.020775 |
| `s_bilinear_gate` | 0.711673 | 0.027878 |
| `s_norm_gate_cross_true_binding_shuffled_0` | 0.716712 | 0.022202 |
| `s_norm_gate_cross_shuffled_binding_true_0` | 0.720412 | 0.028162 |
| `s_norm_gate_both_shuffled_0` | 0.714605 | 0.024683 |
| `s_bilinear_gate_cross_true_binding_shuffled_0` | 0.714197 | 0.030517 |
| `s_bilinear_gate_cross_shuffled_binding_true_0` | 0.713841 | 0.028342 |
| `s_bilinear_gate_both_shuffled_0` | 0.717557 | 0.025582 |
| `s_norm_gate_cross_true_binding_shuffled_1` | 0.718165 | 0.022821 |
| `s_norm_gate_cross_shuffled_binding_true_1` | 0.716591 | 0.030915 |
| `s_norm_gate_both_shuffled_1` | 0.718291 | 0.029187 |
| `s_bilinear_gate_cross_true_binding_shuffled_1` | 0.718660 | 0.029123 |
| `s_bilinear_gate_cross_shuffled_binding_true_1` | 0.717393 | 0.028970 |
| `s_bilinear_gate_both_shuffled_1` | 0.715754 | 0.029521 |
| `late_marginal_binding` | 0.714842 | 0.031885 |

## Gates

- `cross_cov_vs_marginal`: +0.003529, wins 2/3
- `cross_cov_vs_center_shuffle`: +0.005330, wins 2/3
- `binding_vs_marginal`: +0.005064, wins 3/3
- `binding_vs_patch_shuffle`: -0.001585, wins 1/3
- `both_vs_marginal`: +0.010243, wins 3/3
- `both_vs_cross_cov`: +0.006714, wins 2/3
- `both_vs_binding`: +0.005179, wins 2/3
- `both_vs_double_shuffle`: +0.010226, wins 3/3
- `both_vs_cross_true_binding_shuffle`: +0.008416, wins 3/3
- `both_vs_cross_shuffle_binding_true`: +0.005971, wins 2/3
- `late_vs_marginal`: +0.005064, wins 3/3
- `norm_gate_vs_both`: -0.000340, wins 1/3
- `norm_gate_vs_double_shuffle`: +0.003234, wins 2/3
- `norm_gate_vs_cross_true_binding_shuffle`: +0.002243, wins 2/3
- `norm_gate_vs_cross_shuffle_binding_true`: +0.001180, wins 2/3
- `bilinear_gate_vs_both`: -0.008349, wins 0/3
- `bilinear_gate_vs_double_shuffle`: -0.004983, wins 0/3
- `bilinear_gate_vs_cross_true_binding_shuffle`: -0.004756, wins 1/3
- `bilinear_gate_vs_cross_shuffle_binding_true`: -0.003944, wins 1/3

The interaction blocks are projected with fold-train-only PCA; shuffled controls preserve graph-local row marginals while breaking the tested correspondence. `both_vs_double_shuffle` pairs the centre and patch shuffles by repeat.
