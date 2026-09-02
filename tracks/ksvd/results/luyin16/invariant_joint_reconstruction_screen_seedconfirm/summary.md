# MolHIV invariant joint reconstruction screen

Protocol: `luyin16-molhiv-invariant-joint-reconstruction-screen-v1-seed-confirm`

Official-train scaffold folds only; fixed XGBoost; official validation/test not encoded.

Patch multiset relabel invariance: **PASS**

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `s` | 0.680955 | 0.004569 |
| `s_marginal_mean_std` | 0.681101 | 0.031897 |
| `s_joint_l2` | 0.691971 | 0.031840 |
| `s_bilinear` | 0.708739 | 0.029260 |
| `s_pca_reconstruction` | 0.714520 | 0.028465 |
| `s_ksvd_initial` | 0.732304 | 0.017999 |
| `s_ksvd_final` | 0.718368 | 0.033880 |
| `s_joint_l2_shuffled_0` | 0.690129 | 0.025496 |
| `s_bilinear_shuffled_0` | 0.704701 | 0.025663 |
| `s_pca_reconstruction_shuffled_0` | 0.702035 | 0.022800 |
| `s_ksvd_initial_shuffled_0` | 0.711429 | 0.014188 |
| `s_ksvd_final_shuffled_0` | 0.703737 | 0.014955 |
| `s_joint_l2_shuffled_1` | 0.674945 | 0.035866 |
| `s_bilinear_shuffled_1` | 0.700639 | 0.014220 |
| `s_pca_reconstruction_shuffled_1` | 0.709683 | 0.025738 |
| `s_ksvd_initial_shuffled_1` | 0.713291 | 0.029874 |
| `s_ksvd_final_shuffled_1` | 0.708437 | 0.038532 |

## Gates

- `joint_l2_vs_marginal`: +0.010870, wins 2/3 — **PASS**
- `bilinear_vs_marginal`: +0.027638, wins 3/3 — **PASS**
- `pca_vs_joint_l2`: +0.022549, wins 2/3 — **PASS**
- `ksvd_initial_vs_joint_l2`: +0.040333, wins 3/3 — **PASS**
- `ksvd_final_vs_joint_l2`: +0.026397, wins 3/3 — **PASS**
- `ksvd_final_vs_initial`: -0.013936, wins 1/3 — **FAIL**
- `ksvd_final_vs_pca`: +0.003848, wins 1/3 — **FAIL**
- `joint_l2_vs_shuffle`: +0.009434, wins 3/3 — **PASS**
- `bilinear_vs_shuffle`: +0.006069, wins 2/3 — **PASS**
- `pca_vs_shuffle`: +0.008661, wins 3/3 — **PASS**
- `ksvd_initial_vs_shuffle`: +0.019944, wins 3/3 — **PASS**
- `ksvd_final_vs_shuffle`: +0.012282, wins 3/3 — **PASS**

Decision: **BINDING_AND_GENERIC_RECONSTRUCTION_PASS_KSVD_UPDATE_FAIL**
