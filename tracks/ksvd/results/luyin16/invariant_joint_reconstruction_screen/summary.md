# MolHIV invariant joint reconstruction screen

Protocol: `luyin16-molhiv-invariant-joint-reconstruction-screen-v1`

Official-train scaffold folds only; fixed XGBoost; official validation/test not encoded.

Patch multiset relabel invariance: **PASS**

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `s` | 0.683676 | 0.004841 |
| `s_marginal_mean_std` | 0.678335 | 0.037645 |
| `s_joint_l2` | 0.686794 | 0.035935 |
| `s_bilinear` | 0.703142 | 0.030881 |
| `s_pca_reconstruction` | 0.720293 | 0.030619 |
| `s_ksvd_initial` | 0.722971 | 0.018298 |
| `s_ksvd_final` | 0.721478 | 0.035488 |
| `s_joint_l2_shuffled_0` | 0.694339 | 0.027742 |
| `s_bilinear_shuffled_0` | 0.703229 | 0.033693 |
| `s_pca_reconstruction_shuffled_0` | 0.702995 | 0.019340 |
| `s_ksvd_final_shuffled_0` | 0.703756 | 0.018682 |
| `s_joint_l2_shuffled_1` | 0.672377 | 0.034805 |
| `s_bilinear_shuffled_1` | 0.695604 | 0.013422 |
| `s_pca_reconstruction_shuffled_1` | 0.714235 | 0.019813 |
| `s_ksvd_final_shuffled_1` | 0.720443 | 0.030822 |

## Gates

- `joint_l2_vs_marginal`: +0.008459, wins 2/3 — **PASS**
- `bilinear_vs_marginal`: +0.024807, wins 3/3 — **PASS**
- `pca_vs_joint_l2`: +0.033499, wins 2/3 — **PASS**
- `ksvd_initial_vs_joint_l2`: +0.036177, wins 3/3 — **PASS**
- `ksvd_final_vs_joint_l2`: +0.034684, wins 3/3 — **PASS**
- `ksvd_final_vs_initial`: -0.001493, wins 2/3 — **FAIL**
- `ksvd_final_vs_pca`: +0.001185, wins 1/3 — **FAIL**
- `joint_l2_vs_shuffle`: +0.003436, wins 2/3 — **PASS**
- `bilinear_vs_shuffle`: +0.003726, wins 2/3 — **PASS**
- `pca_vs_shuffle`: +0.011678, wins 2/3 — **PASS**
- `ksvd_final_vs_shuffle`: +0.009379, wins 2/3 — **PASS**

Decision: **BINDING_AND_GENERIC_RECONSTRUCTION_PASS_KSVD_UPDATE_FAIL**
