# luyin16-molhiv-exact-topology-conditional-chemistry-pilot-v1

Official-train scaffold folds only; official validation/test were not encoded.

Relabel invariance: **PASS**

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `s` | 0.635649 | 0.047368 |
| `s_wl_structure` | 0.697719 | 0.042243 |
| `s_exact_structure` | 0.685799 | 0.038864 |
| `s_attribute` | 0.636243 | 0.057904 |
| `s_wl_unbound` | 0.714518 | 0.060524 |
| `s_exact_unbound` | 0.683964 | 0.045111 |
| `s_exact_orbit_raw` | 0.682094 | 0.071754 |
| `s_exact_prototype` | 0.698068 | 0.059947 |
| `s_exact_conditional` | 0.689057 | 0.049149 |
| `s_hybrid_unbound` | 0.695406 | 0.067713 |
| `s_hybrid_prototype` | 0.711264 | 0.068900 |
| `s_hybrid_conditional` | 0.702055 | 0.068569 |
| `s_exact_orbit_raw_shuffled_0` | 0.709241 | 0.049043 |
| `s_exact_prototype_shuffled_0` | 0.705430 | 0.037939 |
| `s_hybrid_prototype_shuffled_0` | 0.713078 | 0.047581 |
| `s_exact_conditional_shuffled_0` | 0.686976 | 0.076187 |
| `s_hybrid_conditional_shuffled_0` | 0.711141 | 0.059535 |
| `s_exact_orbit_raw_shuffled_1` | 0.672963 | 0.042865 |
| `s_exact_prototype_shuffled_1` | 0.713855 | 0.070111 |
| `s_hybrid_prototype_shuffled_1` | 0.717017 | 0.067803 |
| `s_exact_conditional_shuffled_1` | 0.643992 | 0.018430 |
| `s_hybrid_conditional_shuffled_1` | 0.666063 | 0.025048 |

## Gates

- `exact_structure_vs_wl`: -0.011920, wins 1/3 — **FAIL**
- `exact_unbound_vs_wl_unbound`: -0.030554, wins 0/3 — **FAIL**
- `orbit_raw_vs_exact_unbound`: -0.001870, wins 2/3 — **FAIL**
- `prototype_vs_exact_unbound`: +0.014104, wins 1/3 — **FAIL**
- `hybrid_prototype_vs_hybrid_unbound`: +0.015857, wins 2/3 — **PASS**
- `conditional_vs_exact_unbound`: +0.005093, wins 1/3 — **FAIL**
- `hybrid_conditional_vs_hybrid_unbound`: +0.006648, wins 2/3 — **PASS**
- `orbit_raw_true_vs_shuffle`: -0.009008, wins 1/3 — **FAIL**
- `prototype_true_vs_shuffle`: -0.011574, wins 1/3 — **FAIL**
- `hybrid_prototype_true_vs_shuffle`: -0.003784, wins 1/3 — **FAIL**
- `conditional_pairing_true_vs_shuffle`: +0.023572, wins 3/3 — **PASS**
- `hybrid_conditional_pairing_true_vs_shuffle`: +0.013453, wins 2/3 — **PASS**

## Fold representation diagnostics

- fold 0: top-32 train/valid mass 0.9473/0.9367; raw orbit 7730D; prototype 256D.
- fold 1: top-32 train/valid mass 0.9432/0.9406; raw orbit 7505D; prototype 256D.
- fold 2: top-32 train/valid mass 0.9433/0.9411; raw orbit 7664D; prototype 256D.

Decision: **PATCH_LEVEL_CONDITIONAL_FUSION_PROMISING**
