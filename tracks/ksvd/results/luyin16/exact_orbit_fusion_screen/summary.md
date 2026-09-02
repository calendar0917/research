# luyin16-molhiv-exact-rooted-orbit-fusion-pilot-v1

Official-train scaffold folds only; official validation/test were not encoded.

Relabel invariance: **PASS**

| view | mean validation ROC-AUC | fold std |
|---|---:|---:|
| `s` | 0.645019 | 0.045910 |
| `s_wl_structure` | 0.700871 | 0.050784 |
| `s_exact_structure` | 0.686141 | 0.038979 |
| `s_attribute` | 0.628397 | 0.052425 |
| `s_wl_unbound` | 0.711804 | 0.055266 |
| `s_exact_unbound` | 0.687753 | 0.037910 |
| `s_exact_orbit_raw` | 0.682969 | 0.073181 |
| `s_exact_prototype` | 0.697670 | 0.060978 |
| `s_hybrid_unbound` | 0.708409 | 0.070184 |
| `s_hybrid_prototype` | 0.723443 | 0.075743 |
| `s_exact_orbit_raw_shuffled_0` | 0.711266 | 0.053431 |
| `s_exact_prototype_shuffled_0` | 0.692235 | 0.042820 |
| `s_hybrid_prototype_shuffled_0` | 0.716121 | 0.029485 |

## Gates

- `exact_structure_vs_wl`: -0.014730, wins 1/3 — **FAIL**
- `exact_unbound_vs_wl_unbound`: -0.024051, wins 1/3 — **FAIL**
- `orbit_raw_vs_exact_unbound`: -0.004784, wins 2/3 — **FAIL**
- `prototype_vs_exact_unbound`: +0.009917, wins 1/3 — **FAIL**
- `hybrid_prototype_vs_hybrid_unbound`: +0.015035, wins 3/3 — **PASS**
- `orbit_raw_true_vs_shuffle`: -0.028297, wins 1/3 — **FAIL**
- `prototype_true_vs_shuffle`: +0.005435, wins 1/3 — **FAIL**
- `hybrid_prototype_true_vs_shuffle`: +0.007322, wins 2/3 — **PASS**

## Fold representation diagnostics

- fold 0: top-32 train/valid mass 0.9473/0.9367; raw orbit 7730D; prototype 256D.
- fold 1: top-32 train/valid mass 0.9432/0.9406; raw orbit 7505D; prototype 256D.
- fold 2: top-32 train/valid mass 0.9433/0.9411; raw orbit 7664D; prototype 256D.

Decision: **COMPACT_EXACT_ORBIT_FUSION_PROMISING**
