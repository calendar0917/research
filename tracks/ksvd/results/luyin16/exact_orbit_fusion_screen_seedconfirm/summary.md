# luyin16-molhiv-exact-rooted-orbit-fusion-pilot-v1-seed-confirm

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
| `s_hybrid_unbound` | 0.695406 | 0.067713 |
| `s_hybrid_prototype` | 0.711264 | 0.068900 |
| `s_exact_orbit_raw_shuffled_0` | 0.709241 | 0.049043 |
| `s_exact_prototype_shuffled_0` | 0.705430 | 0.037939 |
| `s_hybrid_prototype_shuffled_0` | 0.713078 | 0.047581 |

## Gates

- `exact_structure_vs_wl`: -0.011920, wins 1/3 — **FAIL**
- `exact_unbound_vs_wl_unbound`: -0.030554, wins 0/3 — **FAIL**
- `orbit_raw_vs_exact_unbound`: -0.001870, wins 2/3 — **FAIL**
- `prototype_vs_exact_unbound`: +0.014104, wins 1/3 — **FAIL**
- `hybrid_prototype_vs_hybrid_unbound`: +0.015857, wins 2/3 — **PASS**
- `orbit_raw_true_vs_shuffle`: -0.027147, wins 1/3 — **FAIL**
- `prototype_true_vs_shuffle`: -0.007361, wins 1/3 — **FAIL**
- `hybrid_prototype_true_vs_shuffle`: -0.001814, wins 1/3 — **FAIL**

## Fold representation diagnostics

- fold 0: top-32 train/valid mass 0.9473/0.9367; raw orbit 7730D; prototype 256D.
- fold 1: top-32 train/valid mass 0.9432/0.9406; raw orbit 7505D; prototype 256D.
- fold 2: top-32 train/valid mass 0.9433/0.9411; raw orbit 7664D; prototype 256D.

Decision: **EXACT_ORBIT_BINDING_NO_GO**
