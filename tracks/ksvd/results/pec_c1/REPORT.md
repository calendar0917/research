# PEC-C1 — Pure Environment Composition Confirmatory (seed 0, official valid)

Verdict: **PURE_ENV_COMPOSITION_ABSOLUTE_WEAK**

> PEC-v0 Gate 1 remains a historical frozen FAIL. PEC-C1 does not retroactively pass or recalibrate it.

Full official train 10,000 / official valid 1,000. `official_test_loaded = false`.

| arm | role coordinate | params (total / trainable) | best valid MAE | best epoch | **Top-5 soup valid MAE** | wall (s) |
|---|---|---:|---:|---:|---:|---:|
| CK | SparseDict (frozen K-SVD `K=16,s=4`) | 94049 / 93633 | 0.169599 | 210 | **0.160971** | 832.4 |
| CD | DenseRole (trainable, D-initialized) | 94049 / 94049 | 0.159782 | 235 | **0.151767** | 765.5 |

`M_CK = 0.160971`, `M_CD = 0.151767`, `delta_dict = M_CD - M_CK = -0.009204` (positive ⇒ SparseDict better).
Absolute band: `>0.145`.
Soup members CK `[191, 198, 210, 220, 238]`, CD `[178, 199, 219, 235, 239]`.

## Frozen gate outcomes

| case | fired |
|---|---|
| `A_absolute_weak` | True |
| `B_dense_dominates_sparse` | True |
| `C_sparse_signal` | False |
| `D_strong_pure` | False |
| `E_viable_dict_unresolved` | False |

`seed1_authorized = false` (none)

## Mechanism integrity (evaluation-only, on the CK Top-5 soup)

| intervention | valid MAE | degradation | mean abs prediction shift |
|---|---:|---:|---:|
| chem_shuffle | 1.737075 | +1.576104 | 1.710692 |
| neutral_dictionary | 2.128386 | +1.967415 | 2.093377 |
| relation_shuffle | 1.553879 | +1.392907 | 1.507191 |

## Historical context only (not gates, never re-run)

```text
strict-static S0 seed0 soup ≈ 0.140794
strict-static S0 seed1 soup ≈ 0.136423
B-Null ≈ 0.123   B-Full ≈ 0.119–0.118 (different computation class)
```

_Confound:_ CK's dictionary is frozen (PEC-C1 D1) while CD's dense role map is trainable (D2): the comparison is conservative against CK, and a Case B verdict must not be generalised to task-coupled dictionaries (that is PEC-C2's question).
