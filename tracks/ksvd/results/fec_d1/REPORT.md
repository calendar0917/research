# FEC-D1 — Localized Sparse Structural Dictionary Binding (seed 0)

Round `fec_d1`; study `zinc-context-gap`; protocol `fec_d1`.
Official ZINC **test was never loaded**.

## Primary result

* matched frozen base `M_B` (FEC-S1 best checkpoint replay): **0.136782**
* Dict32 localized binding soup `M_D`: **0.131537**
* PCA32 localized binding soup `M_P`: **0.130231**
* assignment-shuffle `M_shuffle` (5 permutations): **0.144669**

## Frozen verdict

```
FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC
```

## Gains / gates

* `G_D = M_B - M_D` = **0.005245** (Gate A ≥ 0.003: True)
* `G_dict-specific = M_P - M_D` = **-0.001306** (Gate B ≥ 0.002: False)
* `G_assign = M_shuffle - M_D` = **0.013132** (Gate C ≥ 0.01: True)

## Six mandatory answers

* Q1 exact SDB `R65→K32/s8` reuse: **32** atoms, `s=8`, dict sha `925d573a5808…`
* Q2 dictionary pure-topology, chemistry only at environment formation: correctness G1/G2 passed = True
* Q3 localized Dict gain ≥ 0.003: **True**
* Q4 Dict better than PCA32 by ≥ 0.002: **False**
* Q5 assignment shuffle worsens MAE by ≥ 0.010: **True**
* Q6 verdict: **FEC_D1_LOCAL_BINDING_SUPPORTED_DICTIONARY_NOT_SPECIFIC**

## Mechanism (evaluation only)

* branch neutralization (`Δe=0`) restores base: True (max |Δpred| 9.537e-07)
* shuffle rows: [{'max_abs_pred_shift_vs_clean': 0.8040962219238281, 'mean_abs_pred_shift_vs_clean': 0.06455075733363629, 'seed': 101, 'valid_mae': 0.14683571156678954}, {'max_abs_pred_shift_vs_clean': 0.7491370439529419, 'mean_abs_pred_shift_vs_clean': 0.06221696898341179, 'seed': 202, 'valid_mae': 0.1428325747437193}, {'max_abs_pred_shift_vs_clean': 0.6480158567428589, 'mean_abs_pred_shift_vs_clean': 0.056067990824580194, 'seed': 303, 'valid_mae': 0.1406039369329228}, {'max_abs_pred_shift_vs_clean': 1.0472736358642578, 'mean_abs_pred_shift_vs_clean': 0.06744286181032658, 'seed': 404, 'valid_mae': 0.14745637970784448}, {'max_abs_pred_shift_vs_clean': 0.9098305702209473, 'mean_abs_pred_shift_vs_clean': 0.06446656650304794, 'seed': 505, 'valid_mae': 0.1456165729648783}]

## Historical FEC-S1 anchor (external context only)

* FEC-S1 seed-0 Top-5 soup: **0.13042183499777457**
* `M_D - anchor` = **0.001116**
* new seed-0 strict-static anchor: **False** (official test stays closed)
