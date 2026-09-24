# E2E-DictEnv-v0 — report

Round `e2e_dictenv_v0`; study `zinc-context-gap`; protocol `e2e_dictenv_v0`.
Official ZINC **test was never loaded**.

## Frozen verdict

```
E2E_DICTENV_ABSOLUTE_WEAK
```

## Primary metrics (fixed Top-5 soup official-valid MAE)

* SparseDictEnv `M_S` = **0.145508** (band **weak**; best 0.150982 @ 228)
* DenseTiedEnv `M_D` = **0.313047** (best 0.317686 @ 146)
* dictionary-specific `G_sparse = M_D - M_S` = **0.167539** (gate ≥ 0.003: True)
* zero-code `M_zero` = 1.000347, `G_dict-use` = 0.854839 (gate ≥ 0.01: True)
* assignment shuffle `M_shuffle` = 0.162797, `G_assign` = 0.017289 (gate ≥ 0.01: True)
* dictionary health: PASS

## Learning dynamics (240 epochs, no early stop)

* Sparse: soup members [223, 228, 234, 235, 236], member MAEs [0.153624, 0.150982, 0.152585, 0.15394, 0.152502]
* Sparse: train MAE at best 0.115179, train minimum 0.111971, valid reconstruction 0.000097
* Dense : soup members [146, 156, 170, 185, 212], member MAEs [0.317686, 0.320195, 0.320014, 0.32083, 0.321133]
* Dense : train MAE at best 0.258287, train minimum 0.237859, valid reconstruction 0.000008
* dictionary movement (soup vs K-SVD init, Frobenius): Sparse 5.940989, Dense 4.342202
* wall clock: Sparse 1586.7 s, Dense 1272.1 s; peak GPU: 168.33740234375 / 155.68212890625 MB

## Dictionary health (trained Sparse soup)

* active atoms: train 27/32, valid 27/32
* effective atom count: train 13.86, valid 13.89
* support entropy (valid) 2.6311, top-1 share 0.1250, top-8 share 0.7733
* exact top-8 fraction: train 1.000000, valid 1.000000
* usage Spearman train-valid 0.998167
* effective rank 10.8291, coherence max 0.862283, mean 0.174930
* movement from K-SVD init (Frobenius) 5.940990
* reconstruction (normalized): train 0.000096, valid 0.000097
* task gradient to D at trained state 0.087363

## Historical anchors (context only)

* FEC-S1 seed-0 soup 0.130422; `M_S - FEC_S1_soup` = +0.015086
* FEC-S1 seed-0 best 0.136783; S0 seed-0 soup 0.140794
* FEC-S1 is context only; the causal comparison is Sparse vs DenseTied

## Provenance

* lambda_rec (frozen, both arms) 135.834921 from L_task^init 1.365938 / L_rec^init 0.010056
* parameters 66158 (FEC-S1 66170, delta -12)
* correctness gates all passed: True; Stage-1 smoke all passed: False (atoms_active override: True)
* commit `eeeb6b34f641260a0373a3daeda1286a855717d7`; official_test_loaded = false

## Mechanism

* zero-code prediction shift: mean 0.981461, max 4.538126
* shuffle prediction shift: mean 0.048935, max 0.393672
