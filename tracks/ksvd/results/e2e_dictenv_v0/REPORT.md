# E2E-DictEnv-v0 — report (STOPPED at Stage-1)

Round `e2e_dictenv_v0`; study `zinc-context-gap`; protocol `e2e_dictenv_v0`.
Official ZINC **test was never loaded**.

## Frozen verdict

```
E2E_DICTENV_MECHANISM_COLLAPSED
```

The preregistered Stage-1 mechanism gate failed, so formal 2-arm training,
mechanism interventions and the performance verdict table were **not** executed
(preregistration §16/§17/§21, case S1).

## Stage-1 mechanism gate

* sub-gates: {'alpha_still_sparse': True, 'atoms_active': False, 'd_grad_nonzero': True, 'effective_atom_count': True, 'environment_rank': True, 'loss_finite': True, 'no_single_atom_dominance': True, 'train_mae_decreased': True}
* active dictionary atoms after the 3-epoch smoke: **23/32** on the official train (all 231,664 atoms), **23/32** on the 512-molecule smoke subset; required ≥ 24
* active dictionary atoms at initialization: 25/32 (official train), 24/32 (512-molecule subset)
* effective atom count (trained): 12.8272
* top-1 support share (trained): 0.1250
* environment effective rank (smoke model): 2.2971
* smoke loss curve: train MAE [1.347, 1.3306, 1.24918]
* reconstruction curve: [0.008435, 0.006535, 0.005322]

Reading: the frozen K-SVD dictionary + tied-IHT mechanism is present atinitialization (≥24/32 atoms active under both scopes) but the 3-epoch train-onlysmoke removes rare-atom support, leaving 23/32 active. This is a genuine,deterministic failure of the preregistered coverage gate, not a loss/reconstructioncollapse (loss decreases, gradients reach `D`, codes stay sparse, no single atomdominates, environment rank > 1).

## What did pass (Gate 0)

* G0..G12 correctness gates: **PASS**
* parameter identity: 66158 (FEC-S1 66170, delta -12)
* lambda_rec (frozen): 135.834928 (L_task^init 1.365938, L_rec^init 0.010056)

## Why no rescue

The preregistration forbids any K/s/IHT/LISTA/dictionary-count/decoder/attention/LayerNorm/λ-sweep/extra-epoch/seed change at this stage. No formal arm was run, so no
performance or dictionary-specificity claim is made.

* commit `e9d424530f4087c57d59089a0cf7844fb0563103`; official_test_loaded = false
