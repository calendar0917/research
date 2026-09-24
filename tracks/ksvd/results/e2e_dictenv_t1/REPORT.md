# E2E-DictEnv-T1 — report

Round `e2e_dictenv_t1`; study `zinc-context-gap`.  Official ZINC test is
loaded **only** at the terminal reporting stage, after the freeze.

## Frozen verdict

```
E2E_DICTENV_TUNED_MECHANISM_LOST
```

## Development result (valid-guided tuning, seed 0)

* tuned Sparse soup valid MAE = **0.125765** (band **strong**)
* candidate A2_COARSE146_SLOT48, lambda scale 0.25, horizon 240, winner `stage_b_lambda025`
* improvement vs v0 `0.145508`: **+0.019743**

## Confirmation result (frozen config, seed 1)

* Sparse soup = **0.125496**
* DenseTied soup = **0.131315**
* `G_sparse^seed1 = +0.005819` (gate 0.003: True)
* seed-1 Sparse viable (<= 0.145): True

## Mechanism (frozen Sparse soup)

* zero-code `G_dict-use = +0.017903` (True)
* assignment shuffle `G_assign = +0.007628` (False)
* dictionary health: True (active train/valid 29/29, initial retention 0.960)

## Historical anchors (context only)

* FEC-S1 seed-0 soup 0.130422; v0 Sparse soup 0.145508
* commit `3c13c816cede53e5e03d5933fdada5e908561201`; official_test_loaded = False (terminal read separated)
