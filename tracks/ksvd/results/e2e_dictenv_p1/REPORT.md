# E2E-DictEnv-P1 — report

Round `e2e_dictenv_p1`; study `zinc-context-gap`; protocol `e2e_dictenv_p1`.
Official ZINC test is reporting-only and was not used for any decision before the freeze.

## Frozen verdict

```
P1_CLEAN_ENVIRONMENT_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC
```

## Primary metrics (fixed Top-5 soup official-valid MAE)

* Sparse seed0 `M_S` = **0.131975** (band `strong`)
* zero-code `G_zero` = 0.347116 (gate >= 0.03: True)
* node-shuffle `G_node` = 0.013124 (gate >= 0.01: True)
* all-shuffle `G_all` = 0.072999 (gate >= 0.015: True)
* dictionary health: True
* seed0 dictionary-specific `G_specific` = 0.002558924165554338
* seed1 dictionary-specific `G_specific` = None

## Anchors (context only)

* T1 seed0 soup 0.125765; v0 soup 0.145508; FEC-S1 soup 0.130422
* `M_S - T1` = +0.006210
* `M_S - FEC-S1` = +0.001553

* commit `684675fbff10df54f66c2d71e1e6bbfed7f467c9`; official_test_loaded = false
