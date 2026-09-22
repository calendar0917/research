# ZINC strict-static dictionary-pair v1 confirmation — results summary

One paired seed-1 confirmation of the v0 seed-0 dictionary-specific signal, with every arm forced to 240 epochs and a shadow early-stopping tracker.

Official ZINC **test was never loaded** in any stage.

## Seed-1 equal-horizon view (primary) — 240 epochs, best official-valid MAE

| arm | best MAE | best epoch | Top-5 soup | params | branch params |
|---|---|---|---|---|---|
| S0 | 0.139389 | 233 | 0.136423 | 66228 | 0 |
| S-Dense | 0.145830 | 164 | 0.141608 | 71417 | 5189 |
| S-Dict | 0.138020 | 240 | 0.134697 | 71429 | 5201 |

* `Gbase_1 = M0 - MK` = +0.001369 (gate 0.004)
* `Gdict_1 = MD - MK` = +0.007811 (gate 0.002)
* equal-horizon soup: `S0 - Dict` +0.001726, `Dense - Dict` +0.006912
* equal-horizon case: **B2_SEED1_DIRECTIONAL_REPLICATION**

## Mechanism check at the seed-1 best checkpoint

| arm | gamma_final | mean residual norm (diag) | ablation mean shift | ablation max shift |
|---|---|---|---|---|
| S0 | +0.100000 | nan | None | None |
| S-Dense | +0.000095 | nan | 9.276717901229859e-07 | 9.5367431640625e-06 |
| S-Dict | +0.000375 | 3.2517451842295486e-08 | 1.0206058621406556e-05 | 0.00010204315185546875 |

A near-zero `gamma_final` means the residual adapter is inert at the best checkpoint (the run is effectively the strict-static base network).

## Seed-1 shadow-protocol view (simulated v0 early stopping)

| arm | best MAE | best epoch | stop epoch | Top-5 soup |
|---|---|---|---|---|
| S0 | 0.139389 | 233 | None | 0.136423 |
| S-Dense | 0.145830 | 164 | 204 | 0.141608 |
| S-Dict | 0.138020 | 240 | None | 0.134697 |

* shadow `Gbase_1` +0.001369, `Gdict_1` +0.007811 -> **B2_SEED1_DIRECTIONAL_REPLICATION**
* equal-horizon and shadow views same direction: True

## Seed-0 common-horizon diagnostic

* `H_common` = 204
* M0c 0.145674 / MDc 0.145958 / MKc 0.139009
* base_gain_common +0.006666, dict_gain_common +0.006949
* verdict: **COMMON_HORIZON_SIGNAL_SURVIVES**

## Two-seed original-protocol paired summary

* mean `S0 - Dict` (best) = +0.005497
* mean `Dense - Dict` (best) = +0.008859
* mean `S0 - Dict` (soup) = +0.004346
* mean `Dense - Dict` (soup) = +0.007871

## Budget

* seed-1 full training runs: 3
* forced full horizon: 240 epochs per arm
* total wall clock: 51.8 min
* seed-0 retrained: false; HPO: none; dictionary-pair kernel: not implemented
* official test loaded: false

