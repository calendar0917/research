# SDPK-v0 — static dictionary-conditioned pair kernel: results summary

Question: does promoting the learnable dictionary from a bypassable local residual adapter to the core coordinate of the occurrence-level static pair kernel open a new absolute strict-static MAE band?

Official ZINC **test was never loaded** in any stage.

## Primary metric — fixed Top-5 weight soup (official-valid MAE)

| run | best MAE | best epoch | epochs | soup MAE | params | wall s |
|---|---|---|---|---|---|---|
| S0 seed0 (reference) | 0.145674 | - | - | 0.140794 | 66228 | (historical) |
| SDPK-v0 seed0 | 0.142193 | 209 | 240 | 0.139735 | 74996 | 888.1 |

## Seed-0 performance gate

* soup improvement over S0 seed0: +0.001059
* best improvement over S0 seed0: +0.003481
* gate (soup <= 0.1328 and delta >= 0.008 and best delta >= 0.006): **FAIL**
* verdict: **SDPK_V0_NO_STRONG_PERFORMANCE_SIGNAL**

## Budget

* full training runs used: 1 / 2
* seed1 purchased: False
* matched dense control: not purchased; HPO: none
* official test accessed = false

