# E2E-DictEnv-Purify-v0 — report

Frozen verdict: **REFERENCE_REPRODUCTION_FAILURE**

Structure basis -> attribute valuation -> static composition.  Official ZINC test was **never** loaded.

## Stage 0 — semantic refactor equivalence

* deterministic CPU path: max |old - new| = 0.000e+00 (frozen tolerance 1e-6, prediction 0.000e+00, bit-identical: True)
* cuda:0: prediction max |old - new| = 8.345e-07, intermediate max |old - new| = 3.815e-06 at a measured same-implementation rerun noise floor of 5.722e-06 (passed: True)
* gate (all checked devices): True
* parameter ledger: reference 97487 -> purified 95711 (-1776, -1.822 %)

## Primary result (Top-5 soup, official valid)

| seed | reference | purified | Delta |
|---|---:|---:|---:|
| 0 | 0.129878 | 0.135497 | +0.005619 |

* seed-0 case: `performance_failure`; seed 1 authorized: False; seed 1 run: False
* fresh reference reproduction drift vs historical H1 context = +0.006330 (tolerance 0.005)

## Diagnosis (pre-registration 10.2)

The fresh reference is +0.006330 above the historical H1 context, so the round stopped before the candidate was interpreted.  The drift is **not** a data, dictionary, initialization, evaluation or code change:

* same data cache (written before H1's run) and identical dictionary sha256,
* no commit touched the P2 model/training module since H1's commit,
* `build_model` seeds the init with `torch.manual_seed(0)` and the model has **no dropout**,
* the round's evaluator reproduces H1's recorded soup exactly from the stored H1 state (0.12354863 vs recorded 0.12354863),
* the batch order is generator-seeded, so it is identical across runs.

The cause is the *execution regime*: repeated identical CUDA forward passes differ by up to 1.4e-04 in the pooled read-outs (`index_add_` accumulation order) while CPU is exactly 0.0, and two back-to-back runs of the identical protocol (same seed, same GPU) differ by 0.001-0.042 in valid MAE after only two epochs.  A historical point therefore cannot be reproduced to within 0.005, and the historical H1 number is not a usable decision baseline.

Evidence: `reproducibility_probe.json`, `notes/e2e_dictenv_purify_v0_analysis.md`.

## Not run

* `reference_seed1.json` — NOT RUN: round stopped by 10.2 (reference reproduction failure); the candidate is not interpreted
* `purified_seed1.json` — NOT RUN: round stopped by 10.2 (reference reproduction failure); the candidate is not interpreted
* `mechanism_interventions.json` — NOT RUN: candidate not accepted
* `constant_channel_ablation.json` — NOT RUN: candidate not accepted
* `dictionary_health.json` — NOT RUN: candidate not accepted

## Claim scope

> no architecture claim is made: the round stopped on pre-registration 10.2 because the fresh reference did not reproduce the historical H1 context within 0.005, so delta_purification (seed 0) is recorded but not interpreted

Not claimed: all chemistry bypasses are unnecessary; global composition is redundant
