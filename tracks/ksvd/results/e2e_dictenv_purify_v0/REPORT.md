# E2E-DictEnv-Purify-v0 — report

Frozen verdict: **REFERENCE_REPRODUCTION_FAILURE**

Structure basis -> attribute valuation -> static composition.  Official ZINC test was **never** loaded.

## Stage 0 — semantic refactor equivalence

* reference checkpoint comparison max |old - new| = 3.815e-06 (tolerance 1e-6, bit-identical: False)
* parameter ledger: reference 97487 -> purified 95711 (-1776, -1.822 %)

## Primary result (Top-5 soup, official valid)

| seed | reference | purified | Delta |
|---|---:|---:|---:|
| 0 | 0.129878 | 0.135497 | +0.005619 |

* seed-0 case: `performance_failure`; seed 1 authorized: False; seed 1 run: False
* fresh reference reproduction drift vs historical H1 context = +0.006330 (tolerance 0.005)

## Not run

* `reference_seed1.json` — NOT RUN: seed-0 Delta > +0.004 (PURIFICATION_PERFORMANCE_FAILURE)
* `purified_seed1.json` — NOT RUN: seed-0 Delta > +0.004 (PURIFICATION_PERFORMANCE_FAILURE)
* `mechanism_interventions.json` — NOT RUN: candidate not accepted
* `constant_channel_ablation.json` — NOT RUN: candidate not accepted
* `dictionary_health.json` — NOT RUN: candidate not accepted

## Claim scope

> local zeroth-order chemistry can be consolidated into the attributed structural measure without material loss under the frozen H1 architecture

Not claimed: all chemistry bypasses are unnecessary; global composition is redundant
