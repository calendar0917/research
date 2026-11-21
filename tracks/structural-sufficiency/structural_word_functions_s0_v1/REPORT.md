# REPORT — structural word functions + real-binding composition, s0 (ogbg-mollipo, seed 0)

Protocol: `PROTOCOL.md` (pre-registered before any branch result existed), config
`configs/swf_s0.json`, protocol id `structural-word-functions-s0-v1`.
Decision: **`STOP_AT_PILOT`** — the *word representation* is competitive, the
*real-binding composition* shows nothing at this pilot.

T0 = 2026-11-21T19:21:32+08:00. All training ran on `res` (2×A100-SXM4-40GB,
pools `res-gpu0`/`res-gpu1`, GPU UUID `7c4f57e5-21e9-d1e6-c7c6-779b2624700d`),
CPU baselines locally (the compute host has no RdKit).

## Direct answers

1. **Scores (official Valid / Test RMSE · MAE, original label units, one seed,
   model selection on Valid only).**

   | arm | Valid RMSE | Valid MAE | Test RMSE | Test MAE | Train RMSE | params | best ep |
   |---|---|---|---|---|---|---|---|
   | `WORD_PREFIX` (word bag) | **0.6991** | 0.5264 | **0.6891** | 0.5139 | 0.2662 | 808 810 | 119 |
   | `FULL` (bag + composition) | 0.6925 | 0.5221 | 0.6980 | 0.5249 | 0.2307 | 808 810 | 45 |
   | `BAG_CONT` (bag + dead context) | **0.6924** | **0.5221** | 0.6980 | 0.5249 | 0.2308 | 808 810 | 45 |
   | `GINE_RAW_REF` | **0.6101** | 0.4514 | **0.6722** | 0.4887 | 0.2459 | 387 974 | 174 |
   | `FP_KRR` (Morgan r2/2048, Tanimoto) | 0.8594 | 0.6660 | 0.7926 | 0.6095 | 0.0333 | — | — |
   | `COUNT_RIDGE` (attributed subgraph counts) | 0.9765 | 0.7515 | 0.8251 | 0.6431 | 0.4124 | — | — |

2. **The composition buys nothing.** `FULL − BAG_CONT` on Valid = **−0.0001**
   (FULL is marginally *worse*), Test = −0.00001. Paired bootstrap over molecules
   (Valid): median **−1.0e-4**, 95 % CI **[−1.75e-4, −3.3e-5]** — the interval
   excludes 0 in favour of the *control*. Training is not the limiting factor:
   both arms stop improving at epoch 45 and then drive Train RMSE from 0.23 to
   0.17 while Valid stays at 0.70.

3. **Why: the composition branch never turned on.** Three independent measurements.
   * Residual scale: `RMS(δ)` where `y = y_bag + δ` is **0.0014** in label units at
     the recorded best state and **0.0081** at the final epoch, against
     `RMS(y_bag) = 2.478` → the whole sentence/relation path carries **0.06 % /
     0.33 %** of the prediction scale (`analysis/conditional.json:composition_scale`).
   * The `join` output layer, initialised at exactly 0, only reached
     `mean|W| = 5.6e-4` (best) / `2.6e-3` (final) after 120 epochs.
   * Gradients at step 1 are exactly 0 for the new branch and at step 2 are
     `q 1.9e-4, k 1.3e-4, rel_mlp 3.4e-4` versus `word_mlp 1.96e-1, bag_mlp 5.1e-2,
     join 2.3e-1` — **three orders of magnitude** smaller, i.e. the zero-initialised
     residual under a shared Adam starves the branch (`analysis/checks.json:branch`).
   * Consistency check: replacing the sentence operator of the trained FULL weights
     by the `EARLY_POOL` operator changes predictions by RMS **8.9e-6** and RMSE by
     **1.0e-7** (Valid). There is no dependence to intervene on.

4. **The word representation itself is the interesting part, and it is not a GNN.**
   1.95 M retained-attribute words, no message passing, no recurrence, no teacher:
   Valid **0.6991** with 0.81 M parameters and 297 s of GPU time. It beats both
   simple baselines by **0.167 / 0.277** RMSE, and sits **0.083** behind a
   same-round GINE that took 322 s. That is the one positive signal of the round,
   and it is a *representation* signal, not a composition signal.

5. **The FULL-vs-GINE gap is split-unstable.** Valid gap 0.0823 (bootstrap CI
   [−0.139, −0.028]) but Test gap 0.0251 (CI [−0.074, +0.025], includes 0). With
   420 molecules per split, "the bag is 0.08 behind GINE" is not a resolved
   quantity; it is 0.03–0.08 depending on the split.

## What was actually built

* Exhaustive rooted-template library, `k = 1..4`, **16 templates** (1 / 1 / 3 / 11).
* Every word is the *full ordered attribute matrix* of its atoms and physical bonds
  in an automorphism frame (root pinned to slot 0); the learned part starts at
  `WordMLP(256→128→64)`. Nothing replaces an attribute column by a learned index.
* Full corpus over all 4200 molecules, computed once (227 s, deterministic):
  1 954 725 words · 2 463 158 frames · 3 316 287 sentences · 9 333 784 covers ·
  16 863 050 frame-pair rows (1.807 rows/cover). Dedup: 71 004 unique word-frame
  inputs (97.1 %) and 27 188 unique relation inputs (99.84 %).
* A *relation payload* is explicit and finite: same-atom matrix `E` (u16), physical
  cross-bond matrix `C` (u16) and 16 cross-bond attribute codes — so "the real
  variable binding" is a table lookup, not a learned attention.
* All label-touching statistics (`mu_T`, `nu_k`, support masks, target scaler) are
  Train-only; 3 of the 16 templates (ids 11, 14, 15) never appear in Train and get
  a zeroed embedding.

## Implementation checks (`analysis/checks.json`, all pass)

| check | result |
|---|---|
| data identity (splits 3360/420/420, `row_id = arange`, local numbering, 1/1/3/11) | pass |
| binding algebra: combine-then-average 0.5 vs average-then-multiply 0.25 on the same exchangeable pair | pass (5/5 cases) |
| cover swap symmetry / scatter-vs-loop equality (31 562 rows, 5 977 sentences) | exact 0.0 |
| branch init == prefix weights | max abs diff 0.0 |
| gradient sanity step 1 / step 2 | new branch 0 → finite non-zero |
| re-labelling invariance, 8 molecules × 5 modes, FP64 | max abs diff **3.5e-17** |

The re-labelling check found a real bug before any training: packed index arrays were
built by subtracting a per-molecule offset, which is only valid when the batch is
sorted, so molecule *n*'s frames leaked into molecule 0's word slots (0.0152
invariance error). It is now an explicit batch-position map (`_local_pos`); the
error is at machine precision.

## Decision (frozen rule, `PROTOCOL.md`)

```
gap    = 0.69247 − 0.61015 = 0.0823      (≤ 0.10, does not stop)
margin = 0.69247 − 0.85941 = −0.1669     (≤ 0.03, does not stop)
gain   = 0.69237 − 0.69247 = −0.0001     (≥ 0 required)  ← trips
```

`gain_bag < 0` → **`STOP_AT_PILOT`**: "BAG_CONT beats FULL: the real-binding
composition has no benefit at this pilot". Per rule 2 the mechanism arms
(`EARLY_POOL`, `FLAT_CONTEXT`) and the ESOL replication were **not launched**; this
is a protocol outcome, not an omission. As the protocol also requires, the
descriptive analyses were still run once (they are not new training): paired
bootstrap, fixed-weight operator intervention, composition decomposition,
structure statistics, structure cases.

## Structure statistics — descriptive reach (not a gain bound)

* 341 933 / 1 954 725 words (17.5 %) have a non-trivial root automorphism;
  they participate in 3 483 472 covers (37.3 %).
* Attributes actually differ across a word's own frames in 2 975 038 covers
  (31.9 %) — i.e. on a third of the covers the `EARLY_POOL`-vs-`FULL` distinction is
  a real distinction, not a relabelling.
* The relation payload differs across frames in 3 000 454 covers (32.1 %).

So the frame machinery *matters* structurally on ~a third of the objects, and still
produced no measurable gain — the failure is not that the frames were trivial.

## What this round does not show

* **It does not show that composition is useless.** The branch was still growing
  monotonically at the last epoch (δ: 4e-4 → 8e-3, `join` W: 5.6e-4 → 2.6e-3) when
  the 120-epoch horizon ended. What is established is: *at 120 epochs, lr 3e-4,
  3 360 training molecules and a zero-initialised residual join, the composition
  path contributes ≤ 0.3 % of the prediction scale and changes RMSE by 1e-4.*
  A fair test of the mechanism needs a separate (or larger) learning rate for the
  new branch, a longer horizon, and/or a non-zero residual initialisation — that is
  a new round, not a re-interpretation of this one.
* **`EARLY_POOL` / `FLAT_CONTEXT` were not run**, by the pre-registered rule. So we
  cannot say whether attribute-permutation retention matters, nor whether "learn one
  large local object directly" would have matched the factorised form. Given that
  the composition is dormant, both controls would most likely have reduced to
  bag-level models.
* One seed, one split, no cross-seed stability, no significance claim on the
  Valid-vs-Test gap. The bootstrap reported here is sample-composition sensitivity
  only; it excludes training-seed and model-selection uncertainty.

## Budget accounting

| unit | wall | GPU·h |
|---|---|---|
| `WORD_PREFIX` (160 ep) | 297 s | 0.082 |
| `GINE_RAW_REF` (224 ep) | 322 s | 0.089 |
| `FULL` (120 ep) | 1604 s | 0.445 |
| `BAG_CONT` (120 ep) | 1699 s | 0.472 |
| `COUNT_RIDGE`, `FP_KRR`, checks, prepare | 994 s CPU | 0 |
| **total** | **≈ 4 900 s (1 h 23 m)** | **1.09 / 8** |

6 of 10 formal units used; 1 of 2 engineering reruns; ≤ 2 GPUs (never more than one
per job, two concurrent); no launch after 22:21; all training finished by 20:41
(T0+1 h 20 m). Deviations are listed in `PROTOCOL.md` and
`analysis/deviations.jsonl`; the material one is the `COUNT_RIDGE` alpha grid, which
was extended because the frozen grid selected its own boundary — the extended grid
makes that baseline **stronger** (0.9765 instead of 1.3697) and the frozen-grid record
is kept in `analysis/baselines_frozen_grid.json`.

## Files

```
PROTOCOL.md                       pre-registration + deviations
code/swf_data.py                  template library, words/frames/sentences/covers
code/swf_cache.py                 exhaustive corpus build, batch packing
code/swf_model.py                 shared word functions + binding + bag readout
code/swf_train.py                 prefix / branch / GINE reference training
code/swf_baselines.py             COUNT_RIDGE, FP_KRR
code/swf_checks.py                identity, binding algebra, invariance checks
code/swf_analyze.py               decision rule, bootstrap, intervention, cases
analysis/main_table.csv           all arms, best + final epoch
analysis/summary.json             decision payload
analysis/conditional.json         bootstrap, intervention, composition scale, cases
analysis/structure_stats.json     template library + descriptives
analysis/checks.json              gating checks
analysis/baselines.json           baseline alphas and metrics
analysis/baselines_frozen_grid.json  the pre-extension COUNT_RIDGE record
results/lipo_*/                   per-run run.json, curves.csv, predictions, states
```
