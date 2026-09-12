# Top-5 Checkpoint Aggregation Stabilization Audit

**Date:** 2026-09-12
**Scope:** checkpoint-estimator stabilization (zero new backbone training)
**Module:** `tracks/ksvd/experiments/luyin16/zinc_top5_checkpoint_aggregation_stabilization.py`
**Results:** `tracks/ksvd/results/top5_checkpoint_aggregation_stabilization/`
**Tests:** `tracks/ksvd/tests/test_top5_checkpoint_aggregation_stabilization.py` (20 pass)
**Official valid:** opened once, only after `confirmation_lock.json`. **Official test:** never loaded.

**Final verdict: `CHECKPOINT AGGREGATION STABILIZATION SUPPORTED` (Decision Case D).**
Primary: **`TOP-5 WEIGHT-SOUP STABILIZATION SIGNAL`** — authorizes
`checkpoint_weight_averaging_protocol_design`; it does **not** authorize EMA/SWA
and does **not** authorize official-test evaluation.

---

## 1. Motivation

The preceding Internal Generalization & Training-Stochasticity Audit established
`CHECKPOINT-SELECTION NOISE DOMINANT`. Both canonical 82,115-parameter
`compact-v4-smallhead` runs (`I0T0`, `I1T1`) showed material selection regret
(evaluation on an internal 2000-molecule probe opened only after training):

| run | raw 800-best epoch | 800 @ raw | 2000 probe @ raw | probe oracle | selection regret |
|---|---:|---:|---:|---:|---:|
| I0T0 | 232 | 0.152610 | 0.131679 | 0.129228 | 0.002451 |
| I1T1 | 173 | 0.154226 | 0.127857 | 0.123545 | 0.004312 |

Mean selection regret `0.003381`; the raw-800 minima are isolated dips
(I0T0 W5 0.005464, I1T1 W5 0.009059). This audit tests exactly one remedy: does
a **fixed** equal-weight aggregation of the five lowest-800-MAE checkpoints
produce a more stable estimator than the single raw argmin?

This is not a new architecture, representation, regularization sweep, EMA/SWA,
optimizer search, K sweep, diversity selection, or official-test benchmark.

## 2. Evidence for checkpoint-selection noise

The 82K seed spread on official valid was `0.007275` (0.145334 vs 0.138059) but
did not reproduce on the genuinely independent internal probe
(`S_raw = 0.003821 < 0.004`, paired-bootstrap 95% CI `[-0.00210, +0.00984]`).
Both runs still lost `0.0025–0.0043` MAE simply because the noisy 800 argmin
landed away from the probe optimum. The 800 curve tracks the probe direction
(Pearson 0.997–0.999) but its epoch-to-epoch noise displaces the argmin. The
problem being tested here is therefore **estimator sensitivity**, not a better
search over evaluation sets.

## 3. Why curve smoothing was rejected

The prior audit pre-registered a centred 5-epoch moving-average selector. It was
**worse** on the probe for both runs (I0T0 gain `-0.006244`, I1T1 `-0.007605`;
mean `-0.006925`). Moving-average selection is therefore forbidden in this
stage; it is not retried in any form.

## 4. Why K=5 is locked

`K = 5` was frozen in `aggregation_protocol_lock.json` **before** any new
evaluation. It is not re-chosen from the 2000 probe, official valid, or
prediction diversity. No `K ∈ {2,3,4,8,10}` and no epoch-spacing constraint are
tested. If the five selected states cluster in adjacent epochs, that is part of
the result, not a defect to repair.

## 5. Why no new training is needed

The audit reuses only the existing `I0T0` / `I1T1` saved trajectories:

* 240 and 213 per-epoch `state_dict` snapshots, all present and SHA-256 verified;
* the frozen 7200/800/2000 official-train split (`assignment_sha256`
  `7b6704d4…`, index-for-index identical to the frozen manifest);
* zero gradient updates, zero new seeds, zero new backbone training.

`official test` is never loaded.

## 6. Top-5 selection rule

For each run independently, epochs are ranked **strictly** by their saved
800-selection MAE. Ties resolve to the **earliest epoch**. Selected Top-5:

| run | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---:|---:|---:|---:|---:|
| I0T0 | **232** (0.152610) | 215 (0.152810) | 223 (0.154181) | 219 (0.154310) | 228 (0.154399) |
| I1T1 | **173** (0.154226) | 197 (0.154724) | 183 (0.154912) | 170 (0.155499) | 195 (0.156336) |

The raw canonical checkpoint is `e_raw = e_1` (I0T0 epoch 232, I1T1 epoch 173).
The 2000 probe and official valid play no role in membership.

## 7. RAW estimator

`E0 = f_{θ_{e_1}}` — the canonical single-argmin estimator, unchanged.

## 8. Prediction ensemble

`E1: ŷ_ens(x) = (1/5) Σ_{k=1}^5 f_{θ_{e_k}}(x)` — a strict **arithmetic mean**
of the five checkpoint predictions. Even though training used L1, prediction
median is forbidden because the audit allows exactly one fixed aggregation rule.
This estimator is a **function-space diagnostic / upper reference**, not the
default deployment candidate: it stores five models and costs ~5 forward passes.

## 9. Weight soup

`E2: θ_soup = (1/5) Σ_{k=1}^5 θ_{e_k}` over every trainable parameter, then
`f_soup = f_{θ_soup}`. Equal-weight arithmetic only — no greedy soup, no
inverse-loss or selection-MAE weighting, no learned weights. The soup remains a
single **82,115-parameter** model with the same architecture, R=302, tokenizer,
topology, inference graph and single-model forward cost. This is the primary
intervention.

## 10. Weight/buffer averaging semantics

The 82,115-parameter model has **46 trainable float32 tensors and 0 buffers**
(`n_buffers = 0`). All soup tensors are finite. `load_state_dict(strict=True)`
reports **no missing / no unexpected keys**, and the loaded soup forwards
normally. There is therefore no `BUFFER_CONFLICT`: no floating non-trainable
buffer and no integer/categorical buffer exists. The pre-registered rule
(bit-identical floating buffers copied; non-identical floating buffers reported
as `BUFFER_CONFLICT`; integer buffers required identical) is satisfied
vacuously and recorded in `soup_construction_*.json`.

## 11. Protocol locking

`aggregation_protocol_lock.json` (written first, never modified) freezes `K=5`,
ranking metric, tie rule, arithmetic prediction mean, arithmetic weight mean, no
epoch spacing, no greedy/weighted soup, primary=soup, diagnostic=ensemble,
development-reuse set=2000 probe, confirmation set=official valid, official
test=locked, and the decision gates. `confirmation_lock.json` was written
**after Stage A and before the first official-valid read**, and records the
protocol hash, Top-5 manifest hashes, soup state hashes (file + tensor),
primary-metric definition, decision gates and timestamp.

## 12. Development-reuse 2000 probe caveat

The 2000 probe already participated in the prior hypothesis formation. Its
numbers are a **development-reuse diagnostic only** — not independent
confirmation, not fresh validation, not an unbiased replication. Stage A has no
continue/stop gate: no K, weight, membership, or averaging change was allowed
based on it, and none was made.

## 13. Prediction disagreement

Per-molecule `s_pred(x) = std(ŷ_1,…,ŷ_5)` on the 2000 probe (descriptive only):

| run | mean | median | p90 | p95 |
|---|---:|---:|---:|---:|
| I0T0 | 0.035872 | 0.032493 | 0.055808 | 0.064794 |
| I1T1 | 0.036624 | 0.031467 | 0.059940 | 0.074843 |

Pairwise checkpoint prediction similarity (descriptive):

| run | Pearson mean (min) | cosine mean (min) | mean abs pairwise diff |
|---|---:|---:|---:|
| I0T0 | 0.999492 (0.999376) | 0.999465 (0.999373) | 0.048127 |
| I1T1 | 0.999389 (0.999091) | 0.999316 (0.998963) | 0.049075 |

The trajectory checkpoint functions are highly correlated but not identical;
diversity is **not** used as a selection criterion.

## 14. Weight-space diagnostics

`‖θ_k − θ_soup‖_2 / (‖θ_soup‖_2 + ε)` (descriptive):

| run | soup norm | rank-1 | rank-2 | rank-3 | rank-4 | rank-5 | mean pairwise |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0T0 | 49.06 | 0.0502 | 0.0500 | 0.0325 | 0.0367 | 0.0387 | 0.0653 |
| I1T1 | 54.99 | 0.0576 | 0.0656 | 0.0418 | 0.0676 | 0.0586 | 0.0881 |

The soup lies at a modest normalized distance from each member (~3–7%).

## 15. Why official valid is only a locked development confirmation

Official valid has been used extensively in earlier project development, so it
is **not** a pristine, project-level independent holdout. However, the Top-5
rule, `K`, checkpoint membership, averaging rules, soup states and decision
gates were all frozen **before** this experiment accessed official valid. This
stage therefore calls it a **locked development confirmation**, not an untouched
final test.

## 16. Confirmation results

Official-valid MAE (1000 molecules), once, on the locked estimators:

| run | RAW | ENSEMBLE | SOUP | Δ_ens = raw−ens | Δ_soup = raw−soup |
|---|---:|---:|---:|---:|---:|
| I0T0 | 0.167234 | 0.163056 | 0.163063 | **+0.004178** | **+0.004171** |
| I1T1 | 0.157832 | 0.152685 | 0.152898 | **+0.005147** | **+0.004934** |

Both runs improve under both estimators. The individual Top-5 checkpoints were
also evaluated for description only (never re-selected):

| run | rank-1 | rank-2 | rank-3 | rank-4 | rank-5 | best single |
|---|---:|---:|---:|---:|---:|---:|
| I0T0 | 0.167234 | 0.166155 | 0.166229 | 0.169733 | 0.167017 | 0.166155 (ep 215) |
| I1T1 | 0.157832 | 0.157700 | 0.158350 | 0.157105 | 0.156062 | 0.156062 (ep 195) |

No official-valid oracle checkpoint was or will be defined, and no
valid-based re-definition of the selector is permitted.

## 17. Ensemble mechanism result

Two-run mean `Δ_ens = 0.004663`, both per-run effects positive, above the
`+0.002` mechanism gate → **`FUNCTION-SPACE AGGREGATION SIGNAL`**. This is a
mechanism diagnostic, not a deployment gate: the ensemble costs ~5 forward
passes and stores five models.

## 18. Soup primary result

Two-run mean `Δ_soup = 0.004552`:

* G1: both `Δ_soup,s ≥ 0` — **pass** (+0.004171, +0.004934);
* G2: mean `Δ_soup = 0.004552 ≥ +0.0015` — **pass**;
* G3: two-run paired-bootstrap 95% CI lower `> 0` — **pass** (see §19).

Verdict: **`TOP-5 WEIGHT-SOUP STABILIZATION SIGNAL`**. Because the ensemble
mechanism gate also passes, the overall case is **D — `CHECKPOINT AGGREGATION
STABILIZATION SUPPORTED`**, with the weight soup as the deployable-form primary
result (single 82,115-parameter model, one forward pass).

## 19. Bootstrap uncertainty

Paired molecule-level bootstrap (`B = 2000`, fixed seed 20260923) on official
valid (`n = 1000`), resampling the same molecule indices for both runs:

| comparison | effect | 95% CI | P(>0) |
|---|---:|---:|---:|
| I0T0 raw vs soup | +0.004171 | [+0.001674, +0.006784] | 1.000 |
| I1T1 raw vs soup | +0.004934 | [+0.002298, +0.007545] | 1.000 |
| **two-run mean raw vs soup** | **+0.004552** | **[+0.002667, +0.006456]** | **1.000** |
| two-run mean raw vs ensemble | +0.004663 | [+0.002829, +0.006494] | 1.000 |

Ensemble-vs-soup is statistically indistinguishable from zero in both runs
(I0T0 −0.000007, CI [−0.000522, +0.000516]; I1T1 −0.000214, CI
[−0.001201, +0.000901]). The single-model soup retains essentially all of the
function-space ensemble benefit.

## 20. Cross-run spread

| estimator | |MAE_0 − MAE_1| |
|---|---:|
| RAW | 0.009402 |
| SOUP | 0.010165 |
| ENSEMBLE | 0.010371 |

`S_soup − S_raw = +0.000762`: the soup did **not** reduce the two-run point
spread on this confirmation set. With only two trajectories this is secondary
descriptive evidence, not a variance estimate, and it does not enter the
primary decision. The primary claim is about each run's estimator vs its own raw
argmin, not about cross-run spread.

## 21. Parameter and inference cost

| estimator | params | stored models | forward (128-mol batch) | multiplier |
|---|---:|---:|---:|---:|
| RAW | 82,115 | 1 | 18.52 ms | 1.00× |
| SOUP | 82,115 | 1 | 18.74 ms | 1.01× |
| ENSEMBLE | 82,115 × 5 | 5 (410,575 stored) | 95.45 ms | 5.15× |

The soup is compute-neutral relative to raw and keeps the exact 82,115-parameter
architecture. We do **not** claim a speedup; the improvement is estimator
stabilization at no inference-model-size increase. The ensemble is ~5× inference
cost and is not the deployment candidate.

## 22. What is and is not proven

**Proven (within this locked protocol):**

* The fixed Top-5 equal-weight weight soup improves official-valid MAE over the
  canonical raw 800-argmin on both development trajectories
  (mean `+0.004552`, G1/G2/G3 all pass).
* The improvement is statistically supported by a paired bootstrap whose 95% CI
  lower bound is `+0.002667 > 0`.
* The soup is a single 82,115-parameter model with unchanged architecture and
  ~1× forward cost, and the ensemble-vs-soup difference is indistinguishable
  from zero, so the soup retains essentially all of the function-space benefit.

**Not proven / not claimed:**

* that checkpoint-selection noise has been solved;
* that the effect generalizes beyond these two trajectories;
* that official valid is a pristine independent holdout for the project;
* that the mechanism is prediction ensembling (a weight space could be flatter
  or better interpolated);
* any official-test performance claim (the test is not evaluated).

## 23. Why EMA/SWA were not run

EMA/SWA are also weight-space averaging mechanisms but require new training-time
hooks and their own pre-registered design. Soup success authorizes
`checkpoint_weight_averaging_protocol_design` as a future protocol family only.
Whether EMA/SWA are worth testing must be separately pre-registered; this audit
does not run them, does not sweep averaging windows and does not sweep decay.

## 24. Why K was not tuned

`K=5` is a fixed a priori estimator. Tuning K on the reused 2000 probe or on the
just-opened official valid would be double dipping and would invalidate the
confirmation. No `K`, no epoch spacing, no greedy selection, no weighted soup and
no membership re-selection were performed after the lock.

## 25. Official-test lock

Official test was never loaded. This remains true even though both gates passed.
No test MAE, prediction, checkpoint or ensemble was computed. No official-test
evaluation is authorized by this result.

## 26. Final verdict

**Decision Case D — `CHECKPOINT AGGREGATION STABILIZATION SUPPORTED`.**
Primary deployable-form result: **`TOP-5 WEIGHT-SOUP STABILIZATION SIGNAL`**.

> A fixed Top-5 weight average reduces the observed sensitivity of the selected
> estimator under the locked development protocol.

This is the maximum-strength statement licensed by two trajectories and a
locked development confirmation; it is not a solved-problem claim and not a
final benchmark result.

## 27. Next authorization

Authorized: **`checkpoint_weight_averaging_protocol_design`** — design how a
canonical checkpoint-weight-averaging rule should be locked in a future 10K
training protocol without ever touching the official test.

Not authorized: EMA/SWA, averaging-window or decay sweeps, K rescue, greedy or
weighted soup, new backbone training, and any official-test access.

---

## Integrity gates

Stage-0 gates G0.1–G0.12 all pass: run fingerprints, 82,115-param architecture,
Top-5 by 800 MAE only, probe/valid excluded from selection, test never loaded,
equal prediction mean, equal parameter mean, soup still 82,115, all soup tensors
finite, buffer policy satisfied (0 buffers), and soup loads/forwards with no
missing/unexpected state. The 20-test suite covers the 16 pre-registered
integrity tests (Top-5 by 800 only, K fixed, tie rule fixed, probe excluded,
valid excluded, test never loaded, checkpoint hashes, equal prediction mean,
equal parameter mean, no greedy, no weighted soup, 82,115 soup params, key
consistency, buffer policy, confirmation-lock ordering, no post-valid parameter
modification, no new backbone training).

## Q1–Q20

* **Q1.** I0T0 Top-5 epochs: 232, 215, 223, 219, 228.
* **Q2.** I1T1 Top-5 epochs: 173, 197, 183, 170, 195.
* **Q3.** Yes — selected strictly by 800-selection MAE (earliest-epoch ties).
* **Q4.** Both sets are epoch-local: I0T0 spans 17 epochs (215–232), I1T1 spans
  27 (170–197). Described only; no spacing constraint was applied.
* **Q5.** Development-reuse 2000 probe — I0T0 raw 0.131679 / ensemble 0.127050 /
  soup 0.127517; I1T1 raw 0.127857 / ensemble 0.122825 / soup 0.124268.
* **Q6.** Prediction disagreement `s_pred`: I0T0 mean 0.035872 / median
  0.032493 / p90 0.055808 / p95 0.064794; I1T1 mean 0.036624 / median 0.031467
  / p90 0.059940 / p95 0.074843.
* **Q7.** Normalized soup distance ~0.03–0.07 per member (mean pairwise 0.0653
  for I0T0, 0.0881 for I1T1).
* **Q8.** Yes — protocol, manifests, soup hashes and gates were frozen in
  `confirmation_lock.json` before the first official-valid read.
* **Q9.** I0T0 official-valid RAW MAE: 0.167234.
* **Q10.** I0T0 ensemble MAE: 0.163056.
* **Q11.** I0T0 soup MAE: 0.163063.
* **Q12.** I1T1 — RAW 0.157832, ensemble 0.152685, soup 0.152898.
* **Q13.** `Δ_ens`: I0T0 +0.004178, I1T1 +0.005147.
* **Q14.** `Δ_soup`: I0T0 +0.004171, I1T1 +0.004934.
* **Q15.** Mean soup gain 0.004552 ≥ +0.0015 — **pass**.
* **Q16.** Two-run paired bootstrap 95% CI `[+0.002667, +0.006456]`, lower > 0 —
  **pass**.
* **Q17.** Ensemble mechanism: two-run mean +0.004663 ≥ +0.002, both runs
  positive — **pass**.
* **Q18.** Soup did **not** reduce the two-run point spread
  (`S_soup − S_raw = +0.000762`); secondary descriptive only.
* **Q19.** RAW 82,115 params / 18.52 ms; SOUP 82,115 params / 18.74 ms;
  ENSEMBLE five models / 410,575 stored params / 95.45 ms (~5.15×).
* **Q20.** Decision Case D — `CHECKPOINT AGGREGATION STABILIZATION SUPPORTED`
  (primary `TOP-5 WEIGHT-SOUP STABILIZATION SIGNAL`).
