# DECISION — LCRC Lipophilicity prototype / mid-range interaction pilot v1

Protocol `lcrc-lipo-proto-interact-pilot-v1` · commit `ff16154d47b6` · seed 0 · v2 Fit2352 scaffold Fold 2
(Train 1568 / Dev 784) · R3 recipe · 0.2775 of the 0.5 A100 GPU·h budget used.

Frozen exploratory rule (fixed before any GPU run):

* PROTO signal ⇔ `BASE_G0 dev RMSE − PROTO dev RMSE ≥ 0.020`
* PAIR signal ⇔ `BASE_G1 dev RMSE − PAIR_TRUE dev RMSE ≥ 0.020` **and**
  `PAIR_NULL dev RMSE − PAIR_TRUE dev RMSE ≥ 0.010`
* otherwise `NO_STRONG_SIGNAL_THIS_CONFIGURATION`

**Outcome: `NO_STRONG_SIGNAL_THIS_CONFIGURATION`.**

Measured gains (positive = candidate better):

| direction | gain | threshold | pass |
|---|---|---|---|
| PROTO vs `BASE_G0` | **−0.0340** | +0.020 | no |
| PAIR_TRUE vs `BASE_G1` | **−0.0086** | +0.020 | no |
| PAIR_TRUE vs `PAIR_NULL` (semantics effect) | **+0.0068** | +0.010 | no |
| device control `BASE_G0` vs `BASE_G1` | 0.0018 (|drift|) | — | — |

---

## 1. Direction 1 — shared local prototypes: no exploratory signal

`PROTO_K16_TOP4` reaches Dev RMSE **0.7419** against `BASE_G0` **0.7079** (−0.0340), Dev MAE
0.5826 vs 0.5477. The deficit is broad, not a tail effect: 409/784 molecules are worse, and removing the
five worst BASE molecules *enlarges* the gap to −0.0521. Train RMSE is also worse (0.3645 vs 0.2796),
so this configuration is not trading bias for variance — it fits the training fold less well.

Routing diagnostics explain the mechanism without changing the conclusion:

* routing is **alive** — permuting only the prototype keys moves every Dev prediction by 2.34 logD on
  average, and prototype/ expert/ bias tensors all receive non-zero gradients;
* routing is **collapsed** — 2.54 effective prototypes (exp of top-1 entropy), one prototype taking 72 %
  of all atoms, 1 of 16 prototypes never used as top-1 on Train (2 on Dev). Train and Dev usage
  distributions are nearly identical, so this is a property of the learned solution, not of overfitting.

Per the frozen protocol this is recorded as **"no signal in this configuration"**, *not* as evidence that
the shared-prototype / dictionary route is invalid. Only one K, one top-k, one temperature, one seed and
one fold were tested, and no auxiliary loss, entropy penalty or re-initialisation was applied. The
observed collapse (`K=16` behaving like `K≈2.5` with a shared linear expert per prototype) is a concrete
reason why *this* configuration underperforms, and it is a candidate explanation to test later — it is not
a refutation.

## 2. Direction 2 — explicit mid-range pair interactions: no exploratory signal, but a small, consistent semantics effect

`PAIR_TRUE` reaches Dev RMSE **0.7182** vs `BASE_G1` **0.7097** (−0.0086). Train RMSE improves
(0.2639 vs 0.2778) while Dev does not, and the generalisation gap widens from +0.4319 to +0.4543: the
2,641 extra parameters absorb training-fold structure that does not transfer. The comparisons are mixed
rather than uniformly negative — Dev MAE is almost unchanged (+0.0022) and 408/784 molecules improve — but
the primary metric is the Dev RMSE and it is worse.

The distance-semantics control is the informative part:

* `PAIR_NULL` (identical pairs `(u,v)`, identical parameter count and initialisation, each molecule's own
  fixed permutation of the 3/4/5 class names) is **worse** than `PAIR_TRUE` by **0.0068** RMSE
  (0.7250 vs 0.7182) and 0.0092 MAE. The effect is in the expected direction — cross-molecule distance
  semantics carry information — and is about 4× the measured device drift (0.0018), but it stays below the
  pre-registered 0.010 threshold, and both pair arms are worse than their BASE control.
* Coverage was not a limiting factor: distances 3/4/5 cover ~100 % of molecules in both folds
  (mean 41.5 / 38.0 / 36.1 pairs per Train molecule), 181,285 Train and 93,385 Dev pairs in total.
* The residual is genuinely used: mean |δ| = 0.208 logD, max 0.870, every Dev graph has δ ≠ 0, every
  delta-path tensor has a non-zero gradient, and zeroing the residual head reproduces the same-weight BASE
  output to 9e-7 logD.

So: **no PAIR signal at the required magnitude**, with a small semantics-consistency effect that the
frozen rule does not count as a signal. This is a single seed, single fold observation; it establishes
neither that pair interactions are useless nor that distance semantics are irrelevant.

## 3. Uncertainty and what would change the reading

* **Run-to-run / device noise is one observation wide.** `BASE_G0` and `BASE_G1` share an initialisation
  hash and differ only by GPU and floating-point reduction order, yet their Dev RMSE differs by 0.0018
  (MAE 0.0010). The PROTO gap (0.034) is ~19× that scale; the PAIR-vs-BASE gap (0.0086) is ~5×; the
  PAIR-vs-NULL gap (0.0068) is ~4×. Any one of these could move materially with a different seed.
* **Fold dependence.** Fold 2 is a scaffold split of 2352 molecules with 784 Dev molecules in 244
  scaffolds; 115 Dev scaffolds are singletons. Scaffold-group statistics (REPORT §4) show no group-wise
  shift beyond noise for the PAIR comparisons, and a systematic `PROTO` deficit in the larger groups.
* **The control is not a pure "semantics" ablation.** Per-molecule permutations preserve content and
  within-molecule grouping but flatten the global distance histogram (Dev: 33,548/30,770/29,067 true vs
  31,172/31,166/31,047 null). Part of the 0.0068 could come from losing that global prior rather than
  from inconsistent semantics.
* **No held-out evidence.** Monitor504, Reserved504 and random42 splits were not evaluated, so nothing is
  claimed about out-of-fold or official-test behaviour.

## 4. What this round does *not* license

* It is **not** a claim that prototype/dictionary reads are worse than per-atom MLP contributions in
  general, nor that mid-range pair interactions do not help LCRC in general.
* It is **not** a paper-level or leaderboard-comparable Lipophilicity result: single seed, single internal
  scaffold fold, Fit2352 only.
* It is **not** a basis for adopting either model. `PROTO_K16_TOP4`, `PAIR_TRUE` and `PAIR_NULL` remain
  exploratory artefacts, checkpointed and reproducible, but not promoted.

## 5. Follow-ups deliberately *not* launched

No additional seeds, folds, K / top-k / temperature settings, entropy or load-balancing terms, distance
windows, gates, message passing, longer schedules or ensembles were run, and none were started after the
decision was computed. The GPU budget stopped at 0.2775 of 0.5 A100 GPU·h with no retries. Any next step
is a research decision for the researcher, not something this round pre-commits.
