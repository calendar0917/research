# Compact-v4 Inductive-Bias & Sample-Efficiency Audit

Track: `ksvd`. Protocol: `inductive_bias_sample_efficiency_audit_v1`.
Question: **at fixed architecture, optimizer, selection set, probe set and
optimizer-step budget, how much held-out MAE is lost by reducing the number of
unique training examples?**

This note records a staged, compute-matched learning-curve audit. It is *not* a
new architecture, a new structural feature, a representation-collision audit, an
optimizer/regularization/EMA/SWA/checkpoint-averaging search, HPO, an
external-model comparison or an official-test benchmark.

## 1. Motivation

The preceding audits closed the "missing information" line: no hard or soft
pre-neural bottleneck, no stagewise representation collision, no stable benefit
from endpoint / covariance / triad / cell / P1 / P2 / broad-state additions. The
natural next question is not *where information is lost* but *how hard the same
information is to learn from a finite number of examples*. This audit measures
the local within-family dependence of held-out MAE on the number of unique
training molecules.

## 2. Why the information-bottleneck search is closed

`raw_graph_patch_system_sufficiency` established, on the official-train only and
with no training, that the pre-neural patch system does not impose a material
hard aliasing: `LB(P3)=0`, progressive signatures `P0`/`P1`/`P2`/`P3` dissolve
raw non-isomorphism at `P1`, the `PATCH_FULL` target geometry beats generic
`RAW` references (`Delta_pre = -0.0878`, paired bootstrap 95% CI
`[-0.0999,-0.0748]`), and the patch/pair encoders improve target geometry.
Combined with the stagewise collision audit (no material degradation) and the
parameter-allocation/compression audit (compression-only opportunity), the
information-sufficiency question is considered answered for the current
representation. Hence this audit does not search for missing information.

## 3. Why checkpoint tuning is no longer the main line

The Top-5 checkpoint aggregation stabilization audit showed that a fixed `K=5`
equal-weight weight soup reduces checkpoint-selection noise on both official
valid (mean `Delta_soup = +0.004552`, paired CI lower `>0`) and the internal
probe. This audit uses exactly that frozen soup as a *measurement stabilizer*
and does **not** reopen K, membership, spacing, greedy/weighted averaging,
EMA or SWA.

## 4. Research question

Estimate the local learning curve `E(N)` and, most importantly, the doubling
gain

```
G_{2x}(N) = E(N) - E(2N).
```

A positive `G` means doubling the data improves held-out MAE. The audit does not
compare against another representation family.

## 5. 82K architecture lock

Exact `compact-v4-smallhead`, 82,115 parameters, `R = 302D`, head
`302 -> 13 -> 13 -> 1` (4,135 parameters), raw `R`, ReLU, no standardisation.
Tokenizer, patch/pair construction, topology, descriptors, widths, head, loss,
optimizer, weight decay and batch size are unchanged. Official test is never
loaded.

## 6. 7200/800/2000 environment

The frozen official-train split is reused index-for-index: 7,200 optimization
pool / 800 checkpoint selection / 2,000 internal probe, with
`split_seed = optimized-manifold-broad-state-screen-v1-20260919`. The audit
verifies the 7200/800/2000 index lists and hashes match the frozen manifest
(`split_inventory.json`). Official valid is not loaded; official test is never
loaded.

## 7. Probe-reuse caveat

The 2,000 probe participated in prior hypothesis formation and is therefore not
a pristine holdout. All N, subset construction, training budget, decision gates
and seed-purchase rules were frozen in `audit_protocol_lock.json`,
`sample_efficiency_subset_lock.json`, `decision_thresholds.json` and
`anchor_protocol_compatibility.json` *before* any new training. The probe never
changed N, budget, optimizer, architecture or the averaging rule.

## 8. Nested target-independent subsets

A deterministic ordering of the 7,200 pool is defined by
`sha256(salt || original_official_train_index)` with a first-seen tie-break on
the index itself. `D1800 subset D3600 subset D7200`. No target, target
quantile, molecule-size, token or chemistry balancing is used
(`target_used=false`, `ordering_uses_target=false`). Subset hashes:

```
D3600 = 3be9f7e1a4fb8cf4ed5eedfbd725b1ac9d618e6bc6d4873fe254ddc1017dfea9
D1800 = 17eb66c1fdd81cf9991501688681c825e18d9b1accdc434ded18ae0eb05a4747
```

## 9. Why compute-matched optimizer steps are used

The canonical 7200 recipe is 240 epochs at 57 steps/epoch. Giving a smaller N
240 epochs would give it proportionally fewer optimizer steps and would
confound *less data* with *less optimization*. The audit therefore uses a fixed
maximum optimizer-step budget derived mechanically from the real anchor.

## 10. 7200 anchor compatibility

From the real `run_I0T0.json`: `steps_per_epoch = 57`, `max_epochs = 240`,
`patience = 40`. Therefore `max_optimizer_steps = 240 * 57 = 13,680`,
`selection_eval_interval = 57` steps and `patience = 40` selection
evaluations. All protocol fields (optimizer, lr, wd, batch, loss, scheduler,
gradient clip, cadence, patience) are checked equivalent at N=7200
(`anchor_protocol_compatibility.json`, `equivalent_at_7200=true`). The existing
7200 `I0T0`/`I1T1` runs are **reused, never retrained**. A legacy terminology
note records that the anchor manifest's `checkpoint_selection` string says
"official-valid" while the code and split inventory prove selection on the
internal 800 split.

## 11. Selection cadence and patience

Selection on the 800 split occurs every 57 optimizer steps. Patience is 40
selection evaluations. Ties in checkpoint selection resolve to the **earlier
step**. Smaller N therefore sees more exposures per unique example by design;
this is reported explicitly rather than hidden.

## 12. RAW estimator

Lowest-800-MAE checkpoint, earliest step wins. Probe MAE computed only after the
run is frozen.

## 13. Locked Top-5 soup estimator

`K=5`, equal-weight arithmetic parameter average of the 5 lowest-800-MAE
checkpoints, earliest step wins. No K sweep, no weighting, no greedy soup, no
spacing, no EMA/SWA. SOUP is the primary sample-scaling measurement channel;
RAW is the canonical-protocol check.

## 14. Stage 1 — N3600_I0T0

Initial state hash equals the historical smallhead `I0` fingerprint exactly
(`63f2cecb...`). The run early-stopped after 106 selection evaluations
(6,042 optimizer steps, 208.4 effective epochs), best 800-selection MAE
0.209662 at step 3,762. Frozen estimators on the 2,000 probe:

| estimator | 3600 seed0 | 7200 seed0 (reused) |
|---|---:|---:|
| RAW probe MAE | 0.181330 | 0.131679 |
| SOUP probe MAE | 0.176633 | 0.127517 |
| RAW train MAE (subset) | 0.073954 | — |
| SOUP train MAE (subset) | 0.048170 | — |

## 15. Local doubling gain

```
G_soup(36->72, seed0) = 0.176633 - 0.127517 = +0.049116
G_raw (36->72, seed0) = 0.181330 - 0.131679 = +0.049652
```

Both are far above the pre-registered material gate of `+0.006`.

## 16. Paired bootstrap

Paired over the same 2,000 probe molecules (`B=2000`, seed `20260912`):

```
G_soup  mean +0.049116   95% CI [+0.042053, +0.056636]   P(>0)=1.000
G_raw   mean +0.049652   95% CI [+0.041602, +0.058385]   P(>0)=1.000
```

The CI lower bounds are strongly positive and RAW agrees in direction, so
Stage 1 is `MATERIAL_LOCAL_DATA_SCALING_SIGNAL`.

## 17. Conditional seed1 replication (Stage 2) — N3600_I1T1

Because Stage 1 passed, exactly one replicate was purchased. Initial state hash
equals the historical smallhead `I1` fingerprint (`c6b838fc...`). The run
early-stopped after 159 evaluations (9,063 steps; best 800 MAE 0.205394 at step
6,783). Frozen estimators:

| quantity | seed0 | seed1 |
|---|---:|---:|
| RAW probe MAE (3600) | 0.181330 | 0.176472 |
| SOUP probe MAE (3600) | 0.176633 | 0.175501 |
| `G_soup(36->72)` | +0.049116 | +0.051234 |
| `G_raw(36->72)` | +0.049652 | +0.048615 |

Two-seed mean `G_soup = +0.050175`; two-seed mean paired bootstrap 95% CI
`[+0.043256, +0.057999]`, lower `>0`; mean `G_raw = +0.049133 > 0`. Both
per-seed gains exceed `+0.005` and the mean exceeds `+0.006`, so the verdict is
`REPLICATED_MATERIAL_NEAR_7200_DATA_SCALING` and N=1800 seed0 is purchased.

## 18. Conditional 1800 point (Stage 3) — N1800_I0T0

Stage 2 replication authorized N=1800 seed0. Initial state equals the
historical `I0` fingerprint. The run early-stopped after 115 evaluations
(6,555 optimizer steps, 437.0 effective epochs), best 800-selection MAE
0.249677 at step 4,275. Frozen estimators:

| estimator | 1800 seed0 | 3600 seed0 |
|---|---:|---:|
| RAW probe MAE | 0.226539 | 0.181330 |
| SOUP probe MAE | 0.225184 | 0.176633 |

`G_soup(18->36) = +0.048551`, paired 95% CI `[+0.041398, +0.055593]`,
`P(>0)=1.000`; `G_raw = +0.045209`. This satisfies the `+0.006` material gate
with positive RAW direction, so N=1800 seed1 is purchased.

## 19. Three-point curve (if available)

Stage 4 (`N1800_I1T1`, historical `I1`) early-stopped after 133 evaluations
(7,581 steps, 505.4 effective epochs), best 800 MAE 0.271737 at step 5,301.
`G_soup(18->36, seed1) = +0.064516`, CI `[+0.056034, +0.072706]`.

Two-seed learning curve (SOUP probe MAE):

| N | seed0 | seed1 | seed mean | mean exposures / example |
|---|---:|---:|---:|---:|
| 1800 | 0.225184 | 0.240017 | 0.232600 | 437.0 / 505.4 |
| 3600 | 0.176633 | 0.175501 | 0.176067 | 208.4 / 312.5 |
| 7200 | 0.127517 | 0.124268 | 0.125893 | 240.0 / 240.0 |

Two-seed mean doubling gains: `18->36 = +0.056533`,
`36->72 = +0.050175`, average `+0.053354`. The per-doubling gain is roughly
constant (if anything slightly larger at the small-N end), i.e. the curve is
close to linear in `log2(N)` over 1800-7200. All four new runs are
`FULL_THREE_POINT_TWO_SEED_REPLICATION`.

## 20. Doubling-gain interpretation

The near-7200 and near-1800 doubling gains are both ~`+0.05` to `+0.06`, an
order of magnitude above the pre-registered material threshold and larger than
the entire ~0.02 architecture-gap scale. The two-seed average per-doubling gain
is `+0.053354`. The current fixed family is therefore *strongly* data-limited in
this regime, and the gain is not shrinking quickly over the observed range.

## 21. Train/probe gaps

For the 3600 runs the generalization gap is large. Seed0: RAW train 0.073954 vs
probe 0.181330 (gap 0.107376); SOUP train 0.048170 vs probe 0.176633 (gap
0.128464). For the 1800 runs the gap is larger still: seed0 SOUP train 0.038626
vs probe 0.225184 (gap 0.186558); seed1 SOUP train 0.041117 vs probe 0.240017
(gap 0.198900). At N=7200 the anchor gap is smaller (~0.075). The gap grows
monotonically as N shrinks. This is descriptive only: a larger gap at smaller N
is consistent with the same function class fitting fewer examples less
transferably, and is not by itself proof of overfitting.

## 22. Compute / exposure accounting

Compute is matched on optimizer steps (max 13,680) and evaluation cadence (every
57 steps). Because smaller N packs the same step budget into fewer unique
examples, per-example exposure rises: `mean_exposures_per_example = effective_epochs`.
The audit reports unique examples, optimizer steps, effective epochs and mean
exposures per unique example per run (`compute_audit.json`). This is a
**compute-matched data-size scaling** curve, not a fixed-epoch learning curve.

## 23. What is and is not sample-efficiency evidence

This is evidence that seeded within-family performance depends strongly on the
number of unique examples. It is **not** evidence that compact-v4 has inferior
inductive bias relative to CIN/GNN or any external model: a cross-family
sample-efficiency comparison requires the same subsets, same probe and another
representation family, which this audit explicitly does not run.

## 24. Why no power-law extrapolation is claimed

With only three (or fewer) sample sizes, any nonlinear fit with an asymptotic
floor is unstable. A descriptive `MAE = a + b log2(N)` is reported, but no
extrapolation to 14.4K / 28.8K / infinite data and no scaling-exponent claim is
made.

## 25. Relation to the ~0.02 architecture-gap scale

The rough contextual scale `F_gap = mean G_{36->72} / 0.02` is `~2.5` here.
This is a *rough contextual scale only* (the `.02` figure is an
architecture-gap order of magnitude and cross-model protocols are not unified).
It does not license the claim "2.5x more data would close the benchmark gap".

## 26. Implication for future inductive-bias design

A strong, replicated within-family data-limited regime means future work should
ask whether a new inductive bias can reach the same performance with fewer
examples, rather than whether a new module can recover `0.002` at full data. The
next candidate must explain *why existing structural information becomes easier
to use from fewer samples* (parameter sharing, reusable composition,
constrained interaction structure, factorized functional form) — not "more
information".

## 27. Existing NO-GOs that remain binding

All prior NO-GOs stay closed: P1 learned composer, P2 one-shot relation refresh,
compact-v4-cell cycle object, pair endpoint association, centre-incidence
co-occurrence, covariance, triadic binding, function-basis accessibility,
corrected tokenizer performance, topology–attribute factorization, larger
radius, attention/message passing as missing-information repairs, EMA/SWA/K
sweeps. This audit adds no structural feature.

## 28. Official-valid/test lock

Official valid is never loaded. Official test is never loaded, even after any
stage. `integrity_gates.json` records `official_valid_used=false` and
`official_test_loaded=false` throughout; a module-level firewall blocks both
extraction paths.

## 29. Final verdict

**Decision Case A — `STRONG_WITHIN_FAMILY_DATA_LIMITED_REGIME`.**

* Stage 1 material: `G_soup(36->72, s0) = +0.049116`, CI lower `+0.042053`,
  `G_raw = +0.049652`.
* Stage 2 replicated: seed1 `+0.051234`; two-seed mean `+0.050175`, CI lower
  `+0.043256`; mean RAW `+0.049133`.
* Stage 3 material: `G_soup(18->36, s0) = +0.048551`, CI lower `+0.041398`,
  RAW `+0.045209`.
* Stage 4 replicated: seed1 `+0.064516`, CI lower `+0.056034`.
* Average per-doubling gain `+0.053354`; all four new runs `early_stopped=true`,
  none boundary-pinned, no `OPTIMIZATION-BUDGET AMBIGUOUS` flag.

Meaning: over 1800-7200 unique training molecules, each data doubling buys
roughly `0.05` probe MAE for the fixed 82K compact-v4-smallhead recipe. The
family is strongly data-limited near the current ZINC training-set scale.

What this does **not** say: it does not prove that more data would necessarily
close the benchmark gap; it does not prove inferior inductive bias relative to
CIN/GNN; and it does not authorize adding structural features or capacity.

Authorized next step: sample-efficiency-oriented inductive-bias **design**
(`authorized_for_design=true`, `hypothesis_family=sample_efficiency_inductive_bias`,
`full_training_authorized=false`) — i.e. asking why the same existing
structural information could be learned from fewer examples.
