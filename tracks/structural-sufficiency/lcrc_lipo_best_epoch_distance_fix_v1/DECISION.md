# DECISION — `lcrc-lipo-best-epoch-distance-fix-v1`

All Test420 numbers are **`EXPOSED_TEST_EXPLORATORY`** (already-exposed random42 partition, read once
after the checkpoint manifest was frozen). The scaffold split in §5 is a development
distribution-shift evaluation on the same 4,200 molecules. Nothing below is a blind confirmation.

## Decisions

| dimension | decision |
| --- | --- |
| `CHECKPOINT_SELECTION` (soup → best single epoch) | **`SUPERSEDED_BY_BEST_EPOCH`** |
| `CONTEXT_GAIN_OVER_BASE` (ROLEBLIND − BASE) | **`SUPPORTED_ON_RANDOM42__NOT_REPLICATED_UNDER_SCAFFOLD_SHIFT`** |
| `DISTANCE_BUCKET_CONTENT` (DIST-TRUE-FIXED − DIST-BLIND) | **`REJECTED`** |
| `STRUCTURE_AWARE_GATING` (G2L-LOO / G2L-ALL − ROLEBLIND) | **`ABSENT`** |
| `DIST_TRUE_AGGREGATION_BUG` | **`CONFIRMED_AND_FIXED`** (gates 20/20) |
| `PERFORMANCE_SIGNAL` (any model beats BASE reliably on both splits) | **`SPLIT_DEPENDENT`** |

## P0 — the aggregation bug was real, and it invalidated the previous distance result

Frozen rule: the historical aggregation and the corrected one must be *distinguishable* on a
hand-computed case, and the historical one must be shown to be a rescaled copy of each atom's own
representation.

* 6-atom chain `h = [0,1,4,9,16,25]`: expected `c34_0 = 12.5`, `c5_0 = 25`; observed exactly that,
  with `c34 = [12.5, 20.5, 25.0, 0, 0.5, 2.5]` and `c5 = [25, 0, 0, 0, 0, 0]`.
* Historical context vs the atom's own `h`: correlation **1.000** (max abs diff to
  `h_i · #roots/|D_i|` = 8.9e-16). Fixed context: correlation **0.0017**.
* Corrected path matches an independent loop implementation exactly (423 atoms, 2,456 + 6,652 flat
  entries, max abs diff 0.0), is batch/permutation invariant, has gradient support exactly on bucket
  membership, and empty buckets give exact zeros.

**Decision:** `CONFIRMED_AND_FIXED`. Every earlier statement about D34/D5 "distance context" must be
read as a statement about a self-copy, and is superseded by §3 below.

## P1 — retiring the last-five-epoch soup

Frozen rule: for every arm with a historical soup comparator, the best single Validation epoch must
be strictly better than the soup, and the same-round last-five-epoch proxy must be worse than the
best epoch for every model.

| arm | historical soup | this round, best epoch | Δ |
| --- | ---: | ---: | ---: |
| `BASE_s0` | 0.580416 | 0.562861 | −0.01756 |
| `BASE_s1` | 0.576780 | 0.560410 | −0.01637 |
| `ROLEBLIND_s0` | 0.534232 | 0.530030 | −0.00420 |
| `ROLEBLIND_s1` | 0.586765 | 0.548668 | −0.03810 |

Same-round proxy (mean of the last five epoch-end Validation RMSEs) is worse than the best epoch by
0.0119–0.0154 for all six models. ROLEBLIND's 3-seed mean (0.541109) beats the mean of its two soup
values (0.5605) by 0.0194 and cuts the between-seed spread from 0.0525 to 0.0098.

**Decision:** `SUPERSEDED_BY_BEST_EPOCH`. The gain is not uniform — on `ROLEBLIND_s0` it is inside
the 0.0036 run-to-run noise floor — but it never hurts, it removes the pathological seed, and it is
the only checkpoint rule that leaves a verifiable "this state came from this epoch" audit trail.
No soup, SWA or EMA is used anywhere in this round.

## P2 — does context conditioning beat BASE?

Frozen rule: mean paired Validation gain must exceed the measured noise floor (0.0036) with 3/3
same-sign seeds, and the Test mean must not reverse sign.

| evidence | random42 (exposed Test) | scaffold shift (development) |
| --- | ---: | ---: |
| Validation gain, 3-seed mean | **+0.0209** (s0 +0.0328, s1 +0.0117, s2 +0.0180; 3/3) | +0.0081 (s0 +0.0156, s1 −0.0089, s2 +0.0176; 2/3) |
| Test gain, 3-seed mean | +0.0101 (2/3; s1 −0.0157) | −0.0008 (2/3) |

Random42 passes the rule; scaffold shift fails it. The seed-1 reversal appears on both splits.

**Decision:** `SUPPORTED_ON_RANDOM42__NOT_REPLICATED_UNDER_SCAFFOLD_SHIFT`. Context conditioning is
the only mechanism in this round whose random42 Validation gain is bigger than noise, and even that
gain is not a transferable structural-sufficiency effect. Nothing here justifies calling
molecule-level context "structurally sufficient".

## P3 — is the corrected distance context useful? (Q3)

Frozen rule: `DIST-TRUE-FIXED` must beat its parameter-, seed- and gate-matched twin `DIST-BLIND`
(gain `RMSE(blind) − RMSE(true) > 0`) in the mean and in at least 2 of 3 seeds, on Validation.
Only the content differs between the twins: `DIST-BLIND` feeds the per-molecule leave-one-out mean
through the same `W_34`/`W_5`, with the same `has34`/`has5` gates.

| split | gain, s0 / s1 / s2 | wins | mean |
| --- | --- | --- | ---: |
| Validation420 | −0.01264 / −0.00697 / −0.00269 | 0/3 | −0.00743 |
| Test420 `EXPOSED_TEST_EXPLORATORY` | −0.01819 / −0.03159 / −0.00826 | 0/3 | −0.01934 |

The corrected model is *worse than its blind control* on every seed and both splits, and worse than
plain ROLEBLIND on Test (0/3). With 20/20 gates certifying that the aggregation is right and that
the correction actually receives the true bucket content, the failure cannot be attributed to the
code path.

**Decision:** `REJECTED`. The D34/D5 bucket content is not merely uninformative — substituting
distance-restricted neighbourhood means for a molecule-level pooled mean *costs* 0.007 Validation /
0.019 Test RMSE. The mechanism's premise ("local distance shells carry usable signal") is not
supported at this scale and this architecture. Stop the D34/D5 line; do not re-open it with another
readout head without a new, pre-registered reason.

## P4 — do the structure-aware gates (G2L-LOO / G2L-ALL) add anything? (Q4)

Frozen rule: mean Validation gain over ROLEBLIND must exceed the noise floor with 3/3 same-sign
seeds.

| pair | Validation, s0 / s1 / s2 | wins | mean | Test mean |
| --- | --- | --- | ---: | ---: |
| G2L-LOO − ROLEBLIND | −0.00659 / −0.00869 / +0.00285 | 1/3 | **−0.00414** | −0.00722 |
| G2L-ALL − ROLEBLIND | +0.00113 / +0.00142 / +0.01003 | 3/3 | **+0.00418** | −0.00420 |

G2L-ALL passes the sign test but 2 of 3 margins (0.0011, 0.0014) are below the noise floor, and its
Test mean is 0.0042 *worse* than ROLEBLIND, with a −0.0334 Test loss on seed 0. G2L-LOO fails both
sign and magnitude. Under scaffold shift G2L-ALL's Validation gain over ROLEBLIND grows to +0.0276
(3/3) — but its scaffold-Test mean is the worst of the three models (1/3 wins).

**Decision:** `ABSENT` for random42 (no structure-aware gain demonstrated); recorded as a
`VALIDATION_ONLY_LEAD` under scaffold shift, with the explicit warning that the same lead did not
survive the corresponding Test split.

## P5 — how much of the ranking is seed noise?

Measured, not assumed: two literal repeats of one arm (same seed, same code, same recipe) differ by
0.00297 and 0.00424 Validation RMSE with best epochs 181→174 and 277→214, and their 360-epoch curves
are never bitwise identical (CUDA `index_add_` reduction order). Between-seed sd is 0.0014 (BASE) to
0.0108 (G2L-LOO).

Consequences accepted in this round:

* Model differences ≤0.004 Validation RMSE are not evidence (this covers G2L-ALL−ROLEBLIND,
  DIST-BLIND−ROLEBLIND, G2L-LOO−ROLEBLIND).
* Single-seed reporting is optimistic by up to +0.028 Test RMSE for the seed chosen on Validation;
  one model's "best" seed is 0.015 *worse* than its 3-seed mean. Rankings in this round are
  3-seed means only.

## Next directions (at most two, both cheap, neither a new architecture)

**D1 — settle the context effect with more seeds, on both splits (8 seeds, +1.4 A100-h).**
Run ROLEBLIND and DIST-BLIND at seeds 3–6 with the identical frozen recipe and best-epoch rule
(8 seeds total), plus the existing BASE seeds, and report the paired per-seed gain
`RMSE(BASE) − RMSE(ROLEBLIND)` as a mean with a bootstrap interval **over seeds** on random42
Validation and on the scaffold Validation. Decision rule fixed before running: the effect is
*established* only if the seed-bootstrap interval excludes zero on both splits; if it holds only on
random42, record `SPLIT_DEPENDENT_CONTEXT_EFFECT` and stop spending on it. This is cheap because
everything (data, recipe, gates, analysis) is already frozen and automated.

**D2 — no new mechanism without a scaffold-shift control.**
Any future structural mechanism must be pre-registered with (a) a parameter- and seed-matched
blind/global control, (b) the scaffold-split Validation *and* Test readout of §11, and (c) the
measured repeat noise floor of its own arm. Mechanisms that win on random42 Validation but lose to
their control under scaffold shift are not to be pursued further — that is exactly what happened to
the D34/D5 line (P3) and, so far, to the G2L gates (P4).

## Do not repeat

* Do not report a mechanism gain without a parameter-matched control trained from the same
  initialisation, order and dropout stream.
* Do not select checkpoints by Test, and do not report a single seed as a result.
* Do not compare against the historical PNA number (0.554034 / 0.528874) as if it were a matched
  control: different architecture *and* different training protocol.
* Do not re-run the soup: it is strictly dominated here, and it hides which epoch produced the state.
* Do not treat random42 Validation as evidence of structural sufficiency — §11 shows the context gain
  shrinking by 60 % and the G2L lead flipping sign on the scaffold Test split.

## Budget

5.983 A100 GPU-hours of the 14-hour cap (18 formal runs 3.209, scaffold stage 2.402, repeats 0.359,
smoke 0.014), ~4.5 h wall clock. The binding constraint was never GPU time; it was the noise floor and
one silent aggregation bug.

## Scope

Six models × three seeds × 360 epochs on one task (Lipophilicity logD), the frozen random42 split
(read-outs: Validation420; Test420 `EXPOSED_TEST_EXPLORATORY` once, after freezing) and the
deepchem-scaffold partition of the same molecules (development). No early stopping, no soup/SWA/EMA,
no architecture search, no Test-based selection.
