# REPORT — cscl-bjoc-source-ownership-v1

Frozen protocol: `configs/bjoc_source_ownership_v1.yaml`
(`protocol_freeze.json` sha256 `cc92184b74df5587…`, git `ef285af5`)
Track: `tracks/structure-contribution` · namespace `results/bjoc_source_ownership_v1`
Execution regime: host `a100-2`, 2× NVIDIA A100-SXM4-40GB, driver 550.163.01,
torch 2.5.1+cu124, python 3.12.14 (process backend, pools `res-gpu0`/`res-gpu1`).

Preregistered questions: (Q1) does source-level Junction ranking carry structural
signal beyond encoder-anomaly / rarity baselines; (Q2) does real
Junction-to-Bond ownership add independent representation value over the
count-matched sham and the no-ownership grouping; (Q3) can the ownership-resolved
local pathway keep BJOC-R3 parity while supporting more trustworthy local
explanations; (Q4) next step.

**Headline.** Phase 0: `SOURCE_SIGNAL_BEYOND_SIMPLE_BASELINES` — all five frozen
gates pass. Phase 2 seed 0: `SEED0_GATES_FAIL` — ownership gives **no confirmed
predictive gain** (O−N +0.00301, O−S −0.00199, both CIs spanning 0); seed 1 was
**not purchased** per the frozen stop rule. Final decision label:
**`OWNERSHIP_NOT_MORE_VALUABLE_THAN_SHAM`** (secondary reading `OWNERSHIP_NULL`
for the O−N contrast). Mechanism: O's local pathway is genuinely used and is the
most faithful of the three arms, but usage ≠ predictive value.

---

## 1. Phase 0 (Q1) — frozen model, label-free, official-train structures

Inputs: official train 10k objects + frozen R3 soups (seed 0/1), diagnostic rows
0–255 / reference rows 256–1279 from the source-attribution audit manifest; no
label was loaded (`labels_used: false`). Outputs: `phase0_results.json`,
`phase0_decision.json`.

**0A math facts (verified numerically).**
* single-junction IG is the exact single-junction baseline replacement
  difference: `IG_j + occlusion_j` max relative error 9.8e-05 (seed 0) /
  6.6e-05 (seed 1); rank correlation `−0.9995`. Sign convention:
  `occlusion_j = pred(w_j←mu_G) − pred_nat`, so `IG_j ≈ −occlusion_j`.
* joint reset of all junctions leaves the graph first moment unchanged and
  collapses the second moment to `mu_G²` (max abs error **0.0** for both
  moments); it probes **encoded-variance removal**, not mean-level contribution.
  The joint-reset prediction shift is +0.124 (seed 0) / +0.179 (seed 1), while
  the signed sum of single-junction IGs is −0.108 / −0.152 — the two are not
  interchangeable (curvature), as preregistered.

**0B simple-baseline exclusion** (512 molecule×seed units, top-4, matched
natural-encoding replacement α=1, 16 random draws, 8 matched controls per
junction). Top-4 response means: IG 0.0871, dev 0.0805, rare 0.0542, random
0.0367.

| contrast | mean Δ (IG − other) | CI95 | one-sided sign p |
|---|---|---|---|
| IG − dev (encoder anomaly) | **+0.00665** | [+0.00484, +0.00856] | 1.2e-08 |
| IG − rare (signature rarity) | **+0.03297** | [+0.02915, +0.03672] | 7.8e-57 |
| IG − random | **+0.05038** | [+0.04638, +0.05458] | 2.7e-98 |
| IG top − matched controls | **+0.01094** | [+0.00262, +0.01924] | 0.00199 |

Matched control: 366 molecules / 640 top-4 junctions covered
(coverage 0.3125; same-molecule `(signature, degree, ring)` with dev-decile
preference; the 4-way structural upper bound is 0.62 by construction —
AMENDMENT_01 changed coverage from a gate to a reported diagnostic before
execution). Dev-decile-exact fraction 0.073; the comparison therefore controls
**structure fully and encoder-space dev only partially**. Top-4 Jaccard vs
dev/rare 0.29. Composition shows IG is not a re-selection of the dev ranking
(mean dev of IG top4 5.18 vs 5.93 for the dev ranking).

**0C deterministic signature** `(center type, sorted endpoint types, sorted bond
categories, center-in-ring, n ring bonds)`: 225 classes over 346,358 corpus
junctions (155 frequent, 28 insufficient). Top-decile |IG| enrichment statistic
**6.45** vs permutation null p95 2.89 (`p=0.001`, 1000 within-(molecule, seed)
draws); centered-IG sign agreement **0.483** vs null p95 0.059 (`p=0.001`).

**Phase-0 verdict** (frozen rule): `P0_BEYOND_DEV`, `P0_BEYOND_RARE`,
`P0_MATCHED`, `P0C_ENRICHMENT`, `P0C_SIGN` all pass →
**`SOURCE_SIGNAL_BEYOND_SIMPLE_BASELINES`**. The strongest signature classes are
all `center_type 0` contexts such as `[0,[0,0],[1,2],1,2]` (corpus 75,860,
mean|IG| 0.017, sign agreement 0.70) and `[0,[0,2],[1,1],0,0]` (corpus 9,930,
sign agreement 0.61) — ring-context junctions with distinct roles, consistent
with the known nonlocal mixing of the Null grouping (any `w_j` perturbs all
bonds), so this is **encoded source importance in a frozen model, not chemical
causality**.

## 2. Phase 2 (Q2) — N / O / S ownership arms

Design: one class, identical parameters (466,017 exact), shared scratch init
`d06d5e16…`, shared recorded SSL encoder `2c2bbd3f…`, shared batch order
`d142e307…`, frozen R3 recipe (360 epochs cosine 1e-3→1e-5, raw-y L1, batch
128, clip 5, last-5 soup). Training provenance:

| arm | run | commit | soup fit MAE | wall |
|---|---|---|---|---|
| N (no ownership, `n_e·mean_G(w)`) | `own-s0-N-…-65313f5e` | 18b3fd8 | 0.00108 | 558.5 s |
| O (true ownership, `Σ_{j∋e} w_j`) | `own-s0-O-…-abf6bfd9` | 18b3fd8 | 0.00073 | 632.0 s |
| S (count-matched shuffle) | `own-s0-S-…-9db9819c` | 0d0ff8ab | 0.00131 | 503.4 s |

Total GPU 28.2 min (cap 1.5 h); the first S attempt
(`own-s0-S-…-f507fd49`, 18b3fd8) was cancelled at 16 s/epoch before writing any
soup and is documented in AMENDMENT_02 (vectorized sham, bit-exact equivalence
test; N/O paths untouched).

**Monitor (exposed exploratory split, single declared read, lock
`monitor_read.lock`, y sha256 `925d3afe…`; 1000 molecules):**

| arm | Monitor MAE | RMSE |
|---|---|---|
| N | 0.09724 | 0.33783 |
| O | 0.10025 | 0.40685 |
| S | 0.10224 | 0.51793 |
| P3-D s0 (frozen reference) | 0.10396 | 0.49317 |

| frozen seed-0 gate | result | pass |
|---|---|---|
| O−N mean ≤ −0.002 and CI95 upper < 0 | **+0.00301** [−0.00269, +0.00990], p1 0.913 | ✗ |
| O−S mean ≤ −0.002 and CI95 upper < 0 | **−0.00199** [−0.01179, +0.00507], p1 0.636 | ✗ (misses the mean bound by 1e-5) |
| ordinary (y>−3) and top-10-removed O−N < 0 | −0.00043 / −0.00047 | ✓ (small) |
| MAE(O) − MAE(P3-D s0) ≤ +0.004 | **−0.00371** | ✓ |

**Decision: `SEED0_GATES_FAIL`.** Per the frozen seed-1 stop rule, no seed-1
arm was purchased (`OWNERSHIP_LOCAL_VALUE_SEED0_ONLY` does not apply because no
seed-1 confirmation was run). The point estimates place N < O < S: ownership is
not better than no-ownership (it is 0.003 worse, n.s.) and is statistically
indistinguishable from the count-matched sham (−0.002, n.s., CI overlapping).
The MAE ordering hides a tail difference: O's RMSE is 0.4069 vs N 0.3378 and
S 0.5179 (P3-D 0.4932), i.e. O's small MAE gain/loss is mixed with a heavier
error tail than N. Both readings use only the frozen vocabulary; the headline is
`OWNERSHIP_NOT_MORE_VALUABLE_THAN_SHAM` because the sham is the design's
specificity control and the primary contrast also failed. RMSE/tail details are
in `monitor_metrics_seed0.json`.

## 3. Mechanism (Q3) — trained-soup audit, 128 Monitor molecules, CPU

| arm | local-pathway deletion Δ | global-block Δ | faithfulness Spearman (mean) | ≥0.5 fraction | ‖c_e‖ mean |
|---|---|---|---|---|---|
| N | 0.408 | 0.136 | 0.485 | 0.50 | 1.77 |
| **O** | **0.532** | 0.026 | **0.598** | 0.73 | 3.02 |
| S | 0.022 | 0.563 | 0.440 | 0.39 | 1.64 |

All frozen mechanism gates pass for O: local pathway used (0.532), global block
cannot change `c` (bit-exact under random `p_global` in every arm), trained
ownership-intervention isolation (5 molecules checked on O, exact outside
touched bonds), faithfulness 0.598 ≥ 0.5. Observations:

* O uses the local ownership channel most and its graph-level block least;
* **S effectively abandons the corrupted local channel** (deletion Δ 0.022) and
  routes prediction through the global content pathway (Δ 0.563) — a mechanism
  explanation for why the sham stayed close to N/O: the shuffle destroyed
  locality and the model fell back to graph-level content, losing only 0.002;
* N still reads its local channel (0.408) but with lower faithfulness (0.485).

**Interpretation limits (frozen wording).** A used local pathway is not an
explanation claim; `c_e` faithfulness is a property of the *local pathway
contribution* (single-bond deletion of `c_e`), not whole-model local
explainability; `p` is a graph-level pathway and is never reported as a sum of
local contributions.

## 4. Secondary descriptive comparison — official valid (exposed dev set)

Replay guard first: frozen recipe-tuning R3 reproduces its historical valid MAEs
exactly (seed 0 0.079101, seed 1 0.080586, tolerance 1e-5), so the evaluation
path is validated. Seed-0 valid MAE: R3 0.07910, N 0.09216, O 0.09331,
P3-D 0.09438, S 0.09639. Same ordering as Monitor; this set was **not** used for
any selection.

## 5. Answers

**Q1 — Does source-level Junction ranking carry structural signal beyond simple
baselines?** **Yes, as an encoded-source measurement.** Under the same matched
independent intervention, `|IG|` top-4 junctions move the frozen R3 Null
prediction more than the encoder-anomaly (`+0.0067`), signature-rarity
(`+0.0330`) and random subsets (`+0.0504`), and more than same-molecule
structure-matched controls (`+0.0109`, p=0.002, 640 junctions/366 molecules).
The signal is enriched in interpretable signature classes (max enrichment 6.45×,
p=0.001; centered-IG sign agreement 0.483, p=0.001) and is *not* the anomaly
baseline re-selected (Jaccard 0.29). Caveats: matched controls cover 31% of
top-4 junctions (structural upper bound 62%) and only 7% match the dev-decile,
and the frozen Null grouping is nonlocal, so this is not chemical causality.

**Q2 — Does real ownership add independent representation value?** **Not
confirmed.** With identical parameters, init, SSL, data, batch order and budget,
O did not beat N (`+0.0030`, CI [−0.0027, +0.0099]) or the count-matched sham S
(`−0.0020`, CI [−0.0118, +0.0051]); the parity gate vs P3-D passed (−0.0037) and
the ordinary/top-10-removed directions were negative but negligible. The
pre-registered seed-1 confirmation was not purchased. Final label
`OWNERSHIP_NOT_MORE_VALUABLE_THAN_SHAM` (secondary `OWNERSHIP_NULL` for O−N).

**Q3 — Can the ownership-resolved local pathway keep parity while supporting
more trustworthy local explanations?** **Parity: yes; trust: only at the
pathway level.** O keeps BJOC-R3/P3-D parity (Monitor MAE +0.0037 vs P3-D, valid
0.0933 vs R3 0.0791 descriptive), uses its local channel the most (Δ 0.532,
‖c_e‖ 3.02) and has the best bond-attribution faithfulness (Spearman 0.598 vs
0.485/0.440 for N/S). But because ownership bought no predictive gain, this
supports only "the local pathway is used and internally more faithful", not
"O gives better explanations of the model's decisions". The sham result is the
cautionary control: corruption of ownership is **not** detected in the metric
(the model bypasses the local channel), so an unexplained "relational" module
can be silently unused.

**Q4 — Next step: Motif interpretation / structured prediction / stop?**
**Motif interpretation.** Phase 0 establishes a reproducible, baseline-beyond
source signal with interpretable signature classes; Phase 2 shows that merely
wiring the true incidence as a sum does not convert that signal into predictive
gain (and the model can route around it). The cheapest informative next step is
therefore to characterize the enriched signature classes / ring contexts
(motif-level interpretation of the frozen model's encoded importance), not to
buy another structured-prediction head on a signal whose relational form just
failed, and not to stop while Q1's signal is unexplained.

**Warning answered explicitly.** A module being *used* (local deletion Δ,
attribution norm) is not an explanatory claim; a *local pathway* (bond-anchored
`c_e`) is not whole-model local explainability. This round reports usage,
isolation, faithfulness of the local pathway and graph-level pathway separation
as separate, narrowly-scoped facts.

## 6. Discipline and limits

* official test: never loaded (`official_test_loaded: false` in every artifact);
  Locked split: never read; official valid: exposed development set, descriptive
  only; Monitor: exposed exploratory split, one declared label read
  (`monitor_read.lock`), no seed-1 read.
* no search of any kind (no new baselines, features, widths, hyperparameters);
  R3 recipe and P3/SSL checkpoints reused read-only.
* amendments recorded before their affected analysis: `AMENDMENT_01.md`
  (random-contrast aggregation; matched-control coverage semantics; matched-unit
  label fix), `AMENDMENT_02.md` (sham vectorization with bit-exact equivalence
  test; trained-soup mechanism completeness; valid-secondary seed availability).
* GPU budget: 28.2 min completed + ~10.5 min cancelled first S attempt
  (< 1.5 h cap).
* artifacts: `phase0_results.json`, `phase0_decision.json`, `cpu_gates.json`,
  `p3_replay.json`, `valid_reference.json`, `train_*_D_s0.json`,
  `soup_*_D_s0.pt`, `predictions/`, `prediction_manifest.json`,
  `monitor_metrics_seed0.json`, `mechanism_s0.json`, `valid_secondary.json`,
  `final_decision.json`.
