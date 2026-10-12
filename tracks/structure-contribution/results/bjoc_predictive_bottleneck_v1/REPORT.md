# cscl-predictive-bottleneck-diagnosis-v1 — report

**Decision: `SAMPLE_EFFICIENCY_BOTTLENECK_SUPPORTED`** (bottleneck identified; frozen rule
`configs/bjoc_predictive_bottleneck_v1.yaml :: decision_rules`). A single-sentence conclusion:
*at fixed optimizer updates, doubling the number of unique labelled molecules from 4k to 8k buys
`-0.0373` locked MAE (seed-mean, CI95 `[-0.0435, -0.0310]`, both seeds negative, survives
top-10-error removal at `-0.0365`), which is ~4x the extra-optimization effect (`-0.0087`) and
~6x the junction-ownership effect (`-0.0064`, seed-unstable: seed1 `+0.0009`); the BJOC-Null
plateau is therefore dominated by labelled-sample quantity / sample utilization, not by junction
ownership and not by an optimization budget deficit.*

Historical context (mandatory): the previous BJOC-Null full-10k **official-test seed-mean is
0.092233**, against the user target band **0.06–0.07**; the remaining gaps are
`0.022233` (to 0.07) and `0.032233` (to 0.06). This round ran on the official **train** split
only, on a new internal grouped split, and **gives no extrapolation to the official test**:
internal locked MAE levels in this report are not comparable to the official-test 0.092233, and no
claim is made that any additional number of molecules reaches any specific official-test MAE.

Protocol summary: new label-blind grouped split `cscl-predictive-bottleneck-v1-A`
(Fit8k 8000 / Monitor 1000 / Locked-dev 1000, nested Fit4k 4000), frozen before any GPU run.
Six trajectories (N4-s0, N4-s1, N8-s0, N8-s1, O8-s0, O8-s1), 413,505 parameters each, raw-y L1,
identical recipe; **T = 15120** updates for every trajectory and readout
(63 steps/epoch at 8k, 32 at 4k; N4 runs 473 epochs). Soups are the arithmetic mean of five
pre-frozen last-epoch snapshots; no checkpoint was selected on monitor; locked labels were read
exactly once after all predictions were hashed (`predictions_meta.json`, `locked_read.lock`).
Budget actually used: 6 trainings ≈ 0.64 GPU-h, +≈2 min prediction, total GPU < 0.7 h
(limit 1.5 GPU-h); wall-clock well inside the 7h45m budget.

## 1. Learning curve (all readouts at matched update count T = 15120)

Per-seed readouts (raw-y MAE):

| readout | fit (s0 / s1) | monitor (s0 / s1) | locked (s0 / s1) |
|---|---:|---:|---:|
| N4-short | 0.070804 / 0.039896 | 0.198718 / 0.195894 | 0.160819 / 0.150557 |
| N4-stepmatched | 0.032230 / 0.026350 | 0.190899 / 0.193662 | 0.148153 / 0.145891 |
| N8 (BJOC-Null) | 0.041599 / 0.037795 | 0.152880 / 0.149511 | 0.114813 / 0.104624 |
| O8 (BJOC-Organized) | 0.033881 / 0.035511 | 0.142115 / 0.146997 | 0.101115 / 0.105485 |

Seed-mean locked distribution numbers: N4-short 0.155688 (RMSE 0.3318, median 0.1027, p90 0.3093);
N4-stepmatched 0.147022 (0.3260, 0.0937, 0.2961); N8 0.109719 (0.3060, 0.0675, 0.2126);
O8 0.103300 (0.2923, 0.0627, 0.2009).

Frozen primary contrasts on **locked** (paired molecule bootstrap, 5000 resamples; negative =
candidate better):

| contrast | seed-mean Δ | CI95 | per-seed | top-10 removed | ordinary y>-3 | tail y≤-3 |
|---|---:|---|---:|---:|---:|---:|
| N4stepmatched − N4short (extra optimization) | **-0.008665** | [-0.011539, -0.005733] | -0.012666 / -0.004665 | -0.008210 | -0.007113 [-0.0100,-0.0044] | -0.029940 [-0.0456,-0.0143] |
| N8 − N4stepmatched (**data at equal updates**) | **-0.037304** | [-0.043517, -0.031045] | -0.033340 / -0.041267 | -0.036507 | -0.034592 [-0.0400,-0.0292] | -0.074469 [-0.1269,-0.0216] |
| N8 − N4short (data at equal epochs) | -0.045969 | [-0.052276, -0.039582] | -0.046005 / -0.045933 | -0.044578 | -0.041705 [-0.0472,-0.0362] | -0.104409 [-0.1559,-0.0534] |
| O8 − N8 (ownership) | -0.006419 | [-0.010001, -0.002772] | -0.013698 / **+0.000861** | -0.005735 | -0.007087 [-0.0103,-0.0038] | +0.002741 [-0.0249,+0.0305] |

Monitor readouts (descriptive, standard training-time readout only) show the same ordering:
N4stepmatched−N4short `-0.005026` `[-0.008097,-0.002093]`; N8−N4stepmatched `-0.041085`
`[-0.053109,-0.031782]`; O8−N8 `-0.006639` `[-0.010557,-0.002846]`.

Reading: 4k models fit the training data *better* than 8k models (N4-stepmatched fit 0.029 vs N8
0.040 at the same update count) while generalizing far worse (locked 0.147 vs 0.110) — the 4k
trajectory memorizes. Extra optimization on 4k (`N4stepmatched − N4short`) reliably helps a
little, so the data effect is not an artifact of the N4-short soup position; it is still ~4x
smaller than the data effect. The data effect is broad (top-10 removal, ordinary-`y` subgroup and
both seeds all confirm it) despite the new monitor split containing two extreme-tail rows
(y = -21.9, -42.0) that are absent from the locked split. The frozen sample-efficiency rule fires
on all four conditions (`≤ -0.006` seed-mean, both seeds negative, CI95 upper < 0, top-10 ≤ 0).

## 2. Null vs Organized (primary: paired locked)

| split | quantity | N8 | O8 | Δ (O−N) | CI95 | better-mol frac |
|---|---|---:|---:|---:|---|---:|
| locked | seed-mean | 0.109719 | 0.103300 | **-0.006419** | [-0.010001, -0.002772] | 0.550 |
| locked | seed0 | 0.114813 | 0.101115 | -0.013698 | [-0.019306, -0.008186] | 0.560 |
| locked | seed1 | 0.104624 | 0.105485 | **+0.000861** | [-0.004135, +0.005795] | 0.496 |
| monitor | seed-mean | 0.151196 | 0.144556 | -0.006639 | [-0.010557, -0.002846] | 0.569 |
| monitor | seed0 / seed1 | – | – | -0.010765 / -0.002514 | – | – |

Locked subgroups: ordinary `y>-3` (n=932) Δ `-0.007087`, CI95 `[-0.0103,-0.0038]`;
tail `y≤-3` (n=68) Δ `+0.002741`, CI95 `[-0.0249,+0.0305]`; top-10-error-removed `-0.005735`.

Frozen ownership rule verdict: **does not fire** — the seed-mean and CI conditions hold, but
seed1 Δ is positive (`+0.0009`), so the "both per-seed deltas < 0" condition fails
(the failure is recorded as `near_miss_signals` in `scientific_decision.json`). The ownership
direction is a weak average signal, not a robust effect at this budget.

Was the historical N/O comparison dominated by individual molecules or statistical noise?
(`historical_ownership_audit.json`; only same-split/same-seed/same-recipe comparisons are used.)
- Cross-bond split-A seed0 monitor: Δ_O−N `-0.008423` CI95 `[-0.017744, -0.001120]`, only 53.4%
  of molecules better; top-1/5/10-removed `-0.005132 / -0.003558 / -0.004404` (the effect shrinks
  by ~half after removing 10 rows); tail `y≤-3` moves the other way (`+0.0151`);
  the worst single row has N error 3.651 vs O error 0.355. Yes — that historical signal was
  partly carried by a handful of molecules and a tail reversal, consistent with the new round's
  seed instability.
- Composer matched monitor (same round, both seeds): Δ_O−N `-0.001230` (s0) and `-0.002588` (s1),
  both CIs crossing 0; composer locked s0 `-0.002732` CI crossing 0. Not seed-stable.
- The full-10k official-valid N/O background (`+0.004975`, O worse, CI `[-0.00198,+0.01115]`) and
  the two null-only R0/B arms are retained only as background and are not used for any decision.
- Direction on the new split is negative on both seeds on monitor (`-0.0108 / -0.0025`) and on
  locked seed-mean (`-0.0064`), i.e. same sign as the historical claim but not seed-robust.

## 3. Common errors (locked, four primary models)

Definition: per-model top 10% of `|error|` on the locked split (n=100 of 1000). All four models
were predicted under label-blind freezing before the single label read.

- Four-way intersection: **36/100**; mean pairwise Jaccard **0.414**; fraction of each model's
  high-error set that lies in the four-way intersection is 0.36 for every model.
- Pairwise intersections: N-s0∩N-s1 53, O-s0∩O-s1 58, N-s0∩O-s0 59, N-s1∩O-s1 63.
- Residual correlation between all model pairs is high (`0.907–0.949`), and among shared
  high-error rows the error signs agree `93–98%` of the time: the four models make the *same*
  mistakes on the same molecules.
- Seed stability: N8 top-10% overlap between seeds 53/100 (Jaccard 0.361), O8 58/100 (0.408);
  47 (N) and 42 (O) molecules are high-error in exactly one seed → roughly half of the
  high-error set is seed-specific fluctuation.
- Stable-error fix asymmetry: of the molecules that are high-error for *both* N seeds,
  O fixes only **8/53**; of those high-error for both O seeds, N fixes **2/58**. The O−N
  average gain is a *distribution-wide* shift, not a set of cleanly repaired molecules.
- Largest prediction changes: when O vs N differ most (top-100 `|Δpred|`), seed0 improves
  (`mean Δerr -0.071`, 70% improved) but seed1 does not (`-0.0079`, 54%).
- Four-way high-error structural profile (n=36): 100% ring-containing, 23.1 atoms on average,
  mean cyclomatic rank 3.14, mean junction count 36.3, mean y -2.76; 39% of the locked
  `y≤-3` rows are in the four-way set, i.e. the persistent errors concentrate on large
  ring-rich molecules but are neither exclusively nor mostly tail molecules.
- Classification: the dominant failure mode is a **persistent model-family error** (shared by
  N and O, stable over seeds on ~half the rows), not input indistinguishability (0 rows with
  inconsistent y at identical input, Section 4) and not a training-fluctuation-only effect.

## 4. Input observability

The model input is a deterministic function of the labelled heavy-atom graph: atom block
108-d (raw ZINC atom-type code + degree/ring/junction/radius statistics), bond block 7-d
(PyG `edge_attr` one-hot + ring flags, single undirected bond), junction block 32-d, global block
33-d (topology statistics, standardized with fit-rows-only mean/scale). Absent from the inputs:
stereochemistry/chirality, formal charges, explicit hydrogens, bond E-Z, isotopes, 3D conformer,
and anything outside the labelled graph (solvent/pH).

Audit on the full 10k official-train rows (`graph_collision_audit.json`):
- Strict labelled-graph (VF2) isomorphism groups: **3 groups / 6 rows**, all with **identical y**
  (y-range 0.0) and identical structural input hashes (rows 1034/3646, 1835/6274, 4092/4927).
- Duplicate-SMILES cross-check: 4 groups; 3 are isomorphic with equal inputs and equal y; the
  fourth (rows 941/2264, a charged stereochemical SMILES) is **not** isomorphic and **not**
  input-equal (atom-type codes differ; y-range 0.01974), i.e. the model does see them differently.
- Empirical indistinguishable floor: **0.0** (sum of |y − group median| over affected rows).
- Frozen threshold: a limitation is declared only if ≥20 rows are input-identical with
  inconsistent `y` and floor ≥0.005 → `INPUT_COLLISION_FLOOR_NOT_ESTABLISHED`.
  Conclusion: input observability does **not** limit this model family at the frozen threshold;
  the plateau is not explained by different-y molecules mapping to identical inputs.

## 5. Common error / ownership / observability synthesis

Taken together: (i) the 4k→8k gap is large and broad; (ii) O8−N8 is small and seed-unstable;
(iii) four-way persistent errors share the same molecules across arms and seeds with no
molecule-level "repair" set; (iv) no input collision floor. The BJOC plateau is therefore
dominated by how many unique labelled molecules the fixed non-iterative architecture can
convert into generalization. This does not prove any specific inductive bias is unnecessary;
it says that under the frozen training budget, the missing inductive bias is not the efficiency
bottleneck the ownership mechanism was designed to fix, and structural input information is not
the limiting factor.

## 6. Next-generation direction (minimal next experiment + stop conditions)

Evidence-supported stage judgment: the historical official-test gap (0.022233/0.032233
vs the 0.06–0.07 target) is the objective, but this round licenses only the internal statement
above — no extrapolation to the official test, and no claim about MAE at 20k molecules.

Recommended minimal next experiment (**P1 — label-blind structural utilization**), no new
architecture: add a *discarded* self-supervised head to the frozen BJOC-Null trunk
(mask ~15% of raw atom-type and bond-type codes and predict them from context; ring-membership
as a second masked objective), pretrain for a frozen update count on official-train graphs with
**all labels masked** (no y access, no external data, no teacher), then fine-tune on Fit4k and
Fit8k with the identical frozen raw-y L1 recipe and matched update count. Primary readouts:
pretrained-Fit4k vs frozen N4-stepmatched and pretrained-Fit8k vs frozen N8 on a **new**
pre-registered grouped split (locked read once). Two seeds only.

Stop conditions (pre-register before running): continue the structural-pretraining line only if
both seeds improve monitor MAE by ≥0.015 at 4k and ≥0.005 at 8k over the frozen baselines;
abandon if any seed locked Δ ≥ 0, if the CI95 crosses 0, or if the gain disappears after
top-10-error removal; never select pretraining length or checkpoints on locked labels; if P1
fails, fall back to P2 — a one-shot 4k-only regularization study (fixed EMA of weights, or a
single fixed weight-decay increase) with the same stop rules. If both fail, the next step is an
architecture question, not more data.

Constraint check: the masked-token head is a discarded module on the frozen BJOC-Null trunk
(single-pass encoding unchanged, no GNN/attention/iterative message passing); it never reads y
(labels masked); the fine-tuned model remains the same decomposable BJOC family, so the CSCL
non-iterative / attributable constraints are untouched by P1 and P2.

## 7. Discipline, provenance, artifacts

- Official valid/test were never loaded (22/22 correctness gates pass, including source scans
  `official_splits_never_read`, `raw_y_only_supervision`, `no_historical_model_supervision`;
  `correctness_gates.json`).
- Locked-dev labels read exactly once, after all 16 prediction files were written and hashed
  (`prediction_manifest.json` file sha256 `06d9ec32…`, combined prediction hash `8c9298d8…`;
  `locked_read.lock` records the single locked-dev read);
  the locked set was never used for model selection; the monitor split is the standard
  training-time readout only.
- Frozen artifacts reproduced exactly on the remote worker: split rows
  fit8k `1cf6a2ec…`, fit4k `3f6b5d69…`, monitor `d493dc7f…`, locked `9cbefd5e…`;
  prep `0b807c02…`/`02938c53…`; steps `0efe93c3…`; object cache `08e3ab13…`;
  config sha256 `4769d7bf…`.
- Provenance note (disclosed): `protocol_freeze_manifest.json` was written at the pre-commit
  working-tree revision (driver_sha256 `34e86a87…`, HEAD `67db049`); the committed revision
  `fd26214` that was deployed and trained has driver_sha256 `bc617c2b…` because a driver edit
  occurred between `freeze` and the commit. The config frozen block is byte-identical
  (config sha256 unchanged), the remote `splits/objects/steps` outputs reproduced the frozen
  hashes, and the only post-training edits are analysis-only (`cmd_common_errors`/`cmd_decision`;
  `git diff fd26214` touches no training or prediction code).
- Two seeds cannot estimate the seed distribution; the molecule-level bootstrap does not cover
  training-seed randomness. All conclusions are conditional on the frozen recipe.
- All JSON/npz/pt artifacts in `results/bjoc_predictive_bottleneck_v1/` are git-ignored with
  hashes recorded in the manifests; `REPORT.md` is tracked.
