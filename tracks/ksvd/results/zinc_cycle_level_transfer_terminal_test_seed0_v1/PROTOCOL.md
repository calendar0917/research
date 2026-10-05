# PROTOCOL — `zinc_cycle_level_transfer_terminal_test_seed0_v1`

Frozen **before** any new-head training, before the purchase-gate read, and before the
terminal roster freeze. CPU only (≤ 8 threads, FP32), GPU-hours = 0, new remote jobs = 0.

## 0. Authorisation and honest access record

This round is **authorised by the user prompt** to (i) test exactly one new intervention —
replacing the continuous `c` regression head with a fixed discrete train-`k`-level
classification head on the T25-branch unseen rows — and (ii) read the **official test** for
the already-frozen systems (`SUM_Q`, `B`, `H`, and `C` only if the train-only purchase gate
passes), unconditionally, even if the new head is not purchased.

The official test **is not pristine**: earlier rounds already accessed it
(`zinc_full_decomposition_valid_test_confirmation_v1`, `zinc_compact_v4_*` test closures).
This round is therefore a *terminal reporting* read, never a selection read. No old
historical test score of any system is used to select anything this round. Old-round test
values read during source exploration (before this freeze) belong to a *different* frozen
system generation and are disclosed in `heldout_access.json`.

The valid split is already exposed (source round + prototype round); this round replays it.
The old rounds' "test_access: blocked / no optimizer / test never read" compliance summaries
are **historical facts of those rounds** and are neither inherited nor rewritten here.

## 1. Question

The prototype round (`zinc_cycle_prototype_transfer_cpu_v1`) showed the exact-lookup rule
has no action on the 12 `UNSEEN_FALLBACK` valid rows (largest `Q` errors among them). This
round asks exactly one new question:

> Fixing T25 unchanged, does replacing the continuous `c` regression by prediction of the
> **discrete integer `k` levels that already appear in train** (a fixed 7-level vocabulary,
> cross-entropy, median-level decode) carry **learnable cross-class signal** for rows whose
> exact T25 class was never seen?

Only this encoding/loss/decoder intervention is tested. No message passing, no new graph
features, no body retraining, no extra seeds/ensembles, no second decoder.

## 2. Budget

* Target wall clock ≤ 90 min, hard cap 120 min; no new small-head training after minute 60.
  ≥ 30 min reserved for terminal mapping/forward/statistics/delivery.
* Local CPU total ≤ 8 threads, FP32; GPU-hours = 0; new remote jobs = 0.
* Exactly 3 train-internal class-grouped folds; per fold one continuous head `R` and one
  discrete head `D`: exactly 6 small-head trajectories. One extra `D_full` only if the
  purchase gate passes (≤ 7 trajectories). No body training.
* Engineering restart at most once per logical trajectory, only recipe-identical and before
  that comparison's OOF score is seen; recorded with resources. No score-driven reruns.
* If the CV is incomplete: mark `INCOMPLETE/NO_BUY`; the terminal test of the already-frozen
  SUM/B/H is still delivered unless the source/data engineering itself fails.

## 3. Frozen sources (read-only, identity-verified before anything new)

* `tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1/`
  (execution commit `947d2837c4936ef1276fcbbf0e9189e7d44ace23`): SUM/COMP bodies
  (297,539 params each), frozen `Q` (3,777 params), full-train prep/targets/constants, the
  four endpoint biases, train/valid cached predictions, deploy/inference tools.
* `tracks/ksvd/results/zinc_cycle_prototype_transfer_cpu_v1/` (source commit `5a510d9`):
  REPORT/DECISION/METHOD_CONTRACT, prototype package, prototype table, T25 cache, `b_H`,
  per-row tables, replay tools — the exact 25-dim float32 key and consistent-class rule.
* `zinc_frozen_chemistry_learned_cycle_v1` small-head recipe (structure/optimiser only).
* The official-test input/ID/label mapping **engineering** of
  `zinc_full_decomposition_valid_test_confirmation_v1` is reused (loader path, split
  mapping, env attach, prep application). Its historical model scores are not used.

Identity requirements (verified in phase `identity`): B/H valid `y`, COMP `g`, `Q` raw
replay within 1e−5 prediction tolerance; state/hash exact; float64 identity/metric checks
1e−9. Four-decimal summaries are never used as full-precision truth. No 8k-model, old
constant, or other-generation substitution.

Frozen unchanged: SUM/COMP body parameters, full-train standardisation, tuple phi
scaler/kappa, target decomposition constants, the exact 25-dim float32 key + consistent-class
rule, the existing prototype table, and the train-only biases `b_y_SUM`, `b_y_COMP`, `b_H`
(and `b_g_*` for diagnostics).

## 4. Train-internal class-grouped 3-fold diagnostic (train only)

Only the official-train 10,000 rows' current T25, frozen `c`/`k`. Rows are never randomly
split across a training/validation boundary of the same key.

1. Group rows by the exact T25 key (355 classes), record support.
2. Classes sorted by support descending; ties by canonical key bytes (lexicographic).
3. Each whole class goes to the fold with the currently fewest cumulative rows; ties to the
   lowest fold index.
4. Save class→fold, row→fold, stable IDs and hashes. Each fold's held classes ∩ fit
   classes = ∅. Train conflict classes stay whole in one fold.

The class→fold assignment is label-blind (fixed before any head training); it is never
re-rolled for better rare-`k` results. A level missing from a fold's fit set is reported
honestly, never borrowed.

**Vocabulary**: the unique integer `k` of all 10,000 train rows
(`{-12,-6,-5,-4,-2,-1,0}`, `K = 7`), fixed once, shared by all folds and `D_full`. No
valid/test level ever enters the vocabulary. Every row gets exactly one `R` and one `D`
OOF cycle prediction. The frozen full-train T25 prep/target system is given; per-fold
body/prep is NOT refit; head weights and the regression init median use only the
corresponding fold's fit rows. The all-train source `Q` is never an OOF control; the
all-train-trained prototype's train error is never reported as OOF generalisation.

## 5. Exactly one R/D recipe and one decoder

Shared hidden stack `25→64→32`, SiLU, built by the existing `Q` seed-0 untrained
construction rule (`torch.manual_seed(0)` then the same `Sequential`); `R` and `D` hidden
initial weights/biases are identical item-by-item; the trained `Q`'s hidden weights are
never reused. Output layers differ and are explicitly a whole-output-parameterisation /
supervision / optimisation comparison (no causal attribution to a single mechanism).

* `R`: `32→1`; last-layer weight = 0, bias = `median(c_fold_fit)`; loss `L1(q, c)`.
* `D`: `32→K (7)`; last-layer weight = 0, bias = 0 (initial uniform probabilities); loss
  plain `cross_entropy(logits, class index)` — no class weight, no label smoothing, no
  focal loss, no temperature, no severity sampling.
* Parameter report: `R = 3777`, `D = 3777 + 33·(K−1)` (K = 7 → 3976). No claim that output
  parameter counts match.

Shared training: CPU FP32, seed 0, batch 128, 300 epochs, Adam lr 1e−3 with coupled
weight decay 1e−5, grad clip 5, fixed parameter soup epochs 296–300 (parameter mean). No
scheduler, no early stopping, no OOF epoch selection. Fold `j` batch shuffle uses an
independent generator seed `20261003 + j`; `R` and `D` of the same fold share the same fit
row order and batch stream. Model construction RNG is isolated from the training generator.
`D_full` (only if purchased) uses generator seed `20261003`, all 10,000 rows, 23,700 steps,
same 296–300 soup.

**Decoder (the only one)**: softmax probabilities over the 7 training levels; levels
ordered by `c_level = (k − mu_cycle)/sigma_cycle` ascending; cumulative sum in that order;
output the `c_level` of the first level whose cumulative probability ≥ 0.5 (the median-level
rule). No argmax/expectation/temperature/threshold/mixed-continuous comparison. All
constants from the source full-train targets. No clipping/snapping of the original `Q`
prediction and no per-row boundary set from valid rows.

Per fold fit/OOF reported: `c`-MAE, `k` accuracy/confusion, per-`k` group `n` and error,
missing fit levels. All six trajectories complete and OOF predictions saved **before** any
gate score is computed; no later fold's configuration is changed after earlier folds.

## 6. Train-only purchase gate (pre-fixed, never relaxed)

`gain_c = MAE(R_OOF, c) − MAE(D_OOF, c)` pooled over all 10,000 OOF rows. `D_full` is
purchased only if **all four** hold:

1. pooled OOF overall `c` gain ≥ 0.003;
2. pooled OOF `k = −2` `c` gain ≥ 0.25 (y-contribution units);
3. pooled OOF `k = 0` `c` worsening ≤ 0.001;
4. pooled OOF `k = −1` `c` worsening ≤ 0.05.

If a required group does not exist the condition is not estimable → `NO_BUY`. Per-fold
direction, `k ≤ −3` behaviour and sample counts are reported completely; rare-level
absence or extrapolation failure is never hidden. Descriptive CI: class-level resampling of
the complete T25 classes, 1000 draws, seed 20261011, the same draws shared across R/D; all
rows of a sampled class kept, endpoints computed on the row multiset. Classes with multiple
rows are never treated as independent replicates for primary uncertainty. The CI never
changes the purchase rule.

* PASS → train exactly one `D_full`, then freeze. FAIL or INCOMPLETE → no `D_full`, no
  loss/init/vocabulary/threshold scan, no fourth fold.
* Either way, the terminal test of the already-frozen SUM/B/H proceeds.

## 7. Candidate C (only if purchased) replaces only UNSEEN_FALLBACK

Same COMP body, same prototype table, same `b_H`:

```text
if key in train_table and class is consistent:  q_C = class_median_c     # identical to H
elif key in train_table:                        q_C = frozen_Q(T25)     # identical to H
else:                                           q_C = D_full_decoded(T25)

y_C_raw = h_COMP_raw + q_C_raw
y_C_cal = y_C_raw + b_H
```

All train keys are in the table, so C's train predictions equal H's; re-deriving the train
bias by the fixed train rule reproduces `b_H`; the source `b_H` is the only deploy bias. No
`g`/Q/head offset, no bias alternative. `CONSISTENT_HIT` and `TRAIN_CONFLICT_FALLBACK`
raw/cal equal H's within 1e−5. Prototype thresholds, rounding, support counts, mixing
weights and `k` conditions are unchanged. `C.forward` reads only the on-graph T25 and the
training model package; true current `k`/`c` are scoring-only. Label zeroing/shuffling,
graph order restoration and reload leave the prediction identity unchanged.

## 8. Terminal roster and unconditional test

Roster frozen (committed `terminal_eval_manifest`) **before** any this-round test
prediction/metric is computed:

| system | body | cycle prediction | single y bias |
|---|---|---|---|
| SUM_Q | source SUM | source Q | source `b_y_SUM` |
| B | source COMP | source Q | source `b_y_COMP` |
| H | source COMP | prototype + source Q fallback | source `b_H` |
| C (only if train gate purchased) | source COMP | prototype + conflict Q + unseen `D_full` | same `b_H` |

The roster is decided **only** by the train purchase gate — never by this round's valid
scores. Even if C's valid performance is poor, its test is reported unconditionally; the
gate is never retro-edited; no single system is cherry-picked. If the gate fails, the user
still receives the three frozen systems' test levels.

No new body, no historical-test-score-based roster choice, no ensemble, no second seed.
Every system: 1000 rows per split, official y units and ID mapping, raw and cal both
reported. The single terminal execution first generates all frozen systems' valid/test
predictions, then computes statistics; no recipe change after seeing valid. Only
deterministic statistical corrections or same-model replays after test; never post-test
training, calibration, epoch/rule/routing selection.

The valid split is recorded as exposed/replayed; the test as this round's post-freeze read.
The project has used test before — disclosed; no "first project test"/"test never touched"
claims. Old records stay untouched.

## 9. Test inputs, models, labels

* Test = the official ZINC subset **test** 1000 rows (not validation, not internal-dev,
  not a train subset). Dataset split, stable IDs, original order, graph/label mapping and
  source hashes are recorded explicitly.
* The prior formal test-confirmation mapping/engineering tools are reused; no old model
  numbers, nothing refit. SUM/COMP, `Q`, `D_full` all use the current 10k full-train
  prep/phi scaler/kappa/constants.
* New graphs' env/tuple-incidence/raw features are built by the frozen algorithm
  (input processing, not fitting). Test statistics never update standardisation, kappa,
  vocabulary, prototype table or model buffers. Any appended input payload/index is a
  non-trainable data object only; builder version recorded; the same algorithm verified to
  reproduce train/valid.
* Graph ID is data alignment only, never an input feature. T25/key come from the graph;
  logP/SA/property tables are label diagnostics only, never fed to h/Q/D. Test `k` is never
  derived first to steer routing/prediction.
* Main test `y` scoring uses the official `y` directly. Reliable component labels exist
  (`test_cycle_audit_label.csv` + GVAE property table + frozen train constants): `k =
  round(label_effective_cycle_snapped)`, `c = (k − mu_cycle)/sigma_cycle` (frozen full-train
  constants), `g = y − c`, `ell = (logP − MU_LOGP)/sigma_logP`, `s = g − ell`; `y = g + c`
  checked. No constant is refit. (The CSV's own `label_cycle_component` column uses the old
  8k refine constants and is **not** used; the frozen train constants are applied at the
  scoring side per the fixed row rule.)
* Predictor label-independence/shape/order/replay checks are done on train and the exposed
  valid **before** opening test; test is not instantiated for pre-smoke.
* Post-freeze engineering failures may only be repaired at the data interface/serialisation
  level with proof that weights, feature algorithm, decoder and bias are unchanged; every
  deviation and access recorded. A test looked bad is never a repair reason. If identity
  cannot be proven: `INCOMPLETE/IDENTITY_FAILED`, never a fake confirmation.

## 10. Evaluation: performance, transfer, point effects answered separately

* Main table: valid/test × all systems × `y` raw/cal, `n`, bias; with reliable component
  labels also `g` raw/cal — `g < 0.09` is never conflated with `y < 0.09`. COMP's `g` for
  B/H/C is identical (body unchanged).
* Fixed comparisons: `B − SUM_Q` (component-supervision benefit on test), `H − B`
  (prototype local repair on test), `C − H` (if purchased; new transfer on the uncovered
  branch). gain = reference MAE − candidate MAE, positive = improvement. Every comparison
  raw/cal with paired 95% CI (1000 row bootstrap, seed 20261012, shared indices across
  systems). Witnesses: identical → 0, swapped → mirror, constant shift → inside bound.
* No test purchase gate: test is unconditional reporting. Test is never model-selection
  data; valid conclusions are not re-edited from test. Actual valid−test gaps reported per
  system (no fixed −0.02 conversion). Single seed, no ensemble, extra supervision disclosed.
* C's **valid** evaluation (if it exists): `POINT_EFFECT_PASS` = overall cal gain ≥ 0.003
  AND raw gain > 0 AND G0 cal worsening ≤ 0.001; `CI_SUPPORTED` = overall cal CI lower > 0;
  `SINGLE_ROW_DOMINATED` = largest positive row ≥ 50% of positive sum OR gain vanishes
  after removing H's largest cal-error row; point-pass-but-CI-fail →
  `TARGETED_REPAIR_ONLY`; `y_cal < 0.09` a separate absolute marker on all 1000 rows (no
  row deletion, no covered-only, no oracle). None of these markers edits the frozen test
  roster.
* Routing tables: `n`, cycle error, `y` error, `Σ|err|/N` contributions adding back to the
  overall exactly. C's Δpred vs H on seen routes must be 0; future test hits of conflict
  classes also route to Q. The unseen branch reports predicted-level distribution, true
  levels (available), new conflicts / out-of-vocabulary cases.
* Component errors are never added as a `y` budget: report `e_g`, `e_c` cancellation,
  `triangle_gap = mean(|e_g| + |e_c| − |e_g + e_c|) ≥ 0`, bias separately.
* valid:0935/0214/0249 explained row-by-row only post-freeze, no special-casing; they are
  the motivating rows, not independent discoveries. Test reports B's fixed top-10 errors and
  full group tables; no good-row cherry-picking.

## 11. Mechanism conclusions and discipline

* OOF win → the fixed T25 representation carries learnable **cross-class** signal for this
  encoding; no claim of removing input conflict or proving the dictionary mechanism.
* OOF fail → stop **this recipe only**; train conflicts and level absences listed
  separately; never "all T25 transfer impossible".
* OOF pass but valid/test no improvement → report transfer instability / distribution and
  error-cancellation boundary; no classifier rescue scan.
* C fixing only a few valid tail rows → local repair and broad performance separated; test
  reported as frozen; test never endorses a new configuration.

After the test read this round **ends**. No later optimisation may use this round's test
feedback for hyperparameters/structure/thresholds/roster; new research returns to
train/valid with a new design. No drive-by `s` optimisation, no dictionary addition, no body
retraining. Single-seed, historic valid/test exposure, extra `g/c/ell/s` supervision labels
and the prototype lookup mechanism are disclosed; no scalar y-only baseline or SOTA claim;
H/C are not a `luyin19` fulfilment claim.

## 12. Delivery

Isolated branch `task/zinc-cycle-level-transfer-terminal-test-seed0-v1`; directory
`tracks/ksvd/results/zinc_cycle_level_transfer_terminal_test_seed0_v1/` (this directory).
No push/merge; `main` untouched; no historical directories or others' uncommitted work
modified.

Deliverables: PROTOCOL / METHOD_CONTRACT / REPORT / DECISION / EXECUTION / EVIDENCE_SCOPE /
ERRATA as needed; `source_identity`; T25 group folds; vocabulary/decoder; per-fold R/D
init/soup/curve/OOF predictions; purchase gate; `D_full` (if purchased) + frozen C wrapper;
fixed SUM/B/H package references and biases; `terminal_eval_manifest`; `heldout_access`;
test official mapping; input builder version/provenance; all systems' row-wise valid/test
predictions; main tables; gain/CI; routing/group/cancellation/tail/sensitivity; actual
valid−test gaps; replay/label-independence/shape/identity/contribution checks; budget;
manifest; single-file load and analysis re-run commands; one appended STATE/ledger entry.
REPORT opens with the five direct answers required by the user prompt.

Before finishing: confirm no running/queued jobs, no body training, no drive-by scans; this
round's authorised small-head training and terminal test are recorded truthfully.
