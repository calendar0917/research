# PROTOCOL — `zinc_cycle_prototype_transfer_cpu_v1`

Frozen **before** building any prototype table or reading any candidate-`H` valid prediction.
CPU only, single fixed rule, no new training. This round is a *targeted exploration on an
already-exposed* official-valid split, not an independent confirmation.

## 1. Question

Source round `zinc_component_supervision_fulltrain_confirmation_seed0_v1` (commit `947d283`,
execution commit `947d2837c4936ef1276fcbbf0e9189e7d44ace23`) produced
`CHEM_PASS_DEPLOY_PASS`:

* COMP official-valid calibrated `g`-MAE ≈ `0.08930046`;
* COMP deployable `y = h + Q + b_y` official-valid calibrated `y`-MAE ≈ `0.11740618`;
* COMP vs matched SUM: G0 cal `g` gain `+0.00535389`, overall cal `g` gain `+0.00456326`,
  overall cal `y` gain `+0.00524704`.

COMP is kept as the single-seed working reference and is **not** re-confirmed. SUM is not
re-trained. No component-supervision coefficient scan, no dictionary / binding / aggregation /
reader revival.

This round asks exactly one thing:

> Among the exact **T25** classes that the *current* frozen `Q` sees, which were already
> **train-consistent** (all train members share one integer cycle label `k`) and were
> **mislearned by the fixed small `Q`**? Does building a train-only **cycle prototype**
> (per-class median of `c_train`) for those classes — used by lookup at inference — improve
> the complete deployable `y`?

This is **not** the earlier "correct total residual per topology class" idea: that residual
mixed chemistry error. This round uses the already-separated, train-supervised cycle target
`c` directly. It is **not** a proof of a learned shared structure–attribute dictionary — it is
only a deployable topology-class prototype / lookup reference.

The gap `0.1174 − 0.0893` is **not** directly a removable cycle budget: total error involves
cancellation and calibration differences. This round computes the row-wise identity instead of
assuming that gap.

## 2. Scope, budget and stop

* Target wall clock ≤ 45 min, hard limit 60 min; no new compute step after minute 45, keep
  delivery time.
* Local CPU total threads ≤ 8, FP32 forward; GPU-hours = 0, new remote jobs = 0.
* Training = 0, optimizer/backward = 0. The only new fitting allowed: exact train class
  grouping, within-class `c` median, and a train-only scalar bias.
* Exactly **one** formal candidate `H` (hybrid prototype). Reference `B` is last round's frozen
  COMP+Q. The fixed-`b_B` variant is a *mechanism diagnostic only*, not a second candidate, and
  no result is selected from the two biases.
* official-test is never read — not data, not historical predictions, not metric tables; no
  ensemble, no extra seed, no fold change.
* official-valid has been used by earlier research and last round; this round is a targeted
  exploration on an exposed validation split and is **not** called independent confirmation.
  Existing artifacts are reused; the whole dataset is not re-downloaded.
* If this round gains, only save/report the candidate, do not auto-start a next round; if it
  fails, do not change coverage gate, rule, key, or add smoothing rescue.

## 3. Source and identity: locate the correct local branch first

Read `AGENTS.md` and the repository research-registration rules first.

Only body/source:
`tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1/`
and the same-named experiment runner. Source execution commit `947d283`; full SHA per local
record (`947d2837c4936ef1276fcbbf0e9189e7d44ace23`).

Required from source (all present locally, verified by SHA-256 against
`frozen_eval_manifest.json` / `manifest.json`):

* REPORT/DECISION/METHOD_CONTRACT/EXECUTION/EVIDENCE_SCOPE/manifest — present;
* full-train prep, targets, constants, stable IDs and T25 — present
  (`full_train_prep.npz`, `full_train_targets.npz`, `full_train_payload.npz`);
* COMP `raw_soup`, frozen `Q` `raw_soup`, deploy wrapper, `b_y_COMP`, `b_g_COMP` — present;
* train/valid `h_raw`, `q_raw`, raw `y_cal`/`raw` and the existing `y/g/c/k` evaluation labels —
  present (`valid_frozen_predictions.npz`, `Q_train_predictions.npz`,
  `COMP_raw_predictions.npz`, `calibration.json`);
* `frozen_eval_manifest` and `heldout_access` — present.

No isolated branch / remote-only artifact is needed; everything is local. No new remote job.

Identity checks performed before any `H` construction:

* source `h_raw + q_raw + b_y` reproduces published COMP `y` (tolerance 1e−5);
* `h_raw + b_g` reproduces published `g` (tolerance 1e−5);
* `y = g + c`, `g = ell + s` aligned to stable ID;
* `q_train`/`q_valid` inputs are the **current** `Q`'s actual T25 (not `topology8`, not the old
  8k prep). Reproduced by re-running the frozen `Q` `raw_soup` state on freshly built T25.

Float64 statistic identities use ≤ 1e−9. Prediction tolerance ≤ 1e−5. Float predictions are
not compared byte-for-byte; only state/hash identity is exact. Source tables are reported at
full precision.

Only last round's already-frozen targets and constants are reused: no re-fit of decomposition,
no re-snap, no construction of a more favourable `g`/`c`.

## 4. Pre-fixed key: only the topology25 that the current `Q` sees

The key is the exact equivalence class of the current `Q` input, a 25-dim float32 vector.
Uniform shape `(25,)`, fixed contiguous little-endian float32, `−0` normalized to `+0`,
NaN/Inf forbidden, then deterministic encoding/serialization into a key. The complete vector is
saved alongside, not just a short hash; if a hash is used, no two different vectors may collide.

The source's already-verified "exact model-input key" is reused, but it must agree with the
actual `Q` input's exact equivalence relation. No nearest-neighbour, no rounding grid, no new
distance threshold.

The old discrete `topo_raw` key is **not** used. No `N`/`E`, atom/bond types, SMILES, graph ID,
`k`/`c`/`y`, cycle-snap output, or valid label is added to the key. Model index comes only from
the T25 already used on the graph.

Minimum invariants checked: same T25 → same key; shuffled train graph order restored by stable
ID → same key/prototype; shuffled label fields do not change the input key; full-vector
collision check. Whether the existing T25 is invariant to node renumbering is disclosed per the
implementation source; the round does **not** infer whole-graph permutation invariance from the
lookup effect.

## 5. The single train-only prototype rule

Only last round's official-train 10000 rows' T25, `c_train`, `k_train` are used. Valid
label/coverage never decides which classes are kept.

For each train key `K` store: support `n_K`, all member stable IDs; number of unique `k`,
min/max; median/min/max/span of `c`; the T25 vector and key.

"Train-consistent class": all its integer `k` are the same. `k` comes from the source frozen
labels and passes integer/finiteness verification; no re-snap this round. For a consistent class
`v_K = median(c_train[K])`. The `c` span must match the existing
`c = (k − mu_cycle) / sigma_cycle` numeric precision; if not, mark source inconsistency and stop
locating instead of adding a tolerance to merge classes.

Singletons are also consistent classes and are allowed. No `n ≥ 2/5` support threshold, no
shrinkage prior, no mean substitution, no severity condition, no "only enable for negative-ring
classes". All consistent classes use the same rule, including `k = 0`.

Inference rule, fixed in advance:

```text
if input_key is in train_table and train_table[input_key].k_is_unanimous:
    q_H = train_table[input_key].median_c
else:
    q_H = frozen_Q(T25)
```

Three routing labels: `CONSISTENT_HIT` / `TRAIN_CONFLICT_FALLBACK` / `UNSEEN_FALLBACK`. The
label is decided only by the input key and the train table; valid true `k`/`c` are used only for
post-freeze scoring.

Train `c` for the prototype is training-side supervision, exactly as `Q`'s training `c` was.
Inference queries no current sample's true `c`/`k`; there is no valid key → valid `c` table; no
valid label is copied or saved into the deploy model.

"Train-consistent" does not mean `c` is globally a function of T25; held-out new conflicts can
appear and must be reported.

## 6. Freeze COMP and calibration; compare B / H

Body `h` fully unchanged; `Q` weights fully unchanged. The `g` prediction is therefore unchanged
and only its identity is replayed — no new chemical `g` gain is claimed.

```text
Reference B:
y_B_raw = h_raw + q_B_raw
b_B     = source frozen b_y_COMP
y_B_cal = y_B_raw + b_B

Candidate H:
y_H_raw = h_raw + q_H_raw
b_H     = median(y_train − h_raw_train − q_H_raw_train)
y_H_cal = y_H_raw + b_H

Fixed-bias diagnostic (not part of candidate selection):
y_H_fixedB = y_H_raw + b_B
```

The same train-only calibration rule can produce different bias values. `H` does not stack
`b_g`, a `Q` offset, or another bias; no valid median is used; no reporting is selected by
raw/cal/fixedB ranking.

All rows verified:

```text
y_H_raw − y_B_raw = q_H_raw − q_B_raw
y_H_cal − y_B_cal = (q_H_raw − q_B_raw) + (b_H − b_B)
y_H_cal − y       = (h_raw − g) + (q_H_raw − c) + b_H
```

For both fallback routes the raw prediction difference must be 0 (within forward tolerance);
the fixed-`b_B` prediction difference is also 0. Their main cal result may move by the common
`Δb`; that movement is not claimed as prototype coverage gain.

Train self-match `q_H` error may be near 0 by construction, especially for singletons; this is
not generalization. A leave-one-out coverage diagnostic (no new model) is attached: after
removing the current train row, does the prototype class still exist and stay consistent; how
many singletons lose support. Any LOO value is a diagnostic only, does not replace the formal
`b_H`/prediction rule, and does not treat source `Q` as an independent unseen-row control.

## 7. Freeze order and official-valid scope

Commit PROTOCOL, METHOD_CONTRACT, key implementation, candidate rule, gate and analysis code
first.

Then build the prototype table with train only, compute `b_H`, run train/wrapper checks, save the
frozen `H` model package and `eval_manifest`: COMP/Q/prep/constants/targets/table/keycode/bias/
prediction-rule hashes.

Only then may any `H` valid prediction be generated or its gain inspected. The main line may
reuse the source valid's fixed feature/prediction/label cache to avoid re-loading data; if a CPU
forward is necessary, the same frozen predictor is used and the real source/access time
recorded.

Valid scores already disclosed in the source REPORT may be used for identity verification; even
knowing historical valid `0172` failures, no per-row special case, key change, consistent-class
rule change or support threshold may be added for that row.

The full 1000 valid rows are the main result; no row deletion recipe. There is no hidden
selection of "evaluate valid only if some train marker passes"; after source/engineering checks
hold, evaluation happens once, unconditionally.

This is not the first read of official-valid and does not write "held-out never accessed". It
states explicitly: the exposed valid was reused; the new `H` is evaluated after freezing. test
remains unaccessed.

## 8. Performance gate and single-row gain boundary (fixed in advance)

`gain = MAE(B) − MAE(H)`; positive means improvement.

`PROTOTYPE_DEPLOY_SUPPORT` requires all:

1. valid overall calibrated `y` gain ≥ 0.003;
2. valid overall calibrated `y` paired 95% CI lower bound > 0;
3. valid overall raw `y` gain > 0;
4. valid G0 calibrated `y` worsening ≤ 0.001.

Paired bootstrap 1000 draws, seed `20261010`, shared indices across B/H; main overall and G0 use
their own row sets. Witnesses: identical predictions → 0, swapped → mirror, constant shift →
inside bound. The CI covers only this fixed model on these rows; it excludes training/selection
uncertainty.

Also report the fixed-`b_B` diagnostic; if main cal improves while raw does not, or fixedB goes
reversed, state the calibration dependence explicitly and do not call it cycle-information gain.

Additional boundary markers this round:

* `SINGLE_ROW_DOMINATED`: the largest positive-gain row is ≥ 50% of the total positive gain, or
  after removing B's largest cal error row the total gain is no longer > 0. It does not
  retroactively change the gate; it only limits the generality interpretation.
* `TARGETED_REPAIR_ONLY`: gate points (1), raw (3), G0 (4) hold but CI (2) does not. Keep the
  numeric repair on this fixed valid, do not claim broad transfer confirmation, do not call a
  CI crossing 0 "no effect".
* `DIRECTIONAL_NOT_CONFIRMED`: positive point values but other gates fail; list conditions
  honestly; do not call it equivalent.
* `NO_CANDIDATE`: no net improvement or G0 harm; no threshold/coefficient rescue.
* `IDENTITY_FAILED` / `INCOMPLETE` are separate from a scientific negative.

`H`'s whole-valid calibrated `y < 0.09` is a separate `BENCHMARK_TARGET_OBSERVED` marker; no row
deletion, no G0-only or oracle or `g`-MAE substitute for it.

## 9. Clarity: coverage, conflict, fit and full error computed separately

Only frozen predictions are used for the following; no new fitting probe.

### A. Train table and unidentifiable part

* class count, consistent/conflict class count, row count, singleton count; support/conflict by
  `k` group;
* train T25 within-class best constant-L1 value: per-class `c` median then `Σ|err|/10000` —
  the train empirical fit floor for this exact input only, not a population error floor;
* isolated fallback conflict-class source ID/`k`/`c` and the original `Q` error;
* do not call same-T25 non-isomorphic graphs "full-graph input unidentifiable"; do not turn the
  helper's order dependence into "official labels are necessarily not permutation invariant".

### B. Valid coverage and new conflict

By three routes × `k = 0/−1/−2/≤−3`, report `n`, train support summary, `q_B`/`q_H` MAE vs `c`,
signed error, chemistry `h` error vs `g`, `y` raw/cal, contribution `Σ|err|/1000` and gain.

For `CONSISTENT_HIT`, compute the train-class-`k` vs valid-true-`k` match rate only on the
scoring side; do not change routing from it. List new conflicts; do not hide "train-consistent
but valid different" rows.

For `UNSEEN_FALLBACK`, the candidate by construction cannot supply new cycle information;
distinguish "no coverage" from "covered but not learned". For `TRAIN_CONFLICT_FALLBACK`, explain
that the current T25 does not uniquely determine the train cycle label, not that `Q` capacity is
insufficient.

### C. The true budget and cancellation

* overall and G0 B/H raw/cal, bias, train/valid gap, and the shared `g` result;
* `q`-vs-`c` improvement need not equal `y` improvement; report the opposite-sign rate of
  `e_g = h − g`, `e_c = q − c` and `triangle_gap`, with bias listed separately;
* report `Σ(|e_g| + |e_c|)` and the true `|e_g + e_c|` — component MAEs are never added as a
  total budget;
* route contributions and `k`-group contributions add back to the overall exactly; cal fallback
  movement corresponds to `Δb`, raw fallback movement is 0.
* fixed oracle diagnostic uses B's same `h` and `b_B`, replacing `q → true c`:
  `y_oracle_fixedB = h + c + b_B`. It reads scoring labels only, never enters deployment,
  calibration or gate; it is not a reachable upper bound and is not called a performance result.
  Its difference from B helps describe cycle error / cancellation and is not necessarily
  purchasable gain.

### D. Tail rows and single-point sensitivity

* valid `0172` if a source mapping exists: train exact-class members, support/consistency, `Q`
  input identity, `c_train` prototype / valid `c`, `h`/`g`/`q_B`/`q_H`/`y` errors listed
  per item. Do not presume it is still the largest error row.
* the full valid's B calibrated absolute error fixed top 10, with three routes and H
  improvement/worsening; rows are not chosen by candidate behaviour.
* report two sensitivities excluding valid `0172` and excluding B's largest cal error row; the
  main gate always uses all 1000 rows.
* largest positive-gain row share of positive-gain sum, harm row count, total positive/negative
  gain sum; separate local extreme-tail repair from broad improvement.

## 10. Mechanism and next responsibility; no automatic re-run

Answer separately "is `q` more accurate?" and "is the complete `y` improved?". A `q` error
improvement is never called a deployable gain.

Where evidence holds, one may write:

* on train-consistent, valid-label-consistent exact-hit classes, the prototype lowers `q` error
  while the original `Q` errs: the current function fit / optimization missed a class signal
  already present in train; it is not that this class's input lacks information;
* train-conflict class: the current T25 cannot simultaneously fit these train `c`; adding
  same-input small-head capacity cannot remove the empirical class-constant L1 floor;
* valid new conflict: train consistency does not guarantee transfer; this is T25 alias / finite
  coverage evidence, not "the whole graph has no usable information";
* uncovered: this exact lookup rule cannot act; neither a learnable transfer is falsified nor
  bought;
* `q` error down but `y` not improved: check cancellation and bias, explain numerically, do not
  directly accuse the chemical body of missing information.

This round designs only one next question, not executed:

* if the candidate passes the gate: save `H` as a single-seed deploy candidate, describe the
  extra supervision and lookup mechanism; do not claim luyin19 fulfilled, do not auto-open test
  or a new training run;
* if the gain is single-row only: frame it as tail repair on covered classes and state that
  independent/unknown-class extrapolation remains open; no auto seed extension;
* if the failure is mainly train conflict: a later structure interface must first show that
  train gains distinguishable information, not try to remove this conflict via same-input
  loss/width;
* if the failure is mainly uncovered: only the exact lookup rule's this part is closed; whether
  the current T25 can transfer across classes is not yet judged, and a miss is not automatically
  attributed to missing input information. The next question is a learnable cross-class cycle
  regularity needing a separate design; no auto scan of small heads;
* if most relevant classes are covered and cycle prediction is already accurate while `y` is
  still poor: keep COMP, the next responsibility returns to chemistry error and cancellation; do
  not keep rescuing the cycle head.

No new nearest neighbour, support threshold, shrinkage, regularization or feature is added just
to force a conclusion.

## 11. Engineering, deployable evidence and delivery

Implement an independently loadable frozen `H` wrapper: COMP + original `Q` + train prototype
table + key encoding + the single `b_H`. Standard forward returns `(n,) y`; optional diagnostics
return `q` and routing. Key serialization is deterministic and reloadable; unknown keys
explicitly use the original `Q`.

Minimum checks: train/valid cache identity and ID mapping, `(n,)` shape against broadcasting,
model prediction vs cache ≤ 1e−5; single-file reload; batch/graph-order restoration; query
sample label zeroing/shuffling does not affect forward; fallback with a fake key; conflict-class
fallback; fixed-bias diagnostic; pred/q/bias difference identities. Label-dependence checks
target the query sample and do not misreport the training prototype table's label supervision
itself as leakage.

No full body-feature recompute and no whole-repo test suite. If source caches suffice, compose
them directly and verify the wrapper on a few train/valid graphs; if a needed cache is missing,
do one CPU forward of the same frozen model, still no GPU. Do not pick by cache/native score;
investigate identity first.

New branch `task/zinc-cycle-prototype-transfer-cpu-v1`; new directory
`tracks/ksvd/results/zinc_cycle_prototype_transfer_cpu_v1/`. Register and commit the new
protocol/implementation/artifacts per repository rules; **do not push / merge, do not move
`main`, do not modify historical results or others' uncommitted files**.

Deliver:

* PROTOCOL, METHOD_CONTRACT, REPORT, DECISION, EXECUTION, EVIDENCE_SCOPE;
* `source_identity`, train_table and member/conflict tables, train-only `b_H`,
  `frozen_eval_manifest`, full deploy package;
* main/group/routing tables, per-row B/H/q/route/label/contribution, coverage/new-conflict/tail/
  sensitivity/cancellation tables;
* paired bootstrap/gate, replay/label-independence/key checks, budget/manifest;
* single-file replay, analysis re-run command; one appended entry in the existing
  RESEARCH_STATE / ledger, not rewriting history.

The REPORT opens by directly answering:

1. Is the current COMP+Q source and published result reproduced? Besides the cycle route and the
   bias produced by the fixed rule, what else changed this round?
2. How many train-consistent classes did the original `Q` miss? Is valid covered, are new
   conflicts produced? Which route do severe errors belong to?
3. What are `H`'s whole-valid `y` raw/cal, gain/CI/gate? Does it harm G0, is it only bias or a
   single-row gain?
4. Does `q → c` improvement convert to `y` improvement, how much does cancellation explain? Is
   `0.09` actually observed?
5. Does this round support function-fit miss, input conflict, uncovered, or still
   unlocalized? State only on the evidence-corresponding subset. What is the next responsibility,
   and which attempts explicitly stop?

Before finishing, confirm this round had no optimizer/backward, no new GPU/remote job, no test
read, no extra configuration. Report the real elapsed time and missing items; "the code is
written" does not substitute for the actual B/H evaluation, and the run is not extended to try
another rule just because this one did not gain.
