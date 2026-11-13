# PROTOCOL — LCRC/Lipo oracle task relevance: Part A clarification + two computational tasks (public122, seed 0)

Round / protocol ID: `lcrc-lipo-oracle-task-relevance-public122-s0-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/`
Source round: `lcrc_lipo_multi_motif_mechanism_public122_v1` (protocol revision `0273cad6`, report `a5febe96`)

Written before the first Part-B training. The scientific definitions below (task
pair, oracle functions, split/target-normalisation discipline, seed-0 isolation,
weights, CF scope, bootstrap, multiplicity, decision boundaries) are **frozen**.
Only observed support counts and implementation-mapping notes may be appended.

---

## 0. Responsibility, questions and the exact amount of training

This round implements, verifies, executes and reports only; it does **not**
re-investigate, re-select a topic / motif, or design a new parameter matrix. The
previous round supported a training-related structural preference *within the
dose-matchable part* of the population; it did not prove every functional group
is preferentially responded to, nor did it give the rewired molecules
experimental logD labels.

**Part A** clarifies the existing results (coverage/claim matrix; the
same-population three-table contrast that separates a *population* change from a
*control* change; the conditional scalar/S2 recovery effect). It trains nothing
and runs no new inference.

**Part B** adds two *computational* supervised tasks on the same 4200 original
molecules and the same frozen rewiring catalogue, to test whether the model's
rewire response follows a known task change whenever a computational reference
value exists:

| task | oracle | property |
|---|---|---|
| `WC_LOGP` | `rdMolDescriptors.CalcCrippenDescriptors(mol, includeHs=True, force=True)[0]` | Wildman-Crippen logP; a computational function reference, **not** experimental logD and **not** a newly measured logP |
| `MOLWT` | `Descriptors.MolWt(mol)` | average molecular weight; composition-preserving rewiring keeps it invariant, so it is the task-invariant control |

Exactly **two** formal trainings: `WC_LOGP_s0` (gpu0) and `MOLWT_s0` (gpu1),
both seed 0, same LCRC-C1, same fixed recipe. One card per task, may run in
parallel in one round on host `res`.

These two trainings buy **task-relevance evidence**, not multi-seed stability;
this round must not be described as zero-training. The original experimental
logD model, original labels, the official split, the six families / graphs /
controls / dose rules and all old numbers are unchanged. No new motif, no new
rewiring, no new dataset, no SSL / probe / auxiliary loss / architecture search /
PNA / PCQM.

## 1. Frozen source and reuse

* Data: ChemBench Lipophilicity, published split `random122`, Train/Valid/Test =
  3360/420/420, original `row_id`s and order kept. Upstream revision
  `0b10d4c1ac72b6929503694a5f60446824736391`.
  `Lipop.csv.gz` `34280f0a…`, `Lipop.random122.ind.pkl` `bf7acb17…`,
  `raw_attributes_v1.json.gz` `1127607d…`; Train/Valid/Test row bytes
  `4e8d42a9…` / `31e3f28f…` / `a4d92ad9…`.
* Feature spec: `lipo_v1` AF51 / BF13 / scalar9 / global42 and the LCRC-C1
  variant (228 674 parameters, `NO_BOND_MEAN`) are reused unchanged. The
  feature normalisation is the checkpoint's **frozen Train-only** buffers,
  never refitted. Each new task uses **only its own Train original labels** to
  fit the target mean/scale (never rewired / Valid / Test labels).
* The frozen ordinary-MSE `A_s0` checkpoint (SHA256 `bdd670e8…`, state
  `7d32221e…`, best epoch 244) and the verified training-start init (state
  `aac1e061…`, full `666b88d7…`) are referenced, not retrained.
* Part B uses only the already-frozen, lawful, range-supported original / rewired
  views; no view is re-selected by oracle effect and no C is re-matched.

If the frozen upstream catalogue / predictions cannot be recovered, the round
reports `MISSING_SOURCE` and does not purchase new training; numbers are never
filled in from a summary. All source SHA256 / state / spec / norm are verified
and cited from the previous manifest (not only by file name).

## 2. Part A — three zero-GPU clarifications

### A1 coverage & claim matrix
Per family / split: pre-calibration and post-calibration eligible
molecules / scaffold clusters / instances / T pairs, C link count, C unique view
count, calibration retention ratio and the largest scaffold-cluster share. The
539 C links are **not** 539 independent molecules; all effects use the original
weighting and the original cluster bootstrap. Report the original `rC`, `INIT`
`rC`, `E`, 95 % descriptive interval and the original 12-item Bonferroni
simultaneous interval; separate "positive point estimate" from "simultaneous
interval supports it".

`E = rC_TRAINED - rC_INIT > 0` only shows a preference increase **relative to
this one initialisation**; whether `rC_TRAINED > 0` is a separate proposition.
A sulfonamide with `E > 0` alone is not written as a clear positive preference.
An ester / aryl-ether reversal means a *weaker response relative to this
control*, not "the model does not use the group". An empty sulfonate cell is
"untested", never "invalid / equivalent / zero effect". `rC` / `E` are
dimensionless and are never mixed with the old raw-logD `J`.

### A2 same-population three tables
For each family, from the existing predictions and eight-cell results:
* `ALL_K0` — every original eligible T / K0 pair;
* `RETAINED_K0` — only the focal T that have a C, still compared to the original K0;
* `RETAINED_C` — the identical focal T, compared to the matched C.

The molecule / instance / control / donor hierarchy is kept (never view-equal in
the middle). Report `J` / `D` / `B` and the seven mechanism terms. `ALL→RETAINED`
is the selected-population / edit change; `K0→C` is the same-population control
change. The two are never merged into "dose calibration removed S1" or "the
learned effect is necessarily produced by S2".

### A3 conditional scalar and S2 recovery
A small standalone `gamma3` cannot prove the whole scalar block is inert. From
the existing eight cells, for the raw prediction F and the J / D endpoints:
`R3 = value(123)-value(12) = gamma3+gamma13+gamma23+gamma123`,
`R2 = value(123)-value(13) = gamma2+gamma12+gamma23+gamma123`.
Report paired means and intervals, plus the per-view absolute recovery (to avoid
mean cancellation), and state whether each belongs to `ALL_K0`, `RETAINED_K0` or
`RETAINED_C`. The original frozen bootstrap draws / weights are used; no
significance rule is re-chosen. Arithmetic residual <= 1e-5.

If `R3` is not small, write "the standalone scalar term is small but there are
interactions", not "inert"; if only the TRAINED J/D path is established, do not
claim the dimensionless `E` learning path is located. These are clarifications:
they do not change the old protocol or its stored numbers. Part A is run and kept
whatever the sign; it never gates whether Part B runs.

## 3. Part B — two computational targets: build the reference first, then train

For all 4200 original molecules and the frozen Valid/Test O/T/K0/C graphs, the
RDKit molecule is reconstructed from the existing SMILES→raw mapping and
verified. The two functions are exactly the frozen calls above (RDKit version
fixed and recorded; no dependency upgrade). A copied molecule may carry a
descriptor cache: it is cleared (`ClearComputedProps`), re-sanitised and
re-computed with `force=True`; a fixed first 20 non-O views are independently
re-parsed from their new canonical SMILES to prove the label is not the original
molecule's cached value. The re-parse is an oracle check only — the model input
continues to read the existing raw-rebuild cache, and no graph is changed. All
original components are kept (no salt stripping, no largest-fragment, no
protonation change); the four-component Train molecule stays and does not enter
the rewiring queue.

Atom mapping, element / isotope / formal charge / H count must agree with the
real new graph; every rewire's MolWt must differ from O by <= 1e-8 (a failure
triggers a mapping / hydrogen / cache investigation, never a graph deletion).

Checks: both targets finite on all 4200 originals and Train std > 0; WC_LOGP does
change for the frozen rewires with a reported fraction and distribution. If
WC_LOGP is numerically unchanged for every rewire, the positive task cannot
answer the question: report that fact and **stop Part B training** (no automatic
third target). The existing global / local input must not directly contain the
WC_LOGP value / Crippen output; a direct label leak stops and is reported, never
silently removed by changing the model. MolWt being computable from composition
is the design of this negative control, not evidence of complex-structure
learning. The original experimental label file is unchanged; new columns are
stored separately in `wc_logp_oracle` / `molwt_oracle` with a `row`/`view`
primary key and SHA256.

## 4. Strict isolation of original-graph training from rewired-test

Train on the Train3360 original molecules' new labels only; select the checkpoint
on the Valid420 original molecules' new labels only. No rewire label ever enters
the optimiser, epoch selection or hyper-parameter selection. The label loader
selects the task and `row_id` explicitly and asserts `source_kind = original`;
a numeric fixture proves it cannot misread the original logD label or the other
task's label. The official split is kept; nothing is re-split by new label.

The feature norm is not refitted; the target norm is computed per task from that
task's Train original labels. The canonical isomeric SMILES marks whether an
existing rewire is exactly identical to a Train original. The **main CF**
evaluation uses the paired records whose O, T and K0 are all absent from the
Train original set; the full frozen queue is the supplement. This is a
label-blind mark of new-task CF generalisation; it does not change the old
population / conclusions and is not a re-split. The C calibration set is not
re-matched; a C containing a Train duplicate is a seen-structure supplement only.
The original 420 Test molecules are always reported fully: if an original Test
molecule exactly duplicates a Train molecule it is listed separately and
stratified as novel-structure; the public split is never modified.

Before training, the PROTOCOL, the two task configs, the oracle label hashes, the
training / CF isolation lists and all primary / secondary metrics are committed.
The target is never swapped and views are never re-selected after seeing results.

## 5. Two fixed trainings, ordinary-MSE recipe

Both models are the original C1 / `NO_BOND_MEAN`, 228 674 parameters, starting
from the same original seed-0 initialisation, **not** fine-tuned from the trained
logD model. The original `build_full_state` then remap-C1 flow is used; the two
tasks' trainable initial hash and batch order are identical; the feature norm is
identical; the task target mean/scale are expected to differ and are recorded.

Fixed 360 epochs, batch 128, AdamW peak_lr `3e-4` → min_lr `1e-5` per-step
cosine, wd `1e-3`, dropout `0.10`, grad_clip `5`; MSE; best epoch chosen only by
each task's own Valid RMSE. No early-stop, EMA, SWA, soup, SSL, pair loss,
auxiliary loss, search, extra seed or re-selection. A forward-only smoke check is
allowed; no separate training smoke to increase the (undisclosed) training count
— the first formal batch already asserts loss / gradient finiteness. Only host
`res` is used: gpu0 = `WC_LOGP_s0`, gpu1 = `MOLWT_s0`, at most one round training
process per card; other jobs are untouched and `res-2` is unused. All remote
execution follows `rr doctor/deploy/run/jobs/status/logs/pull` with a committed
revision, read-only source, no manual `ssh` / `nohup`.

Budget: ~0.2 A100 GPU·h planned (soft 0.5, hard 1.0); CPU prep / analysis soft 30
min. The budget is an upper bound, not a target. Implementation errors may be
fixed and the affected part re-run; a scientific result being adverse is **not**
a reason to re-tune or add training. Actual failures / retries / training count
are reported honestly.

## 6. Primary evaluation — the new graph's own computational reference

Original graph: both tasks' Train/Valid/Test RMSE / MAE / R², the Train mean
constant baseline, with the unit and each task's Train target std stated. A poor
original-graph model cannot have its CF behaviour blamed on the explanation
method. Everything is de-standardised exactly once; WC_LOGP units and MolWt mass
units are never combined. The CF main scope is `FULL` (the whole new molecule is
re-encoded); `G_FIXED` is an internal diagnostic only and never stands in for a
full new molecule.

For view `V`: `delta_pred = f(V) - f(O)`, `delta_ref = oracle(V) - oracle(O)`.

* **WC_LOGP** main quantity: each family's mean `|delta_pred - delta_ref|`,
  compared with the fixed zero-change baseline `mean |delta_ref|`, reporting
  `gain = baseline - model` and its interval. Also report the new graph's
  MAE/RMSE against its own `oracle(V)`, a delta scatter and the sign-agreement
  rate; the sign is computed only when `|delta_ref| > 1e-8` (0-change cases are
  listed separately). The original `y` is never used to score the new graphs.
* For the same T/K0: `q_pred = |delta_pred_T| - |delta_pred_K0|`,
  `q_ref = |delta_ref_T| - |delta_ref_K0|`; report `|q_pred - q_ref|` and the
  sign agreement on non-zero `q_ref`. The core question is whether the model
  follows the correct relative change; `q_ref` may be negative and no family is
  required to have T > K0.
* **MOLWT**: every `delta_ref` is 0. Report the mean / median / 95th percentile /
  max of `|delta_pred|`, its value divided by the Train target std, and its scale
  relative to the original-graph Test error. Do **not** claim invariance from a
  signed mean ≈ 0, do not treat a CI crossing 0 as equivalence, and do not
  amplify a tiny movement by a ratio. The zero-change baseline is already
  optimal (0) for MolWt, so the model's departure from it is reported
  continuously.

The C-matched subset is a fixed secondary scope using the original matching; no
better C is searched by oracle. Weights keep the family→molecule→instance→
k→donor hierarchy and the original primary keys; unique views are deduplicated
for computation but the statistical weights are unchanged. Test is primary,
Valid is a direction/scale check. Every split shares one 5000-replicate original
scaffold whole-group bootstrap, seed 20261114, with the same resampling for both
tasks / baselines / families. The WC_LOGP six-family delta-MAE gain is given a
95 % descriptive interval and a 6-item Bonferroni simultaneous interval;
no-sample is `NA` and the multiplier is not reduced to rescue a result.
Overlapping families are not treated as independent datasets and "several
families positive" is not universalised. The main deliverable is the family
effect and reference-error matrix, and a correct negative preference is allowed.
The experimental logD is cited only via the Part-A calibrated claim and is never
mixed with the two computational tasks into one truth or compared by raw error.

## 7. Interpretation and decision boundaries

If the WC_LOGP delta prediction beats the no-change baseline and MOLWT shows a
small response on composition-preserving rewires, this can support "the same
graph edit's response changes with the supervised task and can be checked for
direction/magnitude on a task that has a computational reference". "Small" must
have a real scale and interval, no unsupported invariant-pass is declared, and
the two computational tasks are never promoted to a universal chemical mechanism
for experimental logD. If both models move a lot on the same rewires but the
delta reference does not support it, that is task-irrelevant / wrong structural
dependence — kept as a model diagnostic, with no added tuning to turn it into a
positive. If the WC_LOGP original-graph performance is already poor, basic
prediction insufficiency is separated from CF-explanation insufficiency and no
positive is bought by re-training. If Part A's S1→S2 change is mainly caused by
the selected population, the mechanism claim is limited to that population; if
it is mainly caused by the control change, it is written as a relative-control
path, not the model's only backbone. The existing amide / sulfonamide `r` / `E`
support is model-dependent evidence; this round's known-function tasks add the
task-relevance test. Experimental-logD rewire labels remain unknown and real MM
pairs remain low-support.

## 8. Deliverables and stop

New directory keeps PROTOCOL, config, code, the necessary tests and REPORT;
analysis contains at least the Part-A coverage / three tables / recovery effect /
claim matrix, the oracle values and cache checks, the training / CF isolation,
both tasks' full predictions and metrics, intervals and runtime. Key tests: label
override does not cross tasks / uses no CF for training, target standardisation
units, independent oracle recomputation, the MolWt invariant, the raw-graph↔RDKit
mapping, and the weight / paired / recovery arithmetic. Three main figures:
Part A same-population mechanism contrast; WC_LOGP delta_pred vs delta_ref; the
two tasks' scale-normalised response on the identical CF set with the MolWt
invariant error. The REPORT first answers: whose strongest previous claim is
covered; whether the full interaction changes the path judgement; whether the
model follows a structural change that has a reference value; whether a target
that should not change is stable; what evidence experimental logD still lacks.
The old PROTOCOL and old formal numbers are not changed; textual clarifications
carry a revision note. Lightweight artifacts are committed per repository
convention; no PR and no unrelated work. After this round: **stop** — no third
property, no seventh structure, no extra seed, no PNA, no large dataset. The
final line states the actual new training count, GPU·h, failures / retries and
remaining uncertainty.
