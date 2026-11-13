# PROTOCOL — LCRC Lipophilicity real-edit differential supervision

Round / protocol ID: `lcrc-lipo-edit-supervision-public122-s0-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/`
Design date: 2026-11-13 (Asia/Shanghai)
Frozen before any formal GPU training. Only the "implementation mapping" and
"P0 observed support" sections may be appended after freezing; the scientific
definitions below are not changed after the first formal run.

---

## 0. Question and boundary

Keep the LCRC-C1 representation, structure and capacity fixed. Ask only:

1. Using **real local edit relations** to differentially supervise already
   existing labels, can the model improve (a) new-molecule property prediction
   and (b) edit response involving new molecules?
2. If there is a gain, does it exceed **paired-endpoint reweighting** (W) and
   **generic random-pair differential training** (N)?

This is a learning-objective experiment. The differential labels are computed
from existing labels; no new measurement information is added. A positive
result is not evidence of unique atom contributions, chemical causality, new
structural information, cross-seed stability, or superiority over all GNNs.

Fixed scope: public ChemBench `random122`, C1 = `NO_BOND_MEAN` (228,674
parameters), seed 0, four arms, one run each. No seed expansion, no
hyperparameter search, no new chemistry features, context modules, propagation
layers, attention, explanation heads, residual correctors or external
pretraining.

---

## 1. Data and split (frozen)

ChemBench Lipophilicity, public published split `random122`:

| file | SHA256 |
|---|---|
| `Lipop.csv.gz` | `34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758` |
| `Lipop.random122.ind.pkl` | `bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d` |
| `raw_attributes_v1.json.gz` | `1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be` |

Upstream commit `0b10d4c1ac72b6929503694a5f60446824736391`. Train/Valid/Test =
3360/420/420. Published row-index bytes, int64, original order:

* Train `4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96`
* Valid `31e3f28f31da69249f70922126f19795920bc1e32cc05d5e29e27507bd5aa365`
* Test  `a4d92ad939bf268c00ff69afe77aa5d58c19a6f809b6a09f6dadc51cd648ec93`

`code/common.py::resolve_source_files` finds the bytes **by content hash**, so
a migrated directory path does not change identity. A mismatch raises
`DATA_OR_SPLIT_MISMATCH` and stops training.

All 4200 molecules participate in ordinary molecule training/evaluation. No
desalting, deduplication, stereo removal, difficult-molecule removal, or CSV
reordering. Pair eligibility affects only pair construction.

Target `exp` (raw logD). `z = (y - mu_train) / sigma_train`, with mean and
`std(ddof=0)` fitted on Train3360 only. Structure normalisation is fitted on
Train3360 structures only and shared by all four arms. The frozen `lipo_v1`
spec is used unchanged: atom_feat=51, bond_feat=13, atom_scalar=9, global=42,
S1=80, S2 path=400.

---

## 2. P0 — real pairs over all 4200 structures (label-blind)

### 2.1 Molecule eligibility

Single connected component; no explicit H atoms; canonical SMILES has no `@`,
`/` or `\`; every atom is a heavy atom. Applied to all 4200 rows.

### 2.2 Cut and fragment rules

Only non-ring single bonds. RDKit `FragmentOnBonds(bond, addDummies=True)`
replaces the severed bond with one dummy, preserving the attachment atom's
valence. The variable fragment is 1–4 heavy atoms. The common core has at least
10 heavy atoms and is at least 75 % of each parent's heavy atoms.

Core identity is decided by **explicit labelled-graph isomorphism** carrying the
attachment-point marker. WL hashes only bucket candidates. Node label =
(element, is_aromatic, formal_charge, total_H, is_attachment); edge label =
SINGLE/DOUBLE/TRIPLE/AROMATIC/OTHER. Every cut verifies
`core_atoms ∪ var_atoms = all source atoms`; every pair's stored mapping is
re-checked node-label exact, edge-label exact, and attachment-to-attachment.

### 2.3 One definition per unordered row pair

Pairs with identical complete canonical molecules or identical variable
fragments are not edits. For one unordered row-id pair, keep exactly one
definition by:

```
(max core_heavy, core_smiles, variable-fragment strings, cut indices)
```

The last item is a deterministic tie-break only after the protocol's mandated
keys. Direction A→B: smaller canonical variable-fragment string is A. The
`pair_id` is built from the **sorted row ids**:

```
pair_id = p<min(row_a,row_b):05d>_<max(row_a,row_b):05d>
```

The reverse direction is never a second pair. The historical Train table had
509 rows but only 501 unique unordered row pairs: 8 unordered pairs had two
directed definitions under the old dedup key. This round reports 501 P_train
pairs with the same 502 participating Train molecules. This is a dedup
correction, not a relaxed filter.

### 2.4 Sets

* `P_train`: real pairs with both endpoints in Train3360.
* `E_T`: real pairs with at least one Test endpoint. Mutually exclusive
  subsets: Train–Test, Valid–Test, Test–Test.
* `E_HH = Valid–Test ∪ Test–Test`: neither endpoint entered gradient training,
  but Valid selected checkpoints.
* `Train–Valid`, `Valid–Valid`: descriptive only.

`E_T` Train–Test cores necessarily occur in Train. Results measure edit
response **involving** a Test molecule, not new-core generalization. E_HH pairs
are marked seen/unseen only by whether their core occurs in **any legal Train
cut**; no other threshold or subgroup is invented.

### 2.5 Components

Training components use only Train molecules and P_train pairs and merge
connected molecules through shared molecules, the same actual matched core, and
identical complete canonical molecules. Evaluation components do the same on
all 4200 molecules and all pairs. Evaluation components are used only for
aggregation, resampling and influence checks; they never feed training sampling.

### 2.6 P0 gates

* P_train ≥ 100 pairs and ≥ 30 pair-bearing training components.
* E_T ≥ 40 pairs and ≥ 15 pair-bearing evaluation components.

Failure → `DATA_INSUFFICIENT`, no GPU training. E_HH below 40 pairs or 15
components is `LOW_SUPPORT` but does not block the round and cannot be used for
a two-endpoints-untrained structural-generalization claim.

**P0 observed (2026-11-13, before formal training):** 823 total pairs;
P_train 501 pairs / 502 molecules / 175 components; E_T 178 pairs / 199
molecules / 59 components (Train–Test 147, Valid–Test 19, Test–Test 12);
E_HH 31 pairs / 51 molecules / 24 components (seen-core 25, unseen-core 6).
All training/evaluation gates pass. E_HH and both its seen/unseen subsets are
`LOW_SUPPORT` by the fixed rule. Details: `analysis/data_audit.json`.

---

## 3. Model and initial state

One model only: C1 = `NO_BOND_MEAN`, 228,674 trainable parameters. Build one
FULL initial state with `dissect_model.build_full_state(spec, train_norm,
seed=0)`, then `remap_state("C1", full_state)`. All four arms load this **same**
state tensor-by-tensor. No trained checkpoint is reused. The parameter count is
checked against 228,674.

---

## 4. Four arms and exact losses

Let `p = f_theta(x)`, `z = (y-mu)/sigma`, `e = p-z`. Every step has one ordinary
molecule batch of 128 (last batch smaller; 27 steps/epoch) and an auxiliary
batch of 32 real pairs:

```
L_mol = mean_i e_i^2

A: L = L_mol
R: L = L_mol + lambda * mean[(e_b - e_a)^2 / 2]
W: L = L_mol + lambda * mean[(e_a^2 + e_b^2) / 2]
N: L = L_mol + lambda * mean[(e_v - e_u)^2 / 2]
```

`lambda=1.0` fixed for all arms; not searched, not calibrated on Valid/Test.
The identity

```
(e_b-e_a)^2/2 = (e_a^2+e_b^2)/2 - e_a e_b
```

means W and R have exactly the same endpoint squared-error weight; the only
difference is cross-molecule residual coupling.

Real-pair sampling: uniformly draw a training component with replacement, then
uniformly draw a real pair within that component. Exactly 32 pairs every step,
including the last ordinary batch. One total `loss.backward()` and one
`optimizer.step()` per step; the auxiliary term never gets its own optimizer
update.

### N control construction

For each step, expand the 32 real pairs into 64 endpoint-occurrence tokens
(repeated molecules remain repeated tokens). Shuffle all 64 tokens with the
pre-generated NumPy stream; pair adjacent tokens into 32 pairs. Reject the whole
shuffle if any edge is:

* the same row,
* the same complete canonical molecule,
* any P_train real edit edge (unordered row pair).

At most 256 shuffles per candidate real batch. If none succeeds, redraw that
step's 32 real pairs and retry, at most 64 redraws; otherwise
`CONTROL_CONSTRUCTION_FAILED` and no formal training. The label difference of
each N pair is recomputed as `z_v - z_u`; it never reuses the original pair's
difference. The final N endpoint multiset must equal the R endpoint multiset at
every step. R/W/N use the same final real-batch sequence.

`code/schedule.py` pre-generates all 9720 steps with one independent
`numpy.random.Generator(20261113)` and records counts of redraws / rejected
shuffles in `analysis/schedule_manifest.json`.

---

## 5. Forward path and randomness

Ordinary `L_mol`: `model.train()`, dropout 0.10 on the final atom-level `h`.
Auxiliary R/W/N: temporarily `model.eval()` **with autograd enabled**, compute
the real inference-function auxiliary loss, then restore `model.train()`. No
`no_grad`/`inference_mode`, no detach. This prevents two independent dropout
masks from being treated as a property change. Existing LayerNorm only; no new
BatchNorm, no running state.

Each step deduplicates auxiliary endpoints by `row_id`, sorts them, performs one
auxiliary forward, then gathers to tokens/pairs. R/W/N process the same molecule
set in that step. N only changes the gather pairing, so molecule order and batch
size cannot change the computation. A skips the auxiliary forward entirely.

Four arms share: initial state, ordinary batch order, dropout random stream,
optimizer/scheduler order. Auxiliary sampling is a pre-generated independent
stream. Record and check: initial-state hash, full ordinary batch-order hash,
auxiliary schedule hash, and per-step CPU/CUDA RNG-state hashes over the first
epoch. Bit-level determinism of CUDA scatter/index_add is **not** claimed.

---

## 6. Fixed recipe

Train3360; 360 full epochs; 27 steps/epoch; 9720 optimizer updates; batch 128;
`drop_last=false`. Shuffle generator seed 917000. Training random seed 0; after
building/loading the initial state, reset CPU and CUDA RNG to 0. AdamW
betas=(0.9,0.999), eps=1e-8, weight_decay=1e-3; per-step cosine 3e-4 → 1e-5,
T_max=9720; gradient clip 5.0; FP32, no AMP. No early stopping, soup, EMA, SWA,
pretraining or other task.

Every epoch, evaluate full Valid420 in eval/no_grad on raw logD RMSE; save on
strict improvement, tie keeps the earlier epoch. Each arm selects its own best
checkpoint by the same rule; the metric is never changed to pair MAE. Save the
best and last epoch states. The last epoch is used only for one endpoint
evaluation, never for re-selection. Reloading the best checkpoint must reproduce
Valid RMSE within 1e-5 of the curve minimum; save row-id-aligned Train/Valid/Test
predictions.

Diagnostics at optimizer steps 1, 28, 271, 2701: `L_mol`, `L_aux`, gradient L2
norms of both terms over all trainable parameters, norm ratio, gradient cosine,
and clipping status. `autograd.grad` is used separately and the original update
is unchanged.

---

## 7. Correctness gates

CPU/small-scale gates (`code/gates.py`, `tests/test_edit_supervision.py`):

1. source bytes, split row ids, hashes, Train-only normalisation, feature-spec
   hash, 228,674 parameters, shared initial state.
2. hand-tensor A/R/W/N losses, the R/W identity, analytic gradients, and the N
   random-pair label difference.
3. reverse-direction strict sign change, unordered dedup, core isomorphism and
   attachment points on all pairs.
4. all-step N/R endpoint-multiset equality, no forbidden N edge, shared ordinary
   batch order.
5. auxiliary eval forward has nonzero gradients and can update weights; restores
   train mode; no RNG or normalisation-buffer side effect.
6. single vs batch legal-molecule inference; row-id joins, never positional
   concatenation.
7. checkpoint reload and independent NumPy metric recomputation; component
   bootstrap weighting.

GPU smoke: four arms × 2 epochs in one job, separate outputs, not used for
selection. Formal runs restart from the same initial state. Only implementation
or infrastructure bugs are fixed; scientific configuration is not changed by
smoke or formal trends.

---

## 8. Evaluation

Primary performance: Test420 molecule RMSE in raw logD; also Train/Valid/Test
RMSE and MAE. Primary edit metric: E_T component-equal pair-difference MAE:

```
for each pair: | (pred_b - pred_a) - (y_b - y_a) |
within component: mean over that component's pairs in the set
across components: equal weight per component with a pair in the set
```

Also report ordinary pair-equal MAE/RMSE and the zero-change reference
`mean |y_b-y_a|`, with consistent pair-equal weights. Report E_T subsets,
E_HH, E_HH seen/unseen core, and descriptive Train–Valid / Valid–Valid. Test
can be described by whether the molecule has a Train real-edit neighbour; this
is descriptive and is not attributed to edit training.

Gain = `metric(reference) - metric(candidate)`; positive means candidate better.
Primary comparison R vs A; mechanism comparisons R vs W and R vs N.

Uncertainty: 1000 paired component bootstrap replicates,
`Generator(20261114)`, 95 % percentile interval. Test RMSE resamples structural
components containing Test molecules, keeps all Test molecules in sampled
components, and recomputes overall molecule-weighted RMSE. E_T/E_HH resample
components with a pair in that set and use the component-equal definition. All
four arms and every comparison use the same resampling for a given metric.
Intervals condition on the four trained models; they are not training-seed
uncertainty and a CI containing zero is not called equivalence.

Fixed influence checks: remove A's largest E_T component (by A component mean
absolute difference error, tie by component id) and recompute the R vs A/W/N edit
gains; remove A's largest Test squared-error molecule (tie by row_id) plus its
associated evaluation pairs and recompute the main gains. Tables retain all
data. Last-epoch Test RMSE and E_T CE-MAE are reported beside best-epoch values
but never re-select a checkpoint.

At most three cases, fixed by R vs A on E_T: largest improvement, largest
worsening, improvement closest to the median (ties by pair_id), with both
SMILES, true/predicted differences, component and split.

---

## 9. Pre-registered decision labels

### Common benefit (R vs A)

1. Test RMSE and E_T CE-MAE gains both > 0, both 95 % interval lower bounds
   > 0, Valid RMSE gain > 0, and both fixed influence checks do not reverse
   sign → `DUAL_SIGNAL_THIS_RUN`.
2. Both Test gains > 0 but any interval crosses zero, Valid reverses, or an
   influence check reverses → `DUAL_TREND_UNCERTAIN_THIS_RUN`, naming the
   uncertain item.
3. Edit improves but molecule RMSE does not → `EDIT_PERFORMANCE_TRADEOFF`.
4. Molecule RMSE improves but edit does not → `PERFORMANCE_ONLY_THIS_RUN`.
5. Neither improves → `NO_JOINT_GAIN_THIS_CONFIG`.
6. Data/execution/budget gate fails → `DATA_INSUFFICIENT` /
   `EXECUTION_INCOMPLETE` / `BUDGET_LIMITED`; no effectiveness conclusion.

### Real-relation increment (R vs W/N)

Only if R vs W and R vs N E_T edit gains are both > 0, both interval lower
bounds > 0, R's Test RMSE point estimate is no worse than W and N, and the
influence checks do not reverse → `RELATION_INCREMENT_THIS_RUN`. Otherwise
`RELATION_INCREMENT_UNRESOLVED` or `CONTROL_BETTER_THIS_RUN`, with the control
data listed. A non-significant difference is never written as "same/equivalent".
If R/W/N are close, the honest statement is that endpoint reweighting or generic
differential training cannot yet be excluded.

E_HH can add a limited observation only if support is sufficient and its
direction agrees with E_T. Low support does not block the round and does not
upgrade a structural-generalization claim.

Every conclusion is limited to public122, C1, seed 0, this target strength and
this pair rule. No cross-seed stability, all-split validity, chemical causality,
unique atom contribution, universal GNN superiority or whole-LCRC-route claim.
A negative result says "this configuration did not obtain joint benefit", not
that message passing is necessary or that edit supervision is invalid.

---

## 10. Execution and budget

Host `res` (two A100 GPUs), pools `res-gpu0` / `res-gpu1`, one round training
process per GPU; no DDP, no `res-2`.

* Wave 1: gpu0 = A_s0, gpu1 = R_s0.
* Wave 2: gpu0 = N_s0, gpu1 = W_s0.

All four arms complete; wave-1 results cannot cancel a control. Build the shared
structure cache and frozen schedule once before concurrent training. Budget:
target ≤ 2 A100 GPU·h, hard cap 4 A100 GPU·h over all smoke, inference, failures
and retries (sum of wall-time over GPU processes). If the budget is insufficient,
report `BUDGET_LIMITED`; do not shrink epochs/batch or drop controls to force a
main table. One same-configuration infrastructure retry is allowed. A scientific
code bug requires rerunning every affected arm and charging it; if the budget is
then insufficient, report incomplete and never mix pre/post-bug results.

Local Git is the source of truth; remote is compute only. Formal runs bind to one
committed revision; no `--allow-dirty`/`--allow-stale` for reported results.
All remote operations use `rr`.

---

## 11. Implementation mapping

| protocol item | code |
|---|---|
| frozen data/hash/split | `code/common.py::verify_source`, `code/data.py` |
| structure cache | `code/data.py::build_all_caches`, `load_cache` |
| pair eligibility/cuts | `code/pairs.py`, upstream `edit_pairs.py` |
| core classes / isomorphism | upstream `edit_graph.py`, `edit_pairs.py`; `code/pairs.py::verify_mapping` |
| unordered dedup / pair_id / direction | `code/pairs.py::assemble_pairs` |
| sets/components | `code/pairs.py::annotate_sets`, upstream `edit_split.py::build_units` |
| schedule / N control | `code/schedule.py` |
| losses | `code/losses.py` |
| model/init | upstream `dissect_model.py`; `code/train.py` |
| training | `code/train.py` |
| metrics/bootstrap | `code/metrics.py` |
| evaluation/decisions | `code/evaluate.py` |
| gates | `code/gates.py`, `tests/test_edit_supervision.py` |
| entry points | `code/prepare.py`, `code/train.py`, `code/evaluate.py`, `code/run.py` |

No upstream file is modified. Historical caches, reports and predictions are
not overwritten.
