# PROTOCOL — css-pcba-s0-v1 (conditional structure statistics, PCBA, seed 0)

Human-readable twin of `protocols/css-pcba-s0-v1.yaml` (machine record) and
`code/css_common.py` (the frozen constants every run hashes into
`analysis/frozen_config.json`).  Where the three disagree, `css_common.py` is the
source of truth and the disagreement is a bug.

## 1. Question and the only comparison that matters

Same structural objects, same encoder widths, same parameter count, same initial
state, same batch order, same 40 epochs, same evaluation: does conditioning each
raw structural occurrence on the molecule's task context (**E**, `EARLY_TASK`)
beat applying the identical conditioning once after the root aggregation (**L**,
`LATE_TASK`)?

E and L have identical parameter names, shapes, initial values and gradient
steps (831,088 parameters each).  Only the aggregation formula differs:

```
E:  k,p -> s = <k_p, q_t>/sqrt(16);  z_f,v,t = sum_{p in P_f(v)} [1+0.5 tanh(s)] v_p / sqrt(n_f(v))
L:  kbar_f,v = mean_p k_p;   z_f,v,t = [1+0.5 tanh(<kbar_f,v, q_t>/sqrt(16))] sum_p v_p / sqrt(n_f(v))
```

E's root state depends on the *order* of the occurrences (each one is gated
separately); L's does not (the gate is one scalar per root).  That difference is
the entire subject of the round.

**S** (`EARLY_SHARED`, 829,040 parameters) is E without the 128x16 task-query
table: it keeps the molecular-background term `Wc c_G` and the public query
`q0`.  S is purchased only by the pre-registered rule in section 7, and it is a
configuration increment (128x16 = 2,048 parameters), not a capacity-matched
causal control.

## 2. Data and what is frozen

| item | value |
| --- | --- |
| dataset | `ogbg-molpcba`, official split 350,343 / 43,793 / 43,793 |
| feature cache | inherited `lcrc_pcba_scale_pilot_v1/cache/pcba_v1` (atom 51, bond 13, atom_scalar 9, global 42) |
| S2 index | inherited CPC cache `lcrc_pcba_cpc_v1/runs/cpc_cache` (middle atom + both physical bond ids) |
| normalisation | inherited Train-only buffers, never re-fitted |
| labels (Train/Valid) | inherited NaN-masked artefact |
| labels (Test) | official `data/ogb/ogbg_molpcba/raw/graph-label.csv.gz`, sha256 `ae9430cf…bb7a2`, overlaid into a separate development view |
| identities | `feature_spec_hash 7270f767…`, `norm_fingerprint 6ce50f98…`, `split_partition_sha256 c872bd63…` — re-checked at load, epoch 0 and analysis |

The frozen artefact keeps Test labels NaN; the development view overlays the
officially released Test labels and is verified to reproduce the frozen
Train/Valid block bit-for-bit.  Test labels only ever reach the *evaluation*
path: `DevTask.training_targets` refuses Test rows, and the label-isolation gate
checks that no Test label participates in the loss, the optimiser, the
normalisation or a query.

Structure objects (no sampling, no truncation, no label in any index):

* **S1** — both directions `(v,u,e)` of every physical bond, root `v`; exactly
  `2 x n_bonds` occurrences.
* **S2** — every rooted non-backtracking two-bond path `(v,w,u,e1,e2)` with the
  three atoms pairwise distinct; different-middle multiplicity preserved,
  triangles kept; closure recomputed from the batch adjacency and cross-checked
  against the frozen `path_closure`.

Both are built once per forward pass (`css_data.collate_batch`) and consumed
unchanged by the encoders.

## 3. Model and readout

Atom MLP(51→256→192), bond MLP(3·192+13→320→192), S1 event (2·192+192→96), S2
event (3·192+2·192+1→96), keys 16-d, values 32-d, root static (393→64), root
statistic (64→64), LayerNorm-64, graph MLP(235→192→96), output `Linear(96→128)`
per task with a 128-long bias.

Readout is graph statistics only — `[sum_v h/sqrt(n), mean_v h, mean_v h², c_G,
log1p n]` (235).  No message passing, no node-level attention, no additive
atom-output head, no extra prediction branch, no `[paths, tasks, 961]`
replication, no pretrained weights, no distillation, no teacher, no LCRC output.

Initialisation: `torch.manual_seed(0)` → one canonical state built once, copied
tensor-for-tensor into E, L and S.  `q0 = qt = Wc.weight = out.bias = 0`, so
`w = 1` and E = L = S at step 0 (verified to 9.5e-07 at fp32 on both CPU and
GPU).  Keys therefore receive no gradient until the query leaves zero — the
gradient-reachability gate asserts exactly that (query > 0 at step 1, keys = 0
at step 1, keys > 0 at step 2).

## 4. Training recipe (inherited, unchanged)

AdamW (0.9, 0.999, eps 1e-8), weight decay 1e-3, clip 5.0, fp32, no TF32, batch
128, 2,738 updates/epoch, 40 epochs = 109,520 updates, one parameter group with
one epoch of linear warmup 3e-5 → 3e-4 and per-step cosine 3e-4 → 1e-5
(`sg_common.scheduled_lr`, `phase=3`).  Observed-entry mean masked BCE.  A batch
with no observed label is skipped and recorded, never filled with negatives.

Batch order is the frozen seed-0 plan (`06f437299fa4…` global,
`1a24dfca5489…` consumed-40 prefix); both hashes are re-verified at load.
Dropout (p = 0.10, on `h` only) uses one `[n_atoms, 128, 64]` mask per Train
batch from a dedicated generator (seed 230,000), sliced per task chunk and
saved/restored with the checkpoint, so 16/8/4 task chunks give identical logits,
identical Train loss and identical gradients.

Data collation is overlapped with the GPU step by a single worker thread
(`PREFETCH_DEPTH = 3`) that builds the next batches from the same frozen plan;
`tests/test_css_round.py` asserts the loss sequence is bitwise identical to the
synchronous path.  This is a pipeline change, not a recipe change.

No early stopping, no EMA/SWA/soup, no extra loss, no `pos_weight`, no
hyper-parameter tuning, no horizon extension, no seed replication.

## 5. Evaluation and selection

Every completed epoch evaluates **Train, Valid and Test in full** (official OGB
`Evaluator('ogbg_molpcba')` AP, cross-checked against an independent sklearn
implementation — agreement 3e-18 on the preflight).

* Selection: maximum Validation AP over the 40 epochs; a strictly larger value
  replaces the selection, an exact tie keeps the earlier epoch; epoch 0 is never
  selectable.
* Reported side by side: the Test column of that frozen best-Validation state,
  the epoch-40 state, and the mean of the last five epochs.
* Every Test number in this round is labelled **DEVELOPMENT_TEST**
  (`test_policy: always`, authorised for this round).  Test never selects
  anything.

## 6. Correctness gates (all must pass before a formal number exists)

CPU: S1 = 2 × bonds; per-root S2 count = `sum_{w in N(v)}(deg(w)-1)`; middle and
physical bond ids aligned; closure correct; batch reordering and atom/bond
renumbering leave the prediction unchanged (≤ 1e-6); isolated atoms, molecules
without S2 and triangle closure stay finite; `q=0` ⇒ E = L = S; equal keys and a
single occurrence ⇒ E = L; the non-exchange witness (scores (+1,−1), values
(+1,−1), n = 2) gives E = `tanh(1)/sqrt(2)` and L = 0; task chunk 16/8/4
equivalence; Test labels cannot reach training; parameter counts, E/L initial
equality, shared RNG, row alignment.

GPU preflight: real heaviest Train batch (atoms + S1 + S2 measured from the
collated batch), step/memory, 1/3/10/100-step gradient reachability, recorded
query norms / gate mean and std / between-task gate spread, checkpoint
continuation equivalence (2.95e-07), official AP vs sklearn agreement, and a
full three-split evaluation with row counts and label sources.

An error that affected a result marks that result **INVALID**; it never enters
the formal table, and the engineering fix precedes the formal run.

## 7. Resources and the S purchase rule

Two idle GPUs, one training process each: E on `res-gpu0`, L on `res-gpu1`, in
parallel.  Budget: wall soft 6 h / hard 10 h, GPU soft 12 h / hard 16 h, GPU
preflight target 0.25 h.  If the two primary arms cannot finish inside the hard
budget after at most ~30 min of mathematically equivalent optimisation, the
round is **COST_BLOCKED** and delivers the implementation, the measured cost and
a recoverable state instead of a half-trained comparison.

After both primary arms finish, S is purchased iff **all** of:

1. `d40 = valid_AP_E(epoch40) − valid_AP_L(epoch40) ≥ 0.002` **or**
   `valid_AP_E(best) ≥ 0.268273`;
2. `valid_AP_E(best) ≥ 0.262273` (floor);
3. used GPU-hours + 1.2 × projected S cost ≤ 16 GPU-hours.

Otherwise S is marked `NOT_PURCHASED` with the concrete reason, and idle budget
is not filled with another model or another seed.  The thresholds buy one extra
explanatory trajectory; the continuous values are always reported next to the
decision, and 0.0019 vs 0.0021 is not a scientific difference.  This is not a
significance test.

Historical anchors (single seed, different architecture, cited as performance
position only, never as equal-parameter controls): W2 0.265273 / Test 0.253617;
W2-DYNAMIC 0.275067 / 0.262226; LCRC_CPC 0.265944; LCRC_RC_REAL 0.267718.

## 8. Failure modes that end the round honestly

`COST_BLOCKED` (budget), `INVALID` (a gate failed after a number existed),
`NO_EVIDENCE` (no measurable resolution between E and L under this budget), or a
decision to stop.  In every case the deliverable is the measured cost, the
recoverable artefacts and the explicit statement of what is not known.
