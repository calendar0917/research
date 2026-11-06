# PROTOCOL — LCRC PCBA explicit-performance sprint v1

Round: `lcrc-pcba-explicit-performance-sprint-v1`
Dataset: `ogbg-molpcba`, official OGB scaffold Train/Validation split, official
Validation macro AP. Seed `0` only. This file is committed **before** the first
formal 40-epoch run; no arm, feature, hyper-parameter, epoch or decision rule
may change after Validation AP is seen.

## 1. Question and boundary

Starting from the frozen no-iteration LCRC-A model (`C1` local S1/S2 statistics
+ 4-head graph attention pooling readout), test three candidate sources of
Validation-AP gain in one screen:

1. per-assay (task-conditioned) structural attention readout (A1),
2. deterministic materialised physicochemical descriptors (A2),
3. explicit rooted two-hop branch tokens, preserving the branch grouping that
   S2 path summation can lose (A3).

No model may have learnable inter-atom message passing, no hidden-state update
along `bond_index`, no inter-node QKV self-attention. The frozen C1 one-shot
S1/S2 statistics, static structural token extraction and graph-level attention
pooling are allowed. Parameter count is unrestricted and is reported only as a
cost.

The only adaptive iteration allowed in this round is the single two-module
COMBO defined in §7. No extra seeds, epochs, width search, loss changes,
continued training, or V runs.

## 2. Frozen assets (read-only inheritance)

| Asset | Source |
|---|---|
| Feature cache `pcba_v1` | `lcrc_pcba_scale_pilot_v1/cache/pcba_v1` |
| Official split rows | `lcrc_pcba_scale_pilot_v1/analysis/pcba_split.json` |
| Labels, Test rows NaN | `lcrc_pcba_scale_pilot_v1/analysis/pcba_labels_trainvalid.npy` |
| Feature spec `lipo_v1` | `lcrc_pcba_scale_pilot_v1/analysis/pcba_spec.json` |
| Train-only normalization | refit from official Train, fingerprint `6ce50f98…` |
| Data loader / 40-epoch order | `lcrc_pcba_lr_probe_v1/code/lr_data.py` |
| Pinned transfer | `lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.PinnedStager` |
| Frozen C1 builder / A readout | `lcrc_pcba_scale_pilot_v1/code/pc_model.py`, `lcrc_pcba_no_mp_sprint_v1/code/nmp_model.py` (arm A) |
| Metric | `lcrc_pcba_scale_pilot_v1/code/pc_metrics.py` (OGB official Evaluator) |
| Deterministic 40-epoch recipe | `lcrc_pcba_structural_gain_sprint_v1/code/sg_common.py` |

Comparability gates, checked by every run before the first step:

1. A0's C1+A initialisation fingerprint equals the historical seed-0 value
   `1b1f1ad35ab33f4107bc600166fce838d5e16ff5def2348743c73c0211dcf7c3`.
2. The 40-epoch batch-order plan equals `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`
   with consumed-40 prefix `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58`.
3. Feature spec, Train-only normalization, split partition, label artifact and
   cache manifest hashes equal the frozen values recorded in `sg_common.SOURCE`.
4. A0, A1, A2, A3 share bitwise-identical C1 **and original A readout**
   parameters at initialisation; added modules use private generators.
5. Official Test rows are NaN in the label artifact; no run indexes, scores or
   infers on a Test row. Test is never reported.

Historical same-round control to beat: A0 seed 0 epoch-40 Valid AP
`0.2463560956` (the formal no-MP Phase-C A run), parameters `287,644`.

## 3. Four fixed models

Common base: original `NMPModel(arm="A")` = frozen C1 (245,184 parameters)
+ original 4-head graph attention pooling readout (42,460 parameters).
`logits = C1_f_rho(h) + original_A_attention_pool(h)`.
Every added module is zero-residual at initialisation, so step 0 logits equal A0.

### A0 — original A (strict same-round control)

Exactly the historical A architecture, no new information, no new parameter.
Rebuilt and retrained in this round; the historical score is a reference only.

### A1 — task-conditioned attention residual

For static atom states `h ∈ R^(N_atoms×64)`, no connectivity, no bond data, no
inter-atom hidden-state update:

```
z_v      = LayerNorm(64)(h_v)
key_v    = Linear(64,64)(z_v)
value_v  = Linear(64,64)(z_v)
q        ∈ R^(128,2,64)                     (learnable, standard normal init)
s[v,t,k] = key_v · q[t,k] / sqrt(64)
alpha    = softmax over v of the same molecule only, per (t,k)
z[g,t,k] = Σ_{v∈g} alpha[v,t,k] · value_v
residual[g,t] = Σ_k w[t,k] · z[g,t,k] + b[t]
logits   = logits_A0 + residual
```

`w` and `b` are zero-initialised; `key_proj`/`value_proj` use the private
uniform initialiser (bound `1/sqrt(64)`), LayerNorm is ones/zeros. The softmax
is computed on `[N_atoms,128,2]` with per-molecule segment max/subtraction; no
cross-molecule normalisation. New parameters: **41,344** (total 328,988).

### A2 — physicochemical descriptor residual

From each official raw SMILES, ten deterministic descriptors in this exact
order (RDKit 2026.09.1):

1. TPSA, 2. NumHDonors, 3. NumHAcceptors, 4. log1p(MolWt), 5. Crippen MolLogP,
6. NumRotatableBonds, 7. total formal charge, 8. NumAromaticRings,
9. maximum ring atom count (0 if no ring), 10. FractionCSP3.

Per-column mean/std are fitted on official **Train** rows only, with `scale=1`
for a zero-std column. A molecule whose descriptors cannot be computed keeps its
row (never dropped), is filled with the Train mean for the ten columns and gets
a missing flag `1.0`; the failure is recorded with its row and exception. The
input is the standardised 10 columns plus the flag (dim 11). Validation uses the
Train statistics; Test never enters the cache.

```
residual[g,:] = Linear(64,128)( Dropout(0.10)( SiLU( Linear(11,64)(x_g) ) ) )
logits        = logits_A0 + residual
```

The last Linear is zero-initialised. The original 42-d global descriptor and
`rho` are untouched. New parameters: **9,088** (total 296,732).

### A3 — explicit rooted branch tokens

For each directed bond `(v,u)` of the molecular graph (the frozen cache stores each
undirected bond exactly once; both orientations are expanded inside the cache
builder), a deterministic signature

```
( atom_type(v), bond_type(v,u), atom_type(u),
  sorted( (bond_type(u,w), atom_type(w), is_edge(v,w)) for w ∈ N(u), w ≠ v ) )
```

is packed as little-endian uint16 (header 3 values, branch list sorted
lexicographically; `is_edge(v,w)` is the existence of bond `(v,w)`), hashed with
stable `blake2b(digest_size=8)` and mapped to `uint64 % 8192`. Python `hash()` is
not used. Atom types are the existing 0–15 classes, bond types the existing 1–4
classes. Signatures are built **offline** from the molecular graph, never from
labels; the learned model never sees the graph.

Per root atom `v`, all out-edge token embeddings (`Embedding(8192,32)`) are
aggregated as sum and mean (zero vector for degree 0), concatenated to 64,
mapped by `Linear(64,64)` (zero weights/bias) to `delta_h`, and added to `h`
before the frozen C1 `f/rho` and the original A attention readout. Embedding is
N(0,1) initialised from the module's private generator.

New parameters: **266,304** (total 553,948).

## 4. Unified training recipe

One entry point (`code/ep_train.py`) and one recipe for every arm:

* Seed 0, from scratch, never from a trained checkpoint.
* Full official Train (`350,343` rows), batch size `128`.
* AdamW, lr peak `3e-4`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `1e-3`.
* Epoch 1 linear warmup `3e-5 → 3e-4`, then per-step cosine to `1e-5` over the
  frozen 40-epoch horizon (2,738 steps/epoch, 109,520 updates).
* Dropout `0.10`, gradient clipping `5.0`, FP32.
* Observed-entry masked BCE; NaN labels are excluded strictly.
* 40 epochs, no early stopping, fixed evaluation at epochs 10/20/30/40, official
  Validation macro AP from the OGB Evaluator; **epoch 40 is the only primary
  comparison value**.
* The frozen seed-0 40-epoch batch order for every arm.
* A2's extra `Dropout(0.10)` consumes the shared RNG after the C1 dropout, so
  later masks differ across arms; this is part of the frozen A2 definition.

The unified `augmented_collate` / prediction path is owned by this round, so
Train and Validation use the same feature keys and no feature can be silently
dropped by an inherited evaluator. A0's batch contains exactly the frozen keys.

## 5. Pre-training gates (must all pass before any formal run)

CPU tests and one GPU preflight:

1. A0 forward equals the historical A implementation in the same state
   (CUDA reduction tolerance 1e-4).
2. A1/A2/A3/A0 have identical logits at zero-residual initialisation (1e-4).
3. C1 and original A readout parameters bitwise identical across arms; new
   modules initialised by private generators without touching the global RNG.
4. Batch-order SHA equals the historical seed-0 plan for every arm.
5. A1 softmax sums to 1 per molecule/task/head, permutation invariance, no
   cross-molecule normalisation or mixing.
6. A2 coverage/finiteness, Train-only mean/std, missing handling and Train/Valid
   row alignment.
7. A3 signature isomorphism invariance, reverse-edge and branch-index
   correctness, per-intermediate grouping, bucket range, test rows empty.
8. Added modules read only their declared inputs (no `bond_index`, `bond_feat`,
   `nbr_*`, `path_*`); A3 adjacency is offline preprocessing only.
9. Effective gradients in every added parameter after three real updates.
10. Masked BCE and official AP unchanged; Train/Valid disjoint.
11. Test labels unreachable and every prediction path refuses Test rows.
12. GPU step-time/memory smoke, parameter count and projected 40-epoch cost.

Any failure stops the round; only demonstrable implementation bugs may be fixed,
and the frozen model definitions may not be changed to improve results.

## 6. Execution, budget and GPU policy

All remote execution through `rr` (`doctor/deploy/run/jobs/status/logs/cancel/pull`)
on host `res` (`a100-2`). Local code is the source of truth; formal runs start
only from a committed revision with a valid deployment marker.

* Two label-free feature caches are built on the remote CPU before preflight:
  A2 descriptors (RDKit installed ephemerally for the build only; not part of
  the training environment) and A3 branch tokens. Both are Git-ignored; their
  hashes and audits are committed.
* GPU allocation is decided from live state (`rr doctor`, `rr jobs`, GPU
  utilisation/memory), not from assumptions. At the time of writing GPU0 is
  externally busy (≈16 GB, 100 % utilisation), so the round uses **GPU1 only**
  and runs sequentially. GPU0 is never preempted or shared if that would disturb
  the external job; if GPU1 is unavailable, the round waits rather than forcing.
* Budget: target ≈8 A100 GPU-hours, hard limit 10, including preflight and
  failed attempts. Before each stage the remaining budget is checked against the
  measured step time. If a stage cannot finish inside the budget, the **COMBO is
  cancelled**, never an existing arm truncated. An arm that cannot complete is
  marked `INCOMPLETE_RESOURCE`, not a negative result.
* Every formal run saves the epoch-40 checkpoint (model, AdamW moments, global
  step, RNG, batch plan, feature hashes), curve, per-task AP and run record.

## 7. The only adaptive iteration: COMBO

After all four 40-epoch runs are complete:

```
delta_i = Valid_AP40(Ai) - Valid_AP40(A0)   for i ∈ {A1, A2, A3}
```

1. Eligible: `delta_i >= 0.003`; ties broken `A1 > A3 > A2`.
2. Fewer than two eligible → stop, no COMBO.
3. At least two eligible but `max(delta_i) < 0.005` → stop, no COMBO.
4. Otherwise the single COMBO is the two largest deltas (one of `A1+A2`,
   `A1+A3`, `A2+A3`), trained from scratch, seed 0, full 40 epochs, no loaded
   single-model weights.
5. Fixed composition order:

```
h = C1 local encode
if A3: h = h + explicit_branch_residual
logits = C1_f_rho(h) + original_A_attention_pool(h)
if A1: logits += task_conditioned_pool(h)
if A2: logits += physchem_residual(descriptors)
```

The thresholds `0.003/0.005` are resource decisions fixed in advance, not
statistical significance.

## 8. Analysis and labels

Per arm: AP at epochs 10/20/30/40; epoch-40 Train AP, Valid AP, Train/Valid
BCE, 128 per-task AP with observed/positive counts, per-task deltas vs A0,
improved/degraded task counts, Train−Valid AP gap, parameters, peak GPU memory,
GPU-hours, step time, model commit, data fingerprint, feature-cache hash,
batch-order hash, checkpoint hash and the no-Test-access audit.

Per-task comparison strata: the 127 scorable tasks overall, Validation positive
count `>=20`, `5–19`, `<5` (a task without both classes is not scorable).

Labels (per arm, main metric only):

* `SCREEN_POSITIVE` — `delta >= 0.005`
* `WEAK_SIGNAL` — `0.002 <= delta < 0.005`
* `NO_CLEAR_GAIN` — `delta < 0.002`
* `REGRESSION` — `delta < 0`
* `INCOMPLETE` / `INCOMPLETE_RESOURCE` / `FAILED` — not a result

The report must state that a single-seed screen cannot claim a stable gain,
Validation gains do not imply Test gains, attention weights are not chemical
causality, and a hashed-token gain is not automatically an "expressivity" gain
(extra parameters/regularisation are confounded). No mechanism claim and no Test
AP claim is made.

## 9. Deliverables

`PROTOCOL.md`, `README.md`, `code/`, `tests/`, `analysis/` (`architecture_audit.json`,
`feature_audit.json`, `preflight.json`, `run_registry.json`, `gpu_ledger.json`,
`epoch40_comparison.csv`, `per_task_comparison.csv`, `decision.json`), one run
directory per formal run, `REPORT.md`. Large checkpoints and caches stay
Git-ignored.

## 10. Deviation log (append-only)

1. **2026-11-06 — A0 rerun cancelled by principal instruction.** The A0 control
   was launched as required by §3 and cancelled at epoch 29/40 (last completed
   evaluation: epoch 20, Valid AP 0.238138) after the principal directed that
   the historical run not be reproduced. Consequence: A0 has **no same-round
   40-epoch result**. All deltas are computed against the historical
   `lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A` epoch-40 Valid AP
   `0.2463560956` (params 287,644), and the same-round implementation-drift
   control is **absent**. This deviation is carried into `decision.json`,
   `gpu_ledger.json` and `REPORT.md` and weakens every causal claim.
2. The cancelled attempt consumed 0.994 GPU-hours; it is charged to the budget.
3. No training-code change was made after the A0 launch. A1/A2/A3/COMBO run on
   the same training code; only documentation/analysis scripts differ.
4. **2026-11-06 — A3 evaluated every epoch** at the principal's request, i.e.
   40 official Validation evaluations instead of 10/20/30/40 (A1/A2 kept the
   frozen four points). Evaluation is inference only, consumes no RNG and never
   selects a checkpoint, so training is unchanged; the extra evaluations add
   ≈2.7 min and are reported as supplementary curve points. Epoch 40 remains
   the only primary comparison value.
