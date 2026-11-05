# PROTOCOL — LCRC PCBA no-edge-MP sprint v1

Round: `lcrc-pcba-no-mp-sprint-v1`  
Dataset: `ogbg-molpcba` (official OGB scaffold split, official macro AP).  
Frozen base: LCRC-C1, 245,184 parameters, from
`lcrc_pcba_scale_pilot_v1`; data, normalization, order, loss, optimizer,
learning-rate schedule, seeds and evaluation metric are inherited unchanged
from `lcrc_pcba_structural_gain_sprint_v1`.

This file is frozen before the first formal training run. No arm, width,
initialisation, stopping threshold, evaluation epoch or promotion rule may
change after seeing Validation AP. Official Test labels are NaN in the frozen
label artifact and are never indexed, scored or used for selection.

## 1. Question

Can explicit S1/S2 local structural statistics plus a capacity-matched static
computation or attention readout approach the PCBA Validation AP of the
previous round's V model **without iterative message passing along real
chemical bonds**?

V is the historical performance reference only. It is not retrained in this
round, and it is *not* a no-MP model (it propagates hidden states along
`bond_index`).

## 2. Frozen assets and comparability

Everything below is read-only inheritance from the previous rounds:

| Asset | Source |
|---|---|
| Feature cache `pcba_v1` | `lcrc_pcba_scale_pilot_v1/cache/pcba_v1` |
| Official split | `lcrc_pcba_scale_pilot_v1/analysis/pcba_split.json` |
| Labels (Test rows NaN) | `lcrc_pcba_scale_pilot_v1/analysis/pcba_labels_trainvalid.npy` |
| Feature spec | `lcrc_pcba_scale_pilot_v1/analysis/pcba_spec.json` |
| Train-only normalization | refit from official Train rows, fingerprint `6ce50f98…` |
| Data loader / order plan | `lcrc_pcba_lr_probe_v1/code/lr_data.py` |
| Pinned transfer | `lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.PinnedStager` |
| C1 builder / metric | `lcrc_pcba_scale_pilot_v1/code/pc_model.py`, `pc_metrics.py` |
| 40-epoch scratch recipe | `lcrc_pcba_structural_gain_sprint_v1/code/sg_common.py` |

Comparability gates, checked before formal training (see
`analysis/initialization_parity.json`, `analysis/protocol_comparability.json`):

1. Base C1 initial parameters for seed 0 and seed 1 equal the historical
   Phase-3 fingerprints bitwise (`1b1f1ad3…`, `3af93edb…`).
2. The 40-epoch batch-order plan for seed 0 and seed 1 equals the historical
   Phase-3 plan (`06f43729…`, `0af6ec6a…`) including the epoch-1/20/40
   prefix hashes.
3. Train-only normalization fingerprint, label artifact hash, split partition
   hash and cache manifest hash equal the frozen values.
4. Optimizer, betas, eps, weight decay, batch size 128, dropout 0.10, gradient
   clip 5.0, masked observed-entry BCE, FP32, 1-epoch warmup `3e-5 → 3e-4`,
   then per-step cosine to `1e-5` over 40 epochs (2,738 steps/epoch,
   109,520 total updates), no early stopping, seeds 0 and 1 only.
5. Test rows never enter the plan; no Test metric is produced.

If any gate fails, the historical C/V are marked *non-strict references* and
the formal runs stop.

## 3. Frozen arms

Notation: after the original local S1/S2 encode and dropout,
`h ∈ R^(N_atoms × 64)`. The original C1 prediction is
`logits_C1 = Σ_v f(h_v) + rho(graph_summary)`. New modules may only use `h`,
`atom_batch` and the molecule count; they must not read `bond_index`,
`bond_feat`, `nbr_*`, `path_*`, an adjacency matrix or any other graph tensor.
The original encoder may still use graph structure to produce `h`; the ban
applies to the **added** modules.

### Arm S — Static FFN2 (strictly no iterative MP)

Two per-atom pre-norm residual blocks:

```
z_v    = LayerNorm(h_v)
delta_v = Linear(160,64)(SiLU(Linear(64,160)(z_v)))
h_v    = h_v + delta_v
```

executed twice. Each atom's transform depends only on its own `h_v`. Final
Linear of each block is zero-initialised. New parameters: **41,664**.
Total: **286,848**.

### Arm A — Attention pooling (strictly no iterative MP)

`h` is not updated. Four permutation-invariant graph-level attention heads:

```
score_vk = Linear(32,1)(SiLU(Linear(64,32)(h_v)))      k = 1..4
alpha_vk = softmax over v in the same molecule only
z_gk     = Σ_v alpha_vk · h_v
z_g      = concat(z_g1, z_g2, z_g3, z_g4)                (256-d)
residual_logits = Linear(88,128)(SiLU(Linear(256,88)(z_g)))
logits_A = logits_C1 + residual_logits
```

The final `Linear(88,128)` weight and bias are zero-initialised, so at step 0
A equals C1. The segment softmax must be numerically stable and normalised
per molecule; no cross-graph mixing. No QKV atom–atom self-attention, no
virtual edges, no atom-state update. New parameters: **42,460**.
Total: **287,644**.

### Arm G — Graph-context only (no chemical-bond MP)

Two residual blocks, each:

```
z_v        = LayerNorm(h_v)
context_g  = Linear(64,64)(mean_{u in g} z_u)
delta_v    = Linear(128,64)(SiLU(Linear(64,128)(z_v + context_g)))
h_v        = h_v + delta_v
```

executed twice. No `bond_index`, `bond_feat`, edge mask or bond message. The
current molecule's mean context is broadcast to its atoms only. Final Linear
of each block is zero-initialised. New parameters: **41,728**.
Total: **286,912**.

G is **not** "no inter-node communication": it uses graph-level mean context
broadcast. It may only support the claim that non-bond, graph-global
computation can be competitive.

## 4. Correctness gates before formal training

CPU tests and a GPU preflight must pass for S, A and G:

* Parameter counts as above; new-parameter difference from historical V's
  42,624 is ≤3%.
* Forward/backward on real molecules; finite logits, loss and gradients;
  effective gradients into every new parameter after a few steps.
* Zero-residual identity: at initialisation each arm reproduces the C1 logits
  within the audited CUDA reduction tolerance (1e-4; no bitwise demand on
  `index_add`).
* Perturbing `bond_index`/`bond_feat` leaves each arm's new-module output
  bitwise unchanged when `h` and `atom_batch` are held fixed.
* S: no aggregation of other atoms' states.
* A: attention weights sum to 1 per molecule and head; no cross-molecule leak.
* G: context mean uses only atoms of the same molecule; no cross-molecule leak.
* Permutation of atom order (with `atom_batch` remapped) leaves graph-level
  logits invariant, and A's per-atom weights follow the permutation.
* Masked BCE equals the historical implementation.
* Train/Valid/Test isolation: no Test row is read.
* Step-time and memory smoke, and a forward-FLOP estimate per arm.

If a gate fails, fix only demonstrable software errors. Do not change model
semantics, recipe or decision rules after seeing Validation AP.

## 5. Training phases (pre-registered)

Two seeds (`0`, `1`), from scratch, 40 epochs, batch 128, exactly the Phase-3
recipe of the structural-gain sprint.

### Phase A — correctness + performance smoke

CPU tests, GPU preflight, parameter/FLOP audit, step timing, projected cost.

### Phase B — seed 0, first 20 epochs

Run S/A/G seed 0 for epochs 1–20. Evaluate at epochs **10 and 20**. Save a full
continuation checkpoint at epoch 20 (model, AdamW, scheduler step, RNG,
batch-order plan hash, completed updates). Three runs on at most two GPUs.

Pre-registered stop: if **all three** arms have AP20 < 0.220 **and**
AP20 − AP10 < 0.002, label `EARLY_NO_SIGNAL`, stop, do not continue. This is a
resource rule, not evidence that static models cannot reach higher AP.

### Phase C — seed 0 to epoch 40

Continue all three from their epoch-20 checkpoints to epoch 40 without
re-initialising model or optimizer. Fixed main comparison: **epoch 40**.
Report Validation AP, Train AP, Train/Valid BCE, per-task AP, differences vs
the historical C seed 0 and V seed 0, number of improved tasks, GPU-hours.

Pre-registered stop: if **all three** arms have epoch-40 AP < 0.242, label
`NO_STRONG_SEED0_SIGNAL` and skip Phase D. Do not switch the primary epoch to a
better-looking intermediate epoch.

### Phase D — seed 1 independent repetition

Only if Phase C did not trigger a stop and all three seed-1 runs can be
completed within the remaining budget. Train S/A/G seed 1 from scratch for 40
epochs; never load seed-0 parameters or epoch-20 checkpoints. Evaluate at
epochs 10/20/30/40; primary comparison epoch 40.

If the three seed-1 runs cannot all be completed, run none of them, report
`SECOND_SEED_NOT_RUN_RESOURCE_LIMIT`, and do not present a selected arm as a
"confirmation".

## 6. Pre-registered interpretation

Official Validation macro AP over the 127 tasks with observed Validation
labels. Historical references: C seed 0 `0.2319447737`, C seed 1
`0.2269055080`; V seed 0 `0.2577723248`, V seed 1 `0.2589524573`; V mean
`0.2583623910`.

For the strict no-MP route, choose the better of S and A **as an architecture
by the two-seed mean epoch-40 AP**, not per seed, task or epoch. Let
`AP_static_mean` be that architecture's two-seed mean.

* `STATIC_CLOSE_TO_V`: Phase D complete; S or A has both seeds ≥ 0.250 and the
  two-seed mean is within 0.008 of `AP_V_mean`; both seeds improve over the
  paired historical C by ≥0.005; no correctness issue.
* `STATIC_POSITIVE_BUT_BELOW_V`: both seeds clearly above historical C but
  `STATIC_CLOSE_TO_V` not met.
* `NO_STATIC_GAIN`: no stable positive improvement, or the seed-0 gate was not
  reached.
* `INCOMPLETE`: a key run or data audit is missing.

Report G separately. If Phase D is complete, G's two-seed mean AP is within 0.008 of
`AP_V_mean`, and G's mean exceeds the best S/A two-seed mean by at least
0.008, label `GLOBAL_COMMUNICATION_PROMISING` (graph-level context feedback,
not bond MP, may recover much of V's gain). Do not merge G with S/A.

If V's two-seed mean exceeds every S/A/G two-seed mean by at least 0.008,
report `EDGE_MP_BUNDLE_RETAINS_ADVANTAGE` without claiming that edge MP is
theoretically necessary, that S1/S2 is proven insufficient, or that attention
weights are causal chemical explanations.

## 7. Resource budget

Target wall clock ≤7.5 h, hard ≤8 h. Target ≤15 A100 GPU-hours, hard ≤16.
At most two GPUs concurrently. All smoke, failed attempts and formal runs count.
No new GPU stage starts when <30 min remain before the hard wall.

## 8. Governance and deliverables

All remote execution through `rr` (`doctor`, `deploy`, `run`, `status`, `logs`,
`cancel`, `pull`) on host `res`, pools `res-gpu0` / `res-gpu1`. Formal runs are
launched only from a committed revision, with remaining wall/GPU budget and
projected cost recorded first. No direct SSH/nohup.

Deliverables under this directory: `PROTOCOL.md`, `README.md`, `code/`,
`tests/`, the analysis JSON/CSV files named in the round brief, one run
directory per formal run (curve, fixed-epoch reports, per-task AP, checkpoints,
optimizer/RNG state, identity, Git commit, stop reason), and `REPORT.md`
answering the seven fixed questions. Official Test is never run in this round.
