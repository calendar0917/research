# PROTOCOL — LCRC PCBA structural-gain sprint v1

Round: `lcrc-pcba-structural-gain-sprint-v1`
Type: performance sprint with pre-registered arms, promotion rules and stop rules.
Dataset: `ogbg-molpcba` (official OGB scaffold split; official macro AP).
Model source: frozen LCRC-C1 (245,184 parameters) from
`tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1`.
Warm start: `tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/runs/L_low_lr/checkpoint_best.pt`
(epoch 9, SHA-256 `b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5`).

This file is frozen before the first formal training run. No arm, width,
learning-rate rule, evaluation epoch, promotion threshold or comparison epoch
may change after seeing Validation AP. The official Test targets are never
read, indexed, scored or used in any selection.

## 1. Question

The frozen model performs one S1/S2 local structural encode, then sums per-atom
contributions `f(h)` and adds a graph correction `rho`. Which of three added
mechanisms — extra local FFN capacity, physical-bond neighbourhood propagation,
or graph-level context feedback — improves PCBA scaffold generalization on top
of the existing structural statistics?

This is a performance screen, not a causal mechanism proof.

## 2. Frozen data and model assets

| Asset | Path | Frozen identity |
|---|---|---|
| Feature cache | `lcrc_pcba_scale_pilot_v1/cache/pcba_v1` | manifest `caad3e90…` |
| Official split | `lcrc_pcba_scale_pilot_v1/analysis/pcba_split.json` | partition `c872bd63…` |
| Labels | `lcrc_pcba_scale_pilot_v1/analysis/pcba_labels_trainvalid.npy` | `5c3b5fbe…`, Test rows NaN |
| Feature spec | `lcrc_pcba_scale_pilot_v1/analysis/pcba_spec.json` | `7270f767…` |
| Train-only norm | refit from official Train rows | fingerprint `6ce50f98…` |
| Warm start | `lcrc_pcba_lr_probe_v1/runs/L_low_lr/checkpoint_best.pt` | `b158f166…`, epoch 9, AP 0.2374902968831333 |
| Data transfer | `lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.PinnedStager` | already gated |

The warm-start checkpoint must pass, before any formal run:

* SHA-256 equals the frozen value and schema/round/group/epoch/AP metadata match.
* `state_dict`: 37 tensors, bitwise load into a freshly built LCRC-C1.
* AdamW state: exactly one group, parameter ids `0..30`, `exp_avg`/`exp_avg_sq`
  shapes equal to the model parameters, every `step` equal to 24,642
  (`9 * 2738`), restored without resetting any moment.
* Feature-spec hash, normalization fingerprint, split partition hash, label
  artifact hash and cache manifest hash all match.
* 128-task output shape.

No fallback checkpoint is permitted. If the warm start is unusable, the round
stops.

## 3. Arms

All arms keep the original AtomEncoder, BondEncoder, S1/S2 `LocalContent`,
dropout 0.10, per-atom `f`, graph `rho`, global descriptor and normalization.
New computation is inserted **after** `h = dropout(LocalContent(...))` and
**before** `f`/`rho`. Final logits remain `contrib + rho`; no new prediction
head.

Notation: 64-d atom state, `B` = pairs of real undirected physical bonds
(cache `bond_index`, `u < v`, no invented edges), `bf_uv` = the 13-d raw bond
feature of that bond, `atom_batch` maps atoms to graphs.

**Arm C — control.** Frozen C1, no new parameters, same warm start, same
40-epoch training protocol, full paired screen.

**Arm F — FFN2.** Two pre-norm residual blocks, applied per atom independently:

```
z   = LayerNorm64(h)
res = Linear128->64( SiLU( Linear64->128(z) ) )
h   = h + res
```

**Arm M — MP2.** Two pre-norm residual blocks with the same FFN shape as F, but
the normalized input receives directed physical-bond messages:

```
P(bf) = bond_proj(bf)                    # one shared Linear(13 -> 64)
z     = LayerNorm64(h)
msg_v = sum_{u->v in B} SiLU( z_u + P(bf_uv) )
res   = Linear128->64( SiLU( Linear64->128( z_v + msg_v ) ) )
h     = h + res
```

Each undirected bond yields exactly two directed messages; source is
`cat(bond_index[:,0], bond_index[:,1])`, destination is
`cat(bond_index[:,1], bond_index[:,0])`; the same `bf_uv` is used in both
directions. No self loops and no virtual edges are added.

**Arm V — MP2 + graph context feedback.** As M, plus, in each block, after the
same adjacency messages:

```
g      = mean_{(v in graph g)} z_v          # current block input, same graph only
ctx_v  = Linear64->64( g )
res    = Linear128->64( SiLU( Linear64->128( z_v + msg_v + ctx_v ) ) )
h      = h + res
```

`ctx` has an independent 64→64 map per block. No external data, labels or
assay identity enter the context.

Parameter accounting (fixed by construction):

| Arm | New parameters |
|---|---|
| C | 0 |
| F | 2 × (LayerNorm 128 + 64×128 + 128 + 128×64 + 64) = 33,408 |
| M | F + (13×64 + 64) = 34,304 |
| V | M + 2 × (64×64 + 64) = 42,624 |

`|M − F| / F = 2.68% ≤ 5%`; widths must not be tuned later.

## 4. Initialization and strict pairing

* All four arms load the identical warm-start `state_dict` into the frozen C1
  part, bitwise.
* New block final `Linear` weight and bias are zero. Therefore at global step 0
  every new residual is exactly zero and all four arms produce the same
  predictions as C1 under the same dropout RNG.
* F and M share identical FFN initial states (one private generator, identical
  consumption order; `bond_proj` is initialized after the two blocks).
* New modules are initialized with a private `torch.Generator`, never with the
  global RNG, so the public dropout stream is unchanged.
* The four arms use the same complete 40-epoch Train batch-order plan
  (`torch.Generator().manual_seed(0)` + per-epoch `randperm`, batch 128), of
  which Phase 1 consumes the first 12 epochs.
* AdamW restoration: build one optimizer over exactly the 31 base parameters in
  source order, `load_state_dict` the source moments, audit bitwise equality and
  step 24,642, then append the new parameters as a second param group. Group 0
  moments and steps are never reset.

## 5. Common training protocol (Phase 1/2)

| Item | Value |
|---|---|
| Start | epoch-9 L-low-lr best checkpoint (weights + AdamW moments) |
| Optimizer | AdamW, wd 1e-3, betas (0.9, 0.999), eps 1e-8 |
| Base parameter LR | per-step cosine 5e-5 → 1e-5 over a 40-epoch horizon |
| New parameter LR | per-step cosine 3e-4 → 3e-5 over the same horizon |
| Scheduler | `lr(s) = lo + (hi-lo) * 0.5 * (1 + cos(pi * s / S40))`, `S40 = 40 * 2738`, applied before step `s` |
| Warmup | none (the source optimizer state is not re-warmed) |
| Batch | 128 |
| Dropout | 0.10, inherited |
| Gradient clip | 5.0 |
| Loss | historical masked BCE, observed entries only |
| Precision | FP32 |
| Epochs | 40 total; Phase 1 = epochs 1–12; Phase 2 = epochs 13–40 |
| Early stopping | none |
| Evaluation | Phase 1 epochs 0/4/8/12; Phase 2 epochs 20/30/40 |

This scheduler is a **new common training protocol** for the round, not a claim
of seamless continuation of the historical epoch-9 optimizer trajectory (the
scheduler state is new; AdamW moments are restored).

At Phase 1 epoch 12 the checkpoint stores model state, AdamW state, scheduler
state, RNG state, consumed steps and batch-plan hash, so Phase 2 is a true
continuation (not a reload of epoch 9).

## 6. Phase 1 promotion rule

`AP0 = 0.2374902968831333`; `AP_C12` = Arm C epoch-12 official Validation AP.
For F/M/V: `D_base = AP_arm12 − AP0`, `D_control = AP_arm12 − AP_C12`.

A candidate is promoted only if all hold:

1. the arm completed all 12 epochs;
2. `D_base >= +0.004`;
3. `D_control >= +0.004`;
4. at least 70 of the 127 scorable tasks improve AP versus C;
5. the mean AP difference on tasks with ≥20 Validation positives is ≥ 0;
6. no data, numerical or configuration anomaly.

If several qualify, take the highest epoch-12 AP; if within `1e-5`, prefer the
simpler structure in the order F, M, V. At most one candidate is promoted.

If none qualifies: report `SCREEN_NO_PROMOTION`, keep all four curves, and stop
all PCBA GPU work.

## 7. Phase 2 promotion rule

Fixed comparison: `D40 = AP_W40 − AP_C40` at the pre-registered epoch 40.

Phase 3 is allowed only if: W and C both complete epoch 40; `AP_W40 >= 0.245`;
`D40 >= 0.006`; W improves at least 70/127 tasks versus C; the ≥20-positive
subset mean difference is ≥ 0; and no correctness/execution anomaly.

Otherwise report `NO_SUSTAINED_GAIN` and stop. A best-epoch AP above 0.25 is
exploratory only and cannot replace the fixed epoch-40 comparison.

## 8. Phase 3 (only if Phase 2 passes)

Four from-scratch 40-epoch runs: {C1, W} × {seed 0, seed 1}. Same seed → same C1
init, same minibatch order, same dropout seed; only W's added structure
differs. AdamW 3e-4 for all parameters, wd 1e-3, batch 128, dropout 0.10,
clip 5.0. One warmup epoch: linear 3e-5 → 3e-4 over steps 0…2737, then per-step
cosine 3e-4 → 1e-5 over the remaining steps of the 40-epoch horizon. Evaluation
epochs 10/20/30/40; epoch 40 is the primary comparison.

Confirmation labels:

* `PCBA_025_REPRODUCED`: both W seeds have epoch-40 Validation AP ≥ 0.250 and
  beat their C seed by ≥ 0.005.
* `POSITIVE_BUT_NOT_REPRODUCED`: positive gains that fail the double-seed rule.
* `NO_REPRODUCIBLE_GAIN`: no positive, sufficiently large double-seed gain.
* `CONFIRMATION_INCOMPLETE`: fewer than four complete audited runs.

If the four 40-epoch runs cannot fit in the remaining wall clock and GPU budget,
Phase 3 is skipped as a whole and labelled
`CONFIRMATION_NOT_RUN_RESOURCE_LIMIT`.

## 9. Phase 4 (optional, only after `PCBA_025_REPRODUCED`)

Only with ≥45 min wall clock and ≥0.8 GPU-hour left. MoleculeNet Lipophilicity,
existing internal scaffold split (Fit 2352 / Monitor 504; Reserved and official
Test never used). F-small vs M-small, 2 layers, matched new-parameter count
≤5%, seeds 0 and 1, 40 epochs from scratch, AdamW 3e-4, wd 1e-3, batch 128,
Train-only normalization, FP32, same order and regression loss; primary metric
Monitor RMSE at fixed epoch 40, best-monitor epoch auxiliary. This is
exploratory evidence on an already-used dataset, not independent confirmation.
If the data adaptation would exceed the remaining budget, skip and say so.

## 10. Correctness gates (before any formal run)

All must pass:

1. source checkpoint SHA, metadata, 37-tensor `state_dict`;
2. four arms' epoch-0 Validation AP reproduce `0.2374902968831333` within 1e-4;
3. with zero new residuals, each arm reproduces C1 logits under the same input
   and RNG;
4. correct bidirectional edge mapping;
5. atom-order permutation leaves graph predictions unchanged (tolerance
   audited for reduction noise);
6. F's **new branch** does not depend on adjacency when its atom states are
   fixed;
7. M's new branch responds to changing real adjacency after residual activation;
8. V's context only pools atoms of the same graph (no cross-graph contamination);
9. gradients reach the new modules after at least two optimizer steps,
   including M `bond_proj` and V context projections;
10. masked BCE matches the historical definition;
11. Train/Valid/Test isolation (Test labels all NaN; order plan contains only
    Train rows);
12. parameter counts, input shapes and finite values.

CPU tests, GPU preflight (epoch-0 AP, identity, adjacency, context, gradient),
30-step smoke timing and the optimizer-state audit must all pass first.

## 11. Resource policy

Host `res`, pools `res-gpu0`/`res-gpu1`, at most two GPUs concurrently. Target
wall clock 7.5 h, hard 8 h; target 15 A100 GPU-hours, hard 16. Every preflight,
smoke, failed and formal run counts. No new GPU stage starts with <30 min left.
Before each stage, measure step speed and check that the entire paired matrix
fits; if not, skip the whole stage. Test targets remain disabled.

## 12. Deliverables

`PROTOCOL.md`, `README.md`, `code/`, `tests/`, `analysis/initialization_audit.json`,
`analysis/architecture_audit.json`, `analysis/phase1_screen.json`,
`analysis/phase2_continuation.json` (if run), `analysis/phase3_seed_confirmation.json`
(if run), `analysis/phase4_lipo_probe.json` (if run), `analysis/gpu_ledger.json`,
`analysis/run_registry.json`, `REPORT.md`, plus per-stage `runs/`.
