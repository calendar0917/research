# PROTOCOL — LCRC PCBA Backbone Scaling v1

Round id: `lcrc-pcba-backbone-scaling-v1`
Status: **frozen before the first formal training step** (see §12 for the
amendment log; the only permitted later edit is the measured Validation
cadence in §6, which is a deterministic function of the rule frozen here).

## 1. The single question

Under the frozen LCRC-A computation — one atom encode, one bond encode, one-shot
S1/S2 local statistics, no iterative message passing, no cross-atom QKV, the
original masked BCE and the original 40-epoch from-scratch recipe — does scaling
the **whole trunk** from 0.288 M to ≈1 M and ≈2.1 M parameters produce a large,
broad MolPCBA Validation AP gain at the fixed epoch 40?

This is a pure **capacity scan**. It is not an architecture search: no module is
added, removed, re-ordered or re-defined, and no new representation is
introduced.

## 2. Frozen reference

| item | value |
|---|---|
| dataset | `ogbg-molpcba` |
| split | official OGB scaffold split |
| seed | 0 |
| epochs | 40 |
| parameters | 287,644 |
| epoch-40 official Validation macro AP | 0.2463560956 |
| epoch-40 Train AP / Train masked BCE | 0.4196570175 / 0.0308577786 |
| source | `lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A` (arm `A`, commit `b1b917dc`) |
| checkpoint SHA-256 | `f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437` |

A0 is **never retrained**. Only read-only checkpoint reloading, parameter
inspection, forward replay and one full Validation evaluation are allowed
(`analysis/a0_replay.json`).

## 3. The two new models

| model | scale | parameters |
|---|---:|---:|
| `W2` = LCRC-A-Wide-2x | 2 | 994,324 |
| `W3` = LCRC-A-Wide-3x | 3 | 2,120,844 |

One seed, one formal from-scratch 40-epoch run each. No additional seed, no
combination run, no warm start from A0, no parameter-expansion initialisation.

### 3.1 Exact widths

| parameter | A0 (read-only) | W2 | W3 |
|---|---:|---:|---:|
| `atom_hidden` | 128 | 256 | 384 |
| `atom_out` | 96 | 192 | 288 |
| `bond_hidden` | 160 | 320 | 480 |
| `bond_out` | 96 | 192 | 288 |
| `stat_rank` | 32 | 64 | 96 |
| `local_hidden` | 192 | 384 | 576 |
| `width` (local output) | 64 | 128 | 192 |
| `rho_hidden` | 64 | 128 | 192 |
| attention scorer hidden | 32 | 64 | 96 |
| attention `pool_hidden` | 88 | 176 | 264 |
| attention heads | 4 | 4 | 4 |
| task outputs | 128 | 128 | 128 |

Every width is `scale × the A0 value` except the head count (4) and the task
count (128), which are frozen.

### 3.2 Frozen input vocabulary (never scaled)

```
atom_feat_dim = 51      bond_feat_dim = 13
atom_scalar_dim = 9     global_dim = 42
n_nbr_codes = 80        n_path_codes = 400
close_emb = Embedding(2, 8)          path_gate = Embedding(400, 1)
```

### 3.3 Parameter accounting

`sum(p.numel())` must equal the three targets exactly, including every bias,
LayerNorm and embedding table. The expected split is

| scale | trunk (`C1` graph) | attention readout | total |
|---:|---:|---:|---:|
| 1 | 245,184 | 42,460 | 287,644 |
| 2 | 848,096 | 146,228 | 994,324 |
| 3 | 1,809,408 | 311,436 | 2,120,844 |

If any count disagrees the round stops; parameters are never added or removed to
make a target.

### 3.4 Construction rule (no historical default may leak)

* `tu_model.TULCRC(spec, norm, 128, variant="C1", cfg=<explicit>)` supplies the
  atom encoder, bond encoder, S1/S2 `TULocalContent`, `f` and `rho`. `cfg` is
  always passed explicitly — the module-level `tu_common.MODEL_CONFIG` is never
  modified and never relied on.
* `nmp_model.AttentionPoolReadout` is imported unchanged and constructed with
  `width`, `hidden=scorer_hidden`, `n_heads=4`, `pool_hidden` and `n_out` passed
  explicitly, so the historical `width=64 / scorer_hidden=32 / pool_hidden=88`
  defaults cannot survive into W2/W3.
* `cont_mean` / `cont_scale` empty buffers are reproduced so the state dict keys
  equal the historical LCRC-A keys exactly.
* Historical code (`tu_model.py`, `nmp_model.py`, `bp_model.py`, `sg_common.py`,
  `sg_data.py`, `pc_metrics.py`, `tu_common.MODEL_CONFIG`) is imported read-only
  and is not edited in this round.

### 3.5 Initialisation rule (identical at every scale)

```
torch.manual_seed(0)
  -> build the scale's FULL TULCRC trunk (CPU)
  -> remap to C1 (drop the 3*atom_out b_mean columns of local.mlp.lins.0)
  -> load into the scaled model
  -> attention readout: private generator seed*1_000_003+17, uniform(-1/sqrt(fan_in),
     +1/sqrt(fan_in)), then readout.fc2.{weight,bias} := 0
```

This is the historical LCRC-A initialisation, parameterised by scale. At
`scale = 1` it must reproduce A0's trunk fingerprint
`1b1f1ad35ab33f4107bc600166fce838d5e16ff5def2348743c73c0211dcf7c3` and A0's
full initial state **bitwise**; that equality is a gate (§4.1).

Consequence, recorded in advance: because `readout.fc2` is zero at step 0, the
readout branch carries no gradient on the first step and receives gradient from
step 1 onwards. This is a property of the frozen historical A initialisation,
not a training failure.

W2 and W3 are different shapes; no per-tensor weight or dropout-mask equality
between widths is claimed or required. What is required and audited is: the same
seed definition, the same data order, the same recipe, and a reproducible
initialisation for each model.

## 4. Pre-training audit (must pass before any formal step)

Recorded in `analysis/model_scaling_audit.json`, `analysis/a0_replay.json`,
`analysis/gpu_preflight.json` and `analysis/evaluation_cost_probe.json`.

1. `scale=1` reproduces the historical LCRC-A state-dict keys, shapes and
   parameter count; its initial state is bitwise equal to
   `nmp_model.build_arm("A", ...)`.
2. `scale=1` reproduces A0's forward logits (max |Δ| = 0 required on CPU) both
   from the frozen initial state and after loading the historical epoch-40
   checkpoint. On CUDA the same comparison is bounded by the audited identity
   tolerance 1e-4 (observed 3.8e-5, relative 4e-6 of the logit scale), because
   `index_add_` / `scatter_reduce_` accumulate with atomics in a
   non-deterministic order; the two implementations agree to 8.5e-7 in AP.
3. `scale=1` with the historical checkpoint reproduces the official Validation
   AP 0.2463560956 within 1e-4.
4. W2/W3 parameter counts are exactly 994,324 / 2,120,844.
5. Every module's observed width equals §3.1 (tensor shapes, not intent).
6. S1/S2 inputs, code vocabularies, path counts, close-embedding and gate
   tables are identical across scales and equal to A0's.
7. One `atom_encoder`, one `bond_encoder`, one `local` call per forward
   (forward-hook counts).
8. Nothing after `local` reads a bond or path field: the post-`local` path is a
   pure function of `(h, atom_batch, n_graphs, global)`; supplying a batch with
   no `bond_index`/`bond_feat`/`nbr_*`/`path_*` gives identical logits (exact on
   CPU, ≤2e-5 on CUDA, observed 1.9e-6, for the same atomic-accumulation reason
   as item 2).
9. Atom renumbering inside each molecule leaves graph logits unchanged
   (tolerance 2e-5) and permutes the per-atom contribution exactly.
10. `global` / `atom_scalar` standardisation is the frozen Train-only version
    (`norm_fingerprint 6ce50f98…`, `fit_rows 350343`), bitwise equal to A0's
    buffers at every scale.
11. Forward and backward are legal, loss/logits finite, and every parameter
    receives a gradient after three real steps (the only structural zero is the
    known `readout.heads.*.fc2.bias`).
12. No official Test row is indexed, scored or inferred on: Test labels are NaN
    in the frozen artifact and no code path reads them.
13. No NaN/Inf and no silent fallback to a historical fixed width (5 + 7 + 5).

## 5. Training protocol (frozen)

From random initialisation; no A0 weights, no parameter expansion.

```
seed = 0                     epochs = 40            batch_size = 128
optimizer = AdamW            peak_lr = 3e-4         weight_decay = 1e-3
betas = (0.9, 0.999)         eps = 1e-8
warmup = 1 epoch linear      warmup_start_lr = 3e-5
scheduler = per-step cosine  final_lr = 1e-5
dropout = 0.10               gradient_clip_norm = 5.0
precision = FP32             loss = original observed-entry masked BCE
early_stopping = false
```

The schedule is exactly `sg_common.scheduled_lr(step, phase=3, kind="all")`
(2,738 warm-up steps, 109,520 total updates); the audited values are
`lr(0) = 3.0000986e-5`, `lr(2737) = 3e-4`, `lr(109520) = 1e-5`.

Data: Train 350,343 / Valid 43,793 / Test 43,793 (**never evaluated**),
2,738 steps per epoch, 109,520 optimizer updates.

The seed-0 40-epoch batch-order plan is the frozen historical plan
(`global_plan_sha256 06f43729…`); both models consume the same raw row-id order.

Batch size, precision, gradient accumulation and training-set size are never
changed for throughput.

## 6. Validation cadence (rule frozen here, value measured)

```
default: every epoch (1..40); primary checkpoint = epoch 40
extra : epoch 0 evaluated once as an initialisation diagnostic
primary metric: official OGB Validation macro AP at epoch 40 (no best-epoch substitution)
```

Before the first formal step, both models are probed
(`analysis/evaluation_cost_probe.json`): a full 43,793-row Validation forward
plus metric pass, and real training steps (5 warm-up + 30 measured, real
collate/stager/model/optimizer/clip/loss path).

Decision rule, applied uniformly to both models:

* if **both** models' full Validation pass is ≤ **15 %** of that model's
  projected epoch time → evaluate **every epoch**;
* if **either** model exceeds 15 % → **both** switch to **every 2 epochs**
  (2, 4, …, 40).

The cadence is one value for both models; it is written into §6.1 below and into
`analysis/decision.json` and is not modified afterwards.

### 6.1 Resolved cadence

```
RESOLVED: every epoch (1, 2, ..., 40)
```

Measured on `res` GPU1 (idle) by `bs-audit` run `bs-audit-20261107-171801-d3eaa1b8`
(`analysis/evaluation_cost_probe.json`):

| model | step p50 | projected epoch | full Validation pass | fraction | decision |
|---|---:|---:|---:|---:|---|
| W2 | 28.73 ms | 78.66 s | 5.99 s | 7.61 % | ≤ 15 % |
| W3 | 27.17 ms | 74.39 s | 6.07 s | 8.16 % | ≤ 15 % |

Both models are below the 15 % threshold, so the frozen rule selects **every
epoch** for both. Projected cost of the two formal runs: 1.86 GPU-hours
(W2 0.95, W3 0.91), inside the 8-hour target.

Measured peak training memory: W2 158 MB, W3 236 MB allocated.

### 6.2 Evaluation discipline

`model.eval()`, `torch.inference_mode()`, the original OGB `Evaluator`, the 127
scorable tasks plus the correct handling of the unscorable one (task 45, zero
Validation positives), no weight update, no RNG consumption (verified by
before/after RNG hashes), no Test access, no per-epoch checkpoint proliferation
(only the resumable `checkpoint_last.pt`, the diagnostic best-AP checkpoint and
the final epoch-40 checkpoint).

Per epoch we record: epoch, global_step, learning rate, Train masked BCE,
Validation macro AP / BCE / scored tasks (when evaluated), epoch train seconds,
epoch valid seconds, peak GPU memory, grad-norm max/mean, gradient-clip events,
step p50/p90 ms, cumulative seconds, RNG hashes.

## 7. Execution (two GPUs in parallel)

Only through `rr` (`rr doctor res` / `rr deploy res` / `rr run res …`); no manual
SSH, `nohup` or `sbatch`.

Preference: `W3 → res-gpu1`, `W2 → res-gpu0`.

Before scheduling W2 on GPU0 the preflight must show: sufficient free memory for
the projected peak plus a safety margin, no diagnostic failure, and no
sustained full-load contention. Sharing GPU0 is allowed only under those
conditions; if GPU0 is saturated or diagnostics fail, W3 runs on GPU1 first and
W2 runs afterwards on GPU1 (or on GPU0 once safe). External processes are never
cancelled, paused, moved or modified.

Recorded per run: rr run id, pool, actual CUDA device, host/GPU name, concurrent
job state, start/end timestamps, throughput, GPU memory peak, GPU-hours, and
whether any contention affected throughput (reported as a resource fact, never
as an architecture-efficiency claim).

## 8. Budget

```
target ≤ 8 A100 GPU-hours      hard cap ≤ 12 A100 GPU-hours
```

The cap covers preflight, both formal runs, Validation evaluation, and failed
attempts/retries. The measured projection is written to
`analysis/evaluation_cost_probe.json` before training; if the projection exceeds
the hard cap the round stops with `BUDGET_NOT_FEASIBLE` and widths/lengths are
**not** reduced to make it fit.

A failed formal run is first resumed from its complete last-epoch checkpoint
(model + AdamW + scheduler + RNG + plan); a completed 40-epoch run is never
rerun.

## 9. Metrics and labels

```
AP0   = 0.2463560956
AP_W2 = W2 epoch-40 official Validation AP
AP_W3 = W3 epoch-40 official Validation AP
D_W2  = AP_W2 - AP0      D_W3 = AP_W3 - AP0      D_32 = AP_W3 - AP_W2
```

Also reported: Train AP40, Train masked BCE40, Validation BCE40, Train−Valid AP
gap (descriptive only — the gap is not by itself evidence of overfitting because
the split's positive rate and scaffold distribution differ), the full Validation
AP curve, the best diagnostic Validation AP and its epoch, the epoch 30→40
change, per-task AP for 127 scorable tasks (improved/regressed counts, mean and
median Δ, strata by Validation positives ≥20 / 5–19 / <5, Spearman vs support,
top-5/top-10 concentration of positive Δ), parameters vs Validation/Train AP,
per-epoch cost vs parameter count, and the complete GPU ledger.

Resource-decision labels (per model, resource allocation only — not a
significance test):

| label | rule |
|---|---|
| `PROMISING_SCALE_SIGNAL` | Δ AP40 ≥ +0.005 **and** mean Δ on ≥20-positive tasks ≥ +0.003 **and** improved ≥20-positive tasks ≥ 60/107 |
| `WEAK_SCALE_SIGNAL` | +0.002 ≤ Δ AP40 < +0.005, **or** Δ ≥ +0.005 without sufficient high-support coverage |
| `NO_CLEAR_SCALE_GAIN` | −0.002 < Δ AP40 < +0.002 |
| `SCALE_REGRESSION` | Δ AP40 ≤ −0.002 |

Four separate judgements are reported independently: does Train AP increase with
width; does Validation AP increase with width; is W3 better than W2; is any gain
broad or concentrated in sparse assays.

If Train AP rises while Validation AP does not, the conclusion is limited to
"with this from-scratch recipe the added capacity did not convert into a
Validation gain" — **not** "LCRC has reached its capacity limit". If either model
is still clearly rising at epoch 40 we report that fact and do **not** extend the
run.

## 10. Deliverables

```
PROTOCOL.md  README.md  REPORT.md
code/bs_common.py code/bs_model.py code/bs_train.py code/bs_audit.py code/bs_compare.py
tests/test_bs_model.py tests/test_bs_training.py
analysis/{model_scaling_audit,a0_replay,evaluation_cost_probe,gpu_preflight,
          w2_epoch40,w3_epoch40,scale_comparison,decision,gpu_ledger}.json
analysis/per_task_deltas.csv
runs/W2/ runs/W3/            (git-ignored)
```

`runs/` keeps the final checkpoint, the resumable optimizer/RNG/scheduler state,
the curves, per-task AP, the order plan and the SHA-256 of each.

## 11. Explicitly out of scope

No change to the S1/S2 definition, radius or enumeration; no S3, motif, RIE,
SSL or Conditional S2; no iterative edge or node message passing; no cross-atom
QKV attention; no extra residual blocks or depth; no tuning of dropout, loss,
LR recipe or augmentation; no Test use for evaluation or selection; no
best-epoch substitution for the primary comparison; no automatic extra seeds or
combination arms; no length extension.

## 12. Amendments (append-only)

| # | date | change | reason |
|---|---|---|---|
| 1 | 2026-11-08 | §4 items 2 and 8: record the CUDA tolerance alongside the CPU-exact requirement (1e-4 for the cross-implementation logits comparison, observed 3.8e-5; 2e-5 for the post-`local` re-run, observed 1.9e-6) | clarification of the same claim, fixed before any formal step; CUDA `index_add_` is not bitwise stable. The primary evidence for item 2 is the CPU bitwise equality (0.0) and the 8.5e-7 AP agreement |
| 2 | 2026-11-08 | §6.1 resolved cadence = every epoch (1..40) | deterministic output of the frozen 15 % rule: both models 7.6 % / 8.2 % of a projected epoch |
| 3 | 2026-11-08 | §5 applied: GPU0 not shared, both runs serial on `res-gpu1` | not a rule change — the co-tenancy clause "no sustained full-load contention" failed (GPU0 at 100 % in 6/6 samples over 60 s from an external job); see `analysis/gpu0_occupancy.json` |
