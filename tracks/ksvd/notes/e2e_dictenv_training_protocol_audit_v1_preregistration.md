# Pre-registration — `e2e_dictenv_training_protocol_audit_v1`

Workstream Z (ZINC dictionary-environment line), CPU only, no GPU, no SSH, no
remote compute.  The official ZINC test split is **never** loaded
(`official_test_loaded = false` in every payload).

Round name: `e2e_dictenv_training_protocol_audit_v1`.

Implementation (frozen with this document):
`tracks/ksvd/experiments/luyin16/e2e_dictenv_training_protocol_audit_v1.py`;
runner:
`tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_training_protocol_audit_v1.py`;
focused tests:
`tracks/ksvd/tests/test_e2e_dictenv_training_protocol_audit_v1.py`.

Nothing in this document may be edited after the first training process starts.
No threshold, gate, fork epoch, learning rate, seed or horizon may be changed
after seeing a result.

---

## 0. The single question

> How much of the current CSSD-q1 plateau (`~0.128–0.130` Top-5 soup valid MAE)
> comes from the frozen training protocol (fixed `Adam lr = 1e-3` run for the
> full 320 epochs), rather than from the architecture itself?

This is a **training-protocol audit**, not an architecture round.  One
architecture only: the frozen CSSD-q1 (C6 clean mask + q1 common structural
coordinate + q1-orthogonal sparse residual dictionary K=32/s=8/IHT-10 + paired
node/edge structure-semantic binding + full clean relation + first/second
moment readout).  Parameter count must be exactly `97727`; a mismatch stops the
round.

Forbidden and not re-opened anywhere in this round: F/R/G, multi-head or
multi-rank fusion, centre update, relation refresh, message passing, new
readout, new features, sparse-vs-dense, K/s/λ changes, CSSD q2, descriptor
cleanup, new chemistry or topology, architecture comparisons, seed 1/2.

---

## 1. Frozen context (not re-opened)

| item | value |
|---|---|
| architecture | `CSSD-q1` (exactly `CSSDModel` from the closed `e2e_dictenv_common_subspace_dictionary_v1` module) |
| parameters | `97727` |
| train-only q1 subspace | `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json`, q1 block (sha of the canonical `[components; rms]` payload recorded in preflight) |
| dictionary | `sdb32` (`D = 65 x 32`), `sha256(D.tobytes()) = b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd` |
| data | frozen phi65 cache, official ZINC train 10 000 / valid 1 000; official test never loaded |
| seed | `0` only |
| loss | `L1(y_hat, y) + H1_LAMBDA * reconstruction`, `H1_LAMBDA = 33.95873017865987` (frozen) |
| frozen historical CSSD-q1 seed-0 soup | `0.13002798487985273`, members `[282, 299, 307, 312, 313]` |
| frozen historical epoch-280 valid MAE | `0.14115870875114342` |
| CSSD-q1 capacity-localization-v2 warm M0 soup (`lr=1e-4`) | `0.1287231611124589` (context only) |
| FINAL-CLEAN sparse seed-0 soup | `0.12849851670576026` (context only, not a matched control) |
| measured CPU soup-level reproducibility floor (v2) | `~6e-4` |

---

## 2. Design: one shared prefix + one exact fork

Do **not** train two independent 320-epoch trajectories from scratch.  Train
one trajectory to epoch 280, clone its exact state, and fork:

```text
epochs 1–280:   ONE shared trajectory, Adam lr = 1e-3
epoch 281:      clone exact model state / optimizer state / RNG state /
                loader state; only then change the learning rate
CONTROL:        epochs 281–320 at lr = 1e-3
LOW-LR:         epochs 281–320 at lr = 1e-4
```

The first 280 epochs are *literally the same run*; they are not two
independently retrained runs with the same seed.

The fork epoch is frozen at **280** because the frozen CSSD-q1 Top-5 soup
members `282 / 299 / 307 / 312 / 313` all lie after epoch 280, i.e. the model
enters its best-checkpoint regime exactly around there.  Testing
`280 + 40` is therefore the trajectory-grounded choice; no `240+80`, `260+60`
or `300+20` alternative is permitted, and the switch epoch may not be changed
after seeing results.

Total compute: `280 + 40 + 40 = 360` epoch-equivalents (not 640).

The two continuations may run concurrently (`2 processes x 4 threads`) only if
batch order and thread count stay identical; otherwise run them sequentially.
Scientific integrity outranks wall clock; the default is sequential.

---

## 3. Optimizer contract

Shared prefix, exactly the frozen CSSD protocol:

```text
Adam
lr           = 1e-3
weight_decay = 1e-5
batch        = 128
grad_clip    = 5.0
loss         = L1 + H1_LAMBDA * reconstruction
```

At the fork the optimizer state is **carried over, never reset**:

* CONTROL: optimizer state unchanged, every `param_group["lr"]` stays `1e-3`.
* LOW-LR: optimizer state unchanged, every `param_group["lr"]` set to `1e-4`.

Forbidden: new Adam, reset `exp_avg`, reset `exp_avg_sq`, reset step counters.
The LR effect may not be confounded with an optimizer-reset effect.

---

## 4. Exact-fork integrity contract (frozen)

Before any optimizer step after the fork, with both arms loaded from the same
saved checkpoint, the round must record in `fork/fork_integrity.json`:

```text
epoch = 280
model_state_hash_control        == model_state_hash_low_lr
optimizer_state_hash_control    == optimizer_state_hash_low_lr   (before LR edit)
exp_avg / exp_avg_sq / step     identical parameter by parameter
prediction_max_abs_diff                == 0.0   (eval mode, deterministic)
train_mode_prediction_max_abs_diff     == 0.0   (dropout path, identical RNG state)
torch_rng_fingerprint_control   == torch_rng_fingerprint_low_lr
loader_generator_fingerprint_*  == same
next_epoch_batch_order_hash     == same
after LR edit: only the `lr` field differs in the optimizer state
```

If any of these fail, the primary comparison is invalid and the round stops
with `FORK_INTEGRITY_FAILURE`.

Hard test also exercised by the focused test-suite: immediately after the fork
and before any optimizer step,

```text
max |prediction_control - prediction_low_lr| == 0
```

including a train-mode (dropout-active) forward from the identical global RNG
state.

---

## 5. Batch-order contract

Both continuations must see exactly the same minibatch ordering for epochs
281–320.  Implementation: the training DataLoader's generator state is captured
at the end of epoch 280 and restored in both continuations (the loader uses an
explicit `torch.Generator`, so the order is a deterministic function of that
state).  Every graph is tagged with a stable `gid`; the per-epoch
`batch_order_hash` is the SHA-256 of the concatenated `gid` sequence in batch
order, recorded per epoch.  Frozen requirement:

```text
batch_order_hash_control(e) == batch_order_hash_low_lr(e)   for all e in 281..320
```

If any epoch hash differs, the primary comparison is invalid.

---

## 6. Metrics and primary comparison

Primary (frozen, no post-hoc switching):

```text
M_C = Top-5 soup over epochs 1–320 of the CONTROL continuation
M_L = Top-5 soup over epochs 1–320 of the LOW-LR continuation
G_schedule = M_C - M_L          (positive => low-LR tail is better)
```

The Top-5 soup rule is exactly the frozen CSSD rule: rank all epochs 1–320 by
valid MAE, take the best five (ties by earlier epoch), average their state
dictionaries, and evaluate that averaged model on official valid.

Secondary (reported, never substituted for the primary):

```text
M_C^tail, M_L^tail = Top-5 soup restricted to epochs 281–320
best valid / best epoch (full 1–320)
last-10 mean valid (311–320), epoch-320 valid
train MAE at epoch 320, generalization gap (valid - train)
number of Top-5 soup members after epoch 280
```

Per-epoch record (both arms): train MAE, valid MAE, task loss, reconstruction
loss (full diagnostic and optimised term), total loss, learning rate, global
pre-clip gradient norm, global parameter-update norm, and gradient/update norms
for exactly five modules — `dictionary` (`D`), node semantic binding
(`W_A_S`), edge semantic binding (`W_E_S`), `pair_encoder`, `reader`.  No other
diagnostics are added.

Noise reference: the v2-measured CPU soup-level reproducibility floor `~6e-4`.
A `~4e-4` LR difference is not a training breakthrough.

---

## 7. Decision thresholds (frozen)

```text
Case A   G_schedule <  0.002   -> TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK
directional 0.002 <= G_schedule < 0.003
                               -> DIRECTIONAL_SMALL_TRAINING_EFFECT
Case B   G_schedule >= 0.003   -> LOW_LR_TAIL_MATERIALLY_SUPPORTED
Case C   G_schedule >= 0.005 or M_L <= 0.125
                               -> TRAINING_PROTOCOL_MAJOR_FACTOR
Case D   M_L <= 0.120          -> BASELINE_UNDEROPTIMIZATION_WAS_SUBSTANTIAL
G_schedule < 0                 -> LOW_LR_TAIL_HARMFUL (keep fixed lr=1e-3,
                                  no rescue, no extra run)
```

`0.003` is ~5x the measured soup noise floor.  Precedence of the reported
headline verdict: `D > C > B > directional > harmful > A`.

No learning-rate sweep: the only candidate rate is `1e-4` (validated as a warm
adaptation rate in v2; `1e-3` warm restart was unstable).  Forbidden by this
preregistration: `3e-4 / 5e-4 / 5e-5 / 1e-5`, cosine, OneCycle,
ReduceLROnPlateau, any scheduler, any second switch epoch.

No horizon extension: both arms stop at exactly epoch 320 even if the best
epoch is 320 and the late slope is negative.  Only
`horizon_may_still_be_binding` is recorded; any horizon study requires a new
preregistration.  This keeps the comparison `LR schedule` vs `more compute`
unconfounded.

No seed 1/2 in this round.  If a material effect is found, the next
architecture round may use this protocol directly.

---

## 8. Prefix failure gate (frozen)

Before forking, the shared prefix must pass all of:

```text
epoch-280 valid MAE  <=  0.14115870875114342 + 0.02  (= 0.16115870875114342)
all losses finite (no NaN / inf)
train MAE at epoch 280 bounded (< 2.0) and no divergence
dictionary finite, non-degenerate and moved from initialisation
```

Otherwise: STOP, do not fork, record `PREFIX_REGIME_MISMATCH` (or the specific
failure) and buy no continuation.  The prefix-vs-historical per-epoch curve
comparison (`mean |Δ|`, median, p90, epoch 40/120/240/280) is reported as
provenance / regime sanity only, never as a primary gate; small drift from CPU
thread scheduling is expected and does not gate.

---

## 9. Dictionary health (minimal set only)

Evaluated on official valid at epoch 280 (prefix), CONTROL epoch 320 and
LOW-LR epoch 320:

```text
active atoms, effective atoms (N_eff), top-5 usage, max activation rate,
residual reconstruction error, dictionary movement (vs initial and vs epoch 280)
```

Purpose: confirm the schedule change did not break the representation.  The
full reusable-structure audit is **not** repeated.

---

## 10. Results layout (frozen)

```text
tracks/ksvd/results/e2e_dictenv_training_protocol_audit_v1/

    preregistration_snapshot.json
    preflight.json

    shared_prefix/
        curve.csv
        epoch280_state.pt
        optimizer_epoch280.pt
        resume_checkpoint.pt
        prefix_summary.json

    fork/
        fork_integrity.json

    control_lr1e3/
        curve.csv
        soup.json
        result.json

    low_lr1e4/
        curve.csv
        soup.json
        result.json

    comparison.json
    analysis_tables.md
    REPORT.md
    DECISION.md
```

Every payload carries `official_test_loaded = false`.  `docs/luyin/luyin19.txt`
stays untouched.

---

## 11. Required final report (Q1–Q7)

The REPORT must answer, using only official-valid evidence:

```text
Q1  after a from-scratch run to epoch 280, which late phase is better,
    lr=1e-3 or lr=1e-4?
Q2  is the low-LR improvement materially above the ~6e-4 CPU soup floor?
Q3  can the historical 0.130028 -> 0.128723 warm improvement be reproduced
    inside a formal 320-epoch schedule?
Q4  is fixed lr=1e-3 a material contributor to the current plateau?
Q5  is the training issue minor polish, material, or major?
Q6  should the canonical training protocol be updated?
Q7  is there evidence that the 320-epoch horizon itself is still binding?
    (report only; do not run longer)
```

---

## 12. Budget (hard ceiling)

```text
shared prefix       280 epochs
CONTROL continuation 40 epochs
LOW-LR continuation  40 epochs
total               360 epoch-equivalents
```

No additional epochs, no seed 1, no LR sweep.  If the prefix gate fails or the
fork integrity fails, only the prefix (or nothing) is bought.

---

## 13. Next-step decision tree (frozen before results)

* Case A / directional (`G_schedule < 0.003`): close the
  training-protocol-as-primary-bottleneck hypothesis.  The next round is a new
  architecture study — a **Dictionary-Conditioned Semantic Operator**
  (`e_v = sum_k alpha_vk f_k(q_v)`, shared dictionary roles driving a
  transferable structure→semantics map), still with **no message passing and
  no environment update** — and not another optimizer/scheduler search.
* Case B/C (`G_schedule >= 0.003`): `280@1e-3 + 40@1e-4` becomes the canonical
  training protocol for all subsequent architecture experiments; no more small
  deltas against the old `320@1e-3` baseline.
* Case D (`M_L <= 0.120`): do not design a new architecture yet; first freeze
  the new canonical baseline and re-run minimal mechanism probes (dictionary
  still load-bearing, node/edge correspondence still used, relation still
  used).
