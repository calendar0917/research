# PROTOCOL — LCRC-W3 regularization diagnostic v1

Round: `lcrc-pcba-w3-regularization-diagnostic-v1`
Status: **frozen before the first formal training step.** The only permitted
later edits are append-only entries in §12 (incidents / deviations) and the
`PROTOCOL.md`-declared measurement slots that are *values* of rules frozen here
(§6.1 cadence, §7 audit results). No rule below is changed after seeing any
formal epoch's metrics.

## 1. The single question

> Historical LCRC-W3 on MolPCBA raised Train AP markedly (0.4197 → 0.7055 from
> A0) while Validation AP stalled (0.2619 → 0.2590 from W2). Is that at least
> partly because the regularization recipe was never adapted to the model size?

This round is a **fixed-design diagnostic**. It does not test the optimal
learning rate, the optimal parameter scale, or the performance ceiling of the
LCRC family. Two and only two new runs are executed.

## 2. Frozen model (historical LCRC-A-W3, unchanged)

`lcrc_pcba_backbone_scaling_v1/code/bs_model.py::build_model("W3", ...)` is
imported and used verbatim: the same `TULCRC(variant="C1", cfg=W3 widths)`
trunk (atom encoder, bond encoder, one-shot S1/S2 `TULocalContent`, `f`, `rho`),
the same `cont_mean/cont_scale` buffers, the same four-head
`AttentionPoolReadout`, the same seed-0 initialisation (FULL→C1 remap, private
readout generator `seed*1_000_003+17`, `readout.fc2` zeroed).

| item | value |
|---|---|
| parameters (`sum(p.numel())`) | **2,120,844** |
| trunk | 1,809,408 |
| attention readout | 311,436 |
| widths | atom 384/288, bond 480/288, stat_rank 96, local 576, width 192, rho 192, scorer 96, pool 264, heads 4, out 128 |
| iterative message passing | none |
| structural features / output head | unchanged |

No module is added, removed or re-defined. No architecture search.

## 3. The two new runs (the entire experiment)

| Run ID | Dropout | AdamW weight_decay |
|---|---:|---:|
| `W3_WD10` | 0.10 | **0.010** |
| `W3_DO20` | **0.20** | 0.001 |

Everything else is the historical W3 recipe, identical in both runs:

```
seed = 0                      epochs = 40            batch_size = 128
optimizer = AdamW             betas = (0.9, 0.999)   eps = 1e-8
peak_lr = 3e-4                warmup = 1 epoch linear, start 3e-5
scheduler = per-step cosine    final_lr = 1e-5
gradient_clip_norm = 5.0      precision = FP32
loss = original observed-entry masked BCE (NaN targets excluded)
early_stopping = false        init = historical W3 seed-0 rule
batch order = frozen historical seed-0 40-epoch plan (global_plan_sha256 06f43729…b6f6)
```

Explicitly **not** run: W2, historical W3, A0, CPC, RHIC, DYNAMIC retraining; no
WD+dropout joint arm; no third hyper-parameter value; no additional seed.

Each run has its own output directory (`runs/W3_WD10`, `runs/W3_DO20`); the
shared feature cache is opened read-only.

## 4. Evaluation: all three splits, every epoch

After the complete epoch of training, in this order, for every epoch 1…40:

1. `model.eval()` and `torch.inference_mode()`;
2. full **Train** split (350,343 rows) inference;
3. full **Validation** split (43,793 rows) inference;
4. full **Test** split (43,793 rows) inference;
5. per split record: official OGB macro AP, masked BCE, per-task AP, per-task
   observed/positive/negative counts, scored-task count, inference and metric
   seconds, logits/probs SHA-256;
6. verify that the evaluation pass did not change model parameters, optimizer
   state or training RNG state (hashes before/after), did not create new
   gradients and did not call the optimizer;
7. `model.train()` again, continue.

No minibatch training loss substitutes a Train AP; Train is never subsampled;
Test is never skipped. All 40 epochs carry all three splits.

Per-epoch logits are **not** persisted: metrics are computed per split and the
array is freed before the next split.

### 4.1 Checkpoint selection

Primary checkpoint = **maximum Validation AP, ties resolved to the earlier
epoch** (`strictly greater` comparison upward). Test is never used for
selection.

### 4.2 Test exposure (explicit)

Official Test is **evaluated every epoch** in this round by design and is
labelled `EXPOSED_TEST_EXPLORATORY`. Test curves may be plotted and discussed.
The checkpoint and every "best" claim remain Validation-based. Historical Test
values used for comparison are read-only *engineering* comparisons; they are
not a blind estimate and are never a selection signal.

## 5. Test labels: official row ids, isolated from training

1. The frozen training artifact `.../lcrc_pcba_scale_pilot_v1/analysis/pcba_labels_trainvalid.npy`
   is **not** modified; its Test rows stay NaN and its hash is re-checked.
2. Test labels are read independently from `data/ogb/ogbg_molpcba/raw/graph-label.csv.gz`
   (chunked read into a preallocated `(437929, 128)` float32 array; no giant
   Python float list).
3. Shape must be `(437929, 128)`; official split row ids are used for slicing.
4. Every Train and Validation entry of the raw file must equal the frozen
   artifact **including the NaN mask**; otherwise the round stops.
5. Test non-missing entries must be exactly 0 or 1.
6. Test labels may reach only the metric functions. They are not passed to the
   model, loss, optimizer, scheduler, batch order or checkpoint selection. A
   static call-site audit (`rd_audit.py --modes test-isolation`) and a runtime
   assertion that the training batch tensor is built from `task.labels` only
   enforce this.

## 6. Pre-training gates (all must pass on CPU and on the GPU host)

`rd_audit.py` writes `analysis/audit.json`, `analysis/alignment.json`,
`analysis/gpu_preflight.json`, `analysis/evaluation_cost_probe.json`.

**Model / experiment variables**

1. both models have exactly **2,120,844** parameters, trunk 1,809,408, readout
   311,436;
2. both initial `state_dict`s are identical (bitwise) and their eval-mode
   initial logits are identical;
3. the two runs differ in exactly one of `{dropout, weight_decay}` each, i.e.
   the pair of vectors is `(0.10, 1e-2)` vs `(0.20, 1e-3)` and nothing else;
4. `W3_WD10`'s AdamW parameter groups all carry `weight_decay = 0.010`,
   `W3_DO20`'s carry `0.001` (read back from the built optimizer);
5. `W3_DO20`'s real dropout module has `p = 0.20` and `W3_WD10`'s `p = 0.10`,
   verified on the **actual forward path** (hooked zero-fraction of the `h`
   tensor in train mode, and absence of zeros in eval mode), not on config text;
6. on the same Train batch and the same initial parameters, the intervention
   measurably changes the training forward (different dropout masks/zero
   fraction) and the optimizer update direction (different `weight_decay`
   multiplier), while eval-mode logits stay identical;
7. both runs' training-order row-id hashes, step count and `global_plan_sha256`
   equal the historical W3 plan; `consumed` prefix hashes match.

**Data / metrics**

8. official Train/Valid/Test row ids are pairwise disjoint, cover the official
   split sizes and are the frozen split;
9. cache, feature spec, Train-only normalization, graph cache and collate are
   the historical ones (hash/fingerprint equality, not a re-implementation);
10. Test labels are correctly aligned (raw file vs frozen artifact on
    Train/Valid, NaN mask included) and never enter training;
11. task 45 (`zero Validation positives`) is excluded from the macro AP by the
    official evaluator, not counted as 0;
12. each split's evaluation covers every row exactly once (batch offset
    bookkeeping asserted);
13. `eval()` inference consumes no training RNG, calls no optimizer step and
    leaves `.grad` untouched;
14. the official OGB macro AP is cross-checked against an independent per-task
    `sklearn.average_precision_score` recomputation on every split (|Δ| ≤ 1e-9);
15. saved state is resumable: `checkpoint_last.pt` restores model, AdamW state,
    RNG and plan, does not lose completed evaluation rows, and does not retrain
    a completed epoch (tested on a 2-epoch CPU/GPU mini-run).

A small **real** training smoke test (≥ 10 optimizer steps on the real
collate/stager/model/loss path plus one full three-split evaluation) and a
three-split evaluation cost probe are run on the GPU host before the formal
runs. The smoke test never becomes the head of a formal run: both formal runs
restart from the frozen initial state in fresh directories.

If a gate fails, the implementation is fixed; the formal runs never start from a
state that was repaired mid-training.

## 7. Audit results (measurements, filled before the formal runs)

See `analysis/audit.json`, `analysis/alignment.json`,
`analysis/gpu_preflight.json`, `analysis/evaluation_cost_probe.json`.

### 7.1 Evaluation-cadence decision

The protocol **requires** three-split evaluation every epoch for all 40 epochs;
there is no cadence rule to resolve. The cost probe only records the projected
wall clock and confirms the budget in §8 is feasible.

## 8. Budget and resources

```
target <= 10 A100 GPU-hours   hard cap <= 16 A100 GPU-hours
```

Preference: `W3_WD10 → res-gpu1`, `W3_DO20 → res-gpu0`, launched in parallel.
GPU0 is used only after a live occupancy check (free memory with margin, no
sustained full-load contention from an external job). Otherwise both runs go
serially on `res-gpu1`. External processes are never signalled, paused or
modified. Per run record: rr run id, pool, `CUDA_VISIBLE_DEVICES`, GPU name,
concurrent state, start/end UTC, train/inference seconds per split, throughput,
peak GPU memory, GPU-hours, retries.

All GPU work goes through `rr` (`doctor` / `deploy` / `run` / `pull`). Local CPU
is used only for pytest, data/alignment/gate checks, metric re-computation and
reporting.

A failed run is first resumed from `checkpoint_last.pt`; a completed 40-epoch
run is never rerun. A run is only stopped early if the resource cap, an
infrastructure failure or result integrity demands it (recorded in §12).

## 9. Comparisons (read-only history, no retraining)

Frozen history from `lcrc_pcba_backbone_scaling_v1/REPORT.md`:

| model | Train AP@40 | Valid AP@40 | Best Valid AP (epoch) |
|---|---:|---:|---:|
| W2 | 0.581875 | 0.261867 | 0.265273 (24) |
| W3 | 0.705498 | 0.259004 | 0.263670 (25) |

Reported for each new run: `Best Valid − 0.263670`, `Valid@40 − 0.259004`, the
same two differences against W2, Train/Valid/Test AP at the best-Valid epoch,
Train/Valid/Test AP at epoch 40, three-split AP at epochs 20/25/30/35/40, the
20–40 Validation/Test slope, per-task deltas by support stratum (Valid positives
≥ 20 / 5–19 / < 5), and the Train−Valid gap computed (a) on each split's own
scorable-task macro AP and (b) restricted to the tasks scorable in both.

If the historical W3 best-Valid and epoch-40 checkpoints are present, **one**
read-only Test inference pass each is allowed to fill the missing historical
Test AP. This is not a baseline training run. If the checkpoints are absent the
field is reported as missing.

## 10. Decision labels (resource/direction thresholds, not significance)

Applied to each run against historical W3 (`Δbest = BestValid − 0.263670`,
`Δ@40 = Valid@40 − 0.259004`):

| label | rule |
|---|---|
| `PROMISING_SINGLE_SEED_SIGNAL` | `Δbest ≥ +0.005` **and** `Δ@40 ≥ +0.003` **and** the gain is not driven by a few low-support tasks **and** Test at the best-Valid checkpoint shows no clear opposite regression |
| `FIT_SUPPRESSION_WITHOUT_GENERALIZATION_GAIN` | Train AP clearly lower than W3's 0.7055 while Valid AP is unchanged or lower |
| `PEAK_ONLY_OR_UNSTABLE_SIGNAL` | `Δbest > 0` but `Δ@40 ≤ 0`, or the peak is confined to a narrow epoch range (reported explicitly) |
| `NO_CLEAR_GAIN` / `MIXED_SIGNAL` | small Validation differences, offsetting per-task gains, or clearly opposite Valid/Test movement |

"Not driven by a few low-support tasks" = the ≥20-positive Validation stratum
improves (mean Δ > 0) and improved tasks in that stratum are a majority.
The labels are single-seed direction indicators; no significance claim is made.

## 11. Deliverables

```
PROTOCOL.md README.md REPORT.md
code/rd_{common,data,model,train,audit,compare}.py
tests/test_rd_{model,data,train}.py
analysis/{audit,alignment,gpu_preflight,evaluation_cost_probe,comparison,decision,
          historical_test}.json
analysis/comparison.csv  analysis/curves_*.csv  analysis/figures/*.png
runs/W3_WD10/  runs/W3_DO20/        (git-ignored; pulled)
```

Per run directory: `epoch_metrics.csv` (epoch × 3 splits), `per_task_ap.csv`
(epoch × 3 splits × 128 tasks), `epoch_log.csv` (training-side per-epoch),
`checkpoint_best_valid.pt`, `checkpoint_epoch40.pt`, `checkpoint_last.pt`,
`rng_state.pt`, `training_order.json`, `run.json`. Checkpoints are large and stay
out of Git; their SHA-256 and paths live in `run.json` and the report.

## 12. Amendments (append-only)

| # | date | change | reason |
|---|---|---|---|
| 1 | 2026-11-10 (before any formal step) | §10 operationalised with pinned numbers used by `rd_compare.decide`: `d_best = BestValid − 0.26367025868637306`, `d_ap40 = Valid@40 − 0.259003625836404`, `d_test = Test@best − 0.25182189501469937` (historical W3 Test at its best epoch, read-only), `d_train40 = Train@40 − 0.7054983915679482`; `broad` = mean Δ > 0 and a majority of improved tasks on the ≥ 20-positive Validation stratum; `longest_above` = longest run of epochs with Valid AP > 0.26367025868637306. Label order: `PROMISING` → `PEAK_ONLY_OR_UNSTABLE` (`d_best ≥ +0.002` and (`d_ap40 ≤ 0` or `longest_above < 5`)) → `FIT_SUPPRESSION` (`d_train40 ≤ −0.05` and `d_ap40 < +0.002`) → `NO_CLEAR_GAIN` (`\|d_best\| < 0.002` and `\|d_ap40\| < 0.002`) → `MIXED_SIGNAL` | The qualitative §10 rules needed fixed comparison numbers and a deterministic priority order; pinned **before** any formal epoch exists. No threshold value is changed. |
| 2 | 2026-11-10 (before any formal step) | §8 applied: both runs serial on `res-gpu1`; projected 186 s/epoch => 2.07 GPU-h per run, 4.14 GPU-h total | GPU0 was at 100 % utilisation in 6/6 samples over 60 s from an external 2-day-old job, so the co-tenancy clause fails; see `analysis/gpu_preflight.json` and `analysis/gpu_occupancy_samples.txt`. GPU0 is re-checked before the second run starts. This is the rule acting, not a change to it. |
