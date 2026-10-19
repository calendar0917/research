# ZINC GRIT reference: cost report

**Verdict.** The reference is credible for *cost and pipeline preparation*, not
for a paper-score reproduction or a pure architecture comparison with P3.  The
model is exactly the pinned GRIT ZINC model (473,473 parameters), the official
config is byte-identical, the official test split is blocked at construction,
and the functional checks pass.  The remaining caveats are the compatibility
shims (no compiled `torch_scatter`/`torch_sparse`), the PyG 2.6.1 stack, and the
fact that no official valid/test score was measured.

## 1. What was run

| item | value |
|---|---|
| host / GPU | `res` = `a100-2`, 1 x NVIDIA A100-SXM4-40GB, GPU 0 |
| driver / CUDA | 550.163.01 / CUDA 12.4 |
| torch / PyG / Python | 2.5.1+cu124 / 2.6.1 / 3.12.14 |
| upstream | `LiamMa/GRIT` @ `6c988ea600a606fbb49a2246c64a2d37396b3ab5` |
| config | official `configs/GRIT/zinc-GRIT-RRWP.yaml` |
| config SHA-256 | `e60b425badb9ff29de9bb8a973bfa73b9d9d4d97cf6d2ae6cc1a8be77edc00bf` |
| compat patch SHA-256 | `dae522601463ae2c56783b3a0174c64f09f9efb0e1f3af388d52150ef9e5a283` |
| seed / batch / precision | 41 / 32 / FP32 |
| schedule | official 2000-epoch cosine + 50-epoch warmup; timing stopped after 4 epochs |
| data | official PyG ZINC subset, **train only**: 10,000 graphs |
| optimizer steps | 313 per epoch (last batch 16) |
| params | 473,473 (paper ZINC value, difference 0) |

The parameter count was rebuilt from the official width (64), 10 layers, 8
heads, type-dictionary encoders, and the 21-step RRWP encoder.  It matches the
paper exactly, so there is no width/embedding discrepancy to explain; no width
was adjusted to force the match.

The timing run executed one warmup epoch plus three timed epochs, then a
no-grad forward over the first 1,000 *train* graphs as an eval-cost proxy.  No
official valid or test loader was created.  The 1,000-row pass computed no
loss or metric.  The executed timing revision is recorded in `timing.json`
(`research_git_commit=0889f45...`).  The delivered bundle adds one
post-measurement loader guard: `make_loaders()` refuses to call
`create_loader()` unless `reference.splits` is explicit.  The measured run
already set `reference.splits=train`, so the guard does not change any measured
training step.

## 2. Functional credibility checks

All checks passed on both CPU and the A100.

| check | result |
|---|---|
| parameter count | 473,473 = paper's 473,473 (delta 0) |
| RRWP tensors | node `[N,21]`, pair index `[2,N^2]`, pair values `[N^2,21]` |
| RRWP vs independent dense implementation | max abs difference `0.0` |
| RRWP participation | zeroing pair values changes prediction by `0.0220`; zeroing node abs changes by `0.2307` |
| one forward/backward | finite loss; 236/236 grad tensors finite and nonzero |
| eval renumbering | max abs error `7.15e-7` on GPU (`4.17e-7` CPU), tolerance `5e-6` |
| test construction guard | `splits='test'` raises before dataset creation |
| loaders / splits | 1 loader, `num_splits=1`, no `test*` attribute in dataset |

The data audit compared all 10,000 official train graphs with the research
track's `objects_train10k.pt` and `encoded_train.pt`: atom types, undirected
bond types, and labels match exactly (`max_abs_label_diff = 0.0`).  Train
atom types are `0..20`, bond types `{1,2,3}`, labels mean `0.0153`, std
`2.0108`, range `[-42.04, 3.80]`.

## 3. Measured epoch time and memory

Two independent successful runs were made.  The first was repeated only to fix
a provenance field that counted unrelated untracked experiment lock files as
"dirty"; both runs exited 0.  The table uses the repeated run (v2); the first
run is kept in `timing.json` as a repeat sample.

| epoch | train loop (s) | write (s) | total (s) | LR | peak allocated (MB) | peak reserved (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 warmup | 34.928 | 0.016 | 34.944 | 2e-5 | 684.4 | 892.0 |
| 1 timed | 35.593 | 0.012 | 35.606 | 4e-5 | 741.5 | 922.0 |
| 2 timed | 35.578 | 0.012 | 35.591 | 6e-5 | 677.9 | 922.0 |
| 3 timed | 34.994 | 0.018 | 35.012 | 8e-5 | 698.1 | 922.0 |

- Timed epoch mean: **35.40 s**; min 35.01 s, max 35.61 s.
- Repeat run mean: **35.08 s** (min 35.03, max 35.14).
- Combined train-only estimate: **35.24 s/epoch**.
- Peak allocated: **741.5 MB**; peak reserved: **922.0 MB** (< 2.5% of 40 GB).
- Setup: 8.6 s; dataset/RRWP build: 5.8 s; model build: 0.31 s.
- 1,000-train-row eval proxy: **1.17 s** (v2; 1.12 s in v1), 32 batches,
  finite outputs, no score.

The paper's Table 12 reports 29.4 s/epoch on V100 (about 16.3 V100-h for 2,000
epochs).  Our A100 train-only epoch is ~35.2 s, i.e. **not faster** than that
V100 number.  This is plausible for 23-node graphs and 313 Python/PyG steps per
epoch: the per-step work is tiny, so launch/collation overhead can dominate.
No SM-utilization trace was captured, so this is an inference from the graph
size, step count, and <2.5% memory footprint, not a measured utilization number.
It is not an apples-to-apples benchmark: the paper's original stack used compiled
`torch_scatter`/`torch_sparse`, while this reproducible stack uses PyG 2.6.1
shims.  Do not use the V100 number as this reference's cost.

## 4. Actual GPU budget used

Accounting uses `rr` meta `started_at -> completed_at`, one GPU per run, which
includes shell/entry setup (159 s each) rather than only the Python process
(150.9 s in v2).

| run | run id | wall (s) | GPU-hours |
|---|---|---:|---:|
| repeat / v1 | `zinc-grit-ref-timing-20261019-211031-c351fef8` | 159 | 0.0442 |
| primary / v2 | `zinc-grit-ref-timing-v2-20261019-211711-39097194` | 159 | 0.0442 |
| **total** | | **318** | **0.0883** |

Cap was 0.15 A100-hours; actual occupancy **0.0883 h (5.3 min)**, remaining
**0.0617 h (3.7 min)**.  There were no failed GPU runs.  GPU 1 stayed idle; the
job used `CUDA_VISIBLE_DEVICES=0`.

Tracked tree was clean (`rr` meta `git_dirty=false`).  The remote checkout had
9 unrelated untracked lock/backup files from frozen experiments; they were not
modified by this run.  The run's own untracked artifact is only the pulled
result directory.

## 5. Projection to 2,000 epochs

Formal mode adds a 1,000-graph valid eval every epoch and best-valid
checkpointing.  Using the measured forward proxy plus logging/checkpoint
overhead:

| component | low | central | high |
|---|---:|---:|---:|
| train-only epoch | 35.01 s | 35.24 s | 35.61 s |
| valid eval + log/ckpt overhead | 1.17 s | 1.25 s | 1.40 s |
| formal epoch estimate | **36.2 s** | **36.5 s** | **37.0 s** |

One seed, 2,000 epochs:

- Train-only (no valid eval): ~19.6 A100-h.
- Formal literature recipe: **20.1–20.6 A100-h**; planning number **20.5**.
- Add one-time data/model setup (~0.005 h, negligible).

Two seeds:

- GPU-hours = sum of per-GPU times: **40.2–41.2 A100-h**; planning number
  **41**.
- Two A100s running one seed each in parallel: wall clock **20.1–20.6 h**
  (assuming both start together; no queue/contention).  This is not 20.3
  GPU-hours; the GPU-hours remain the sum, ~40.5.
- One A100 running the two seeds sequentially: wall clock **40.2–41.2 h**.

Uncertainty: validation cost is proxied from train rows; checkpoint I/O was not
instrumented; one host and one timing sample; thermal/clock and other-tenant
load can move epoch time by several percent.  A 21 h / 42 h planning envelope
is safer than the point estimates.

## 6. How to interpret against P3

Do **not** read a 2,000-epoch + best-valid GRIT number as a pure architecture
gain over P3.  At least these differ:

| factor | GRIT reference | P3 recipe |
|---|---|---|
| parameters | 473,473 | 413,505 (-12.7%, delta 59,968) |
| training budget | 2,000 epochs | 360 epochs |
| checkpoint selection | best official-valid epoch | last-5 soup |
| inputs | official atom/bond types + RRWP only | P3 descriptors / SSL / extra features |
| selection data | official valid (no test) | no best-valid selection in the original recipe |

A fair comparison needs a matched protocol, for example:

1. **Budget-controlled**: GRIT at 360 epochs with the same last-5 soup (no
   best-valid), against P3 at 360 epochs/last-5.
2. **Recipe-controlled**: give P3 2,000 epochs and best-valid checkpointing,
   then compare with GRIT's literature recipe.
3. **Feature-controlled**: keep the same inputs and budget, and vary only the
   architecture.

The paper's `0.059 ± 0.002` is an official **test MAE**.  This task never reads
test and makes no reproduction claim.  If a terminal test read is later
authorized, report it separately from valid-based checkpoint selection.

## 7. Readiness and recommended next step

**Ready to run** the formal 2,000-epoch reference for one or two seeds.  The
entry point is `run_reference.py --mode train --confirm-long-run`; it constructs
train+valid (never test), evaluates valid every epoch, and keeps the best-valid
checkpoint.  It is deliberately not auto-started.

Recommended minimum: two seeds (41, 42) on the two idle A100s, ~20.5 h wall and
~41 A100-hours total.  Use one seed only for a cost/plumbing check, not for a
stable comparison.  Freeze the protocol and the test-read policy before looking
at any valid score.

## 8. Failures and unresolved issues

- No GPU/training failure; both timing runs exited 0.  The first run was
  repeated for metadata correctness, not because of a numerical failure.
- Official valid/test scores were not measured.  The 1,000-row eval is a
  train-row timing proxy, not validation.
- No 2,000-epoch run was started; projections include the stated uncertainty.
- The compatibility shims replace compiled extensions.  RRWP is checked
  against a dense reference, but the shim is not a bit-for-bit replacement for
  every possible upstream op.
- The PyG 2.6.1 / torch 2.5.1 stack is not the paper's original stack, so epoch
  time (and possibly low-level numerics) can differ.
- Checkpoint I/O for the formal best-valid run was not directly measured; the
  projection includes a small overhead allowance.
- The remote checkout contains unrelated untracked experiment locks; tracked
  files were clean.  They are left untouched.
- `torchmetrics`, `wandb`, `mlflow`, and `tensorboardX` are not used.  The
  logger change only makes classification-metric imports optional; the
  regression L1/MAE path is unchanged.
