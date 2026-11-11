# PROTOCOL — PCQM4Mv2 training efficiency + LCRC width capacity (v1)

Round id: `lcrc-pcqm4mv2-efficiency-width-v1`
Status: **frozen before the first profiling step and before any formal training
step**. §12 is an append-only amendment log; nothing else may change after the
first formal step.

Two independent questions, fixed by the execution instruction:

* **Q1 (engineering efficiency).** With W2, batch 128, FP32, AdamW, the frozen
  training order, loss and evaluation protocol unchanged, can real end-to-end
  training throughput be increased materially?
* **Q2 (model capacity).** On the same 1,000,000 PCQM4Mv2 Train molecules and
  the same 40-epoch recipe, does the LCRC parameter count change Validation
  MAE?

The historical W2 run is a **read-only control** and is never retrained:
939,079 parameters, 40 epochs, seed 0, 1M Train molecules, best Validation MAE
0.132310904 eV at epoch 40, Train-monitor MAE 0.105891 eV, 5.162 A100
GPU-hours, code `93ae701f`.

## 1. Inherited, read-only ingredients

| item | source | reused as |
|---|---|---|
| `pcqm_train1m` cache (1,000,000 rows, 5.5 GiB) | `lcrc_pcqm4mv2_transfer_pilot_v1/cache` | read-only mmap, hash-verified at every run start |
| `pcqm_valid` cache (73,545 rows) | same | read-only mmap |
| fixed subset rows / label artifact / monitor rows | same | unchanged |
| feature + label normalisation (`lipo_v1`, Train-300K constants) | same | unchanged |
| `qr` official `PCQM4Mv2Evaluator` | `pq_data._evaluator` | unchanged |
| model graph, init rule, recipe, batch order | `bs_model`, `pq_model`, `pq_common`, `pq_data` | unchanged |

No data is re-downloaded, no cache is rebuilt, no existing file is modified.
The new round imports the frozen parent modules; it never writes into the
parent directory.

## 2. Q1 — stage 1: measured bottleneck analysis

Host: `res` (2× A100-SXM4-40GB), pool `res-gpu1`, real 1M cache, historical W2
model (939,079 parameters), real batch-128 training steps.

* ≥ 400 measured steps after 50 warm-up steps, for a cold-cache pass and a
  hot-cache pass over the same rows (cold = `posix_fadvise(DONTNEED)` on the
  cache files, verified with `mincore`; hot = replay of the same rows).
* Separate instrumentation for: CPU `collate`; pinned staging and H2D; forward;
  loss; backward; clip; AdamW update; explicit `torch.cuda.synchronize`;
  `.item()` / `float()` / Python-bool synchronisation; per-epoch logging,
  checkpoint and full Validation/monitor evaluation (amortised per step).
* `torch.profiler` (CPU + CUDA activities) for a 30-step window, CUDA events
  for phase spans, and process wall clock for end-to-end throughput. CUDA
  asynchronous execution is never accounted as host time: host spans and device
  spans are reported separately.
* Also recorded: CPU utilisation (cores and % of host), RSS, GPU utilisation,
  peak GPU memory, step-time p50/p90, real molecules/sec.
* The instrumented step is checked against the frozen step (loss, gradient
  norm, post-step weights and AdamW state) before any measurement is accepted.

Specific suspects that are measured, not assumed: the Python loops in
`lcrc_data._ranges`, the per-atom `nbr_idx` / `path_idx` arrays, the ~21 tensor
transfers per batch, per-step `float(loss)` / gradient-norm reads /
`torch.cuda.synchronize()`, and the possibility of overlapping CPU graph
assembly with GPU training.

## 3. Q1 — stage 2: implementations and correctness gates

Three optimisations, each measured alone and combined:

* **E1 — equivalent vectorised collate.** Replace per-molecule/per-atom
  `np.arange` concatenation with `repeat`/`cumsum` index algebra. No field,
  dtype, shape or value changes.
* **E2 — fewer host synchronisations.** Per-step loss, gradient-norm and
  finiteness reads move to a fixed block (64 steps); the checks stay, the
  device-side flags are read back once per block and a non-finite value is
  reported with its exact step. Gradient clipping and AdamW are unchanged.
* **E3 — bounded async prefetch.** One producer thread per epoch with a
  bounded queue (depth 3) collates the next batches while the GPU works; the
  shared read-only mmap cache is not copied and no graph object is duplicated
  across processes.

Measured configurations: `frozen`, `E1`, `E2`, `E3`, `E1E2`, `E1E2E3`. Each
starts from the same seed-0 state, consumes the same row order and takes the
same number of steps, so the final weights **must be identical** across
configurations (hash gate). Batch sizes 256 and 512 are probed for throughput
investigation only; the formal runs stay at batch 128.

Correctness gates before any implementation may be used for a formal run:

1. same row-id batch ⇒ every input tensor equal field-by-field (value, dtype,
   shape) to the frozen collation;
2. forward result equal (declared tolerance 1e-6, expected exact because the
   inputs are identical);
3. per-step loss and gradient norm equal, and post-step weights/AdamW state
   bitwise equal, over ≥ 200 real training steps;
4. random seed, initialisation, batch order, loss, clip and optimizer state
   unchanged; RNG state after the window equal;
5. fixed-sample predictions equal (≤ 1e-6);
6. checkpoint save → restore → continue reproduces the uninterrupted run
   bitwise;
7. atom renumbering leaves the graph-level prediction unchanged (≤ 1e-4);
8. official MAE == independent NumPy MAE on the full Validation split
   (≤ 1e-9).

Selection rule: adopt the **fastest correct and stable batch-128
implementation**. If the end-to-end gain is below 15 %, the simple
implementation is kept and no complex machinery is forced in.

Budget for stage 1 + stage 2 (profiling, benchmarks, gates): **≤ 0.75 A100
GPU-hour**. When that budget is reached, optimisation exploration stops and the
best verified implementation is frozen.

## 4. Q2 — stage 3: fixed width experiment

One final training implementation, three width points; W2 is the historical
read-only control.

| run | width scale | parameters (analytic, verified at build) | data | epochs |
|---|---:|---:|---|---:|
| `PCQM_W1_1M` | 1× | 259,831 | 1M frozen Train subset | 40 |
| `PCQM_W2_1M` (historical) | 2× | 939,079 | same 1M | 40 |
| `PCQM_W3_1M` | 3× | 2,038,167 | same 1M | 40 |

Frozen for both new runs: same cache and normalisation, 2D SMILES only, seed 0,
batch 128, 40 epochs, AdamW (betas 0.9/0.999, eps 1e-8), peak LR 3e-4, weight
decay 1e-3, dropout 0.10, 1-epoch linear warmup 3e-5 → 3e-4, per-step cosine to
1e-5 over the 40-epoch horizon of 312,520 steps, L1 on the standardised
HOMO–LUMO gap, gradient clip norm 5.0, FP32, fresh random initialisation, no
early stopping, full official Validation MAE plus the same 20,000-row
Train-monitor MAE after every epoch.

Model: the inherited LCRC graph (atom encoder, bond encoder, one-shot S1/S2
local content, per-atom contribution `f`, global `rho`, four-head attention
readout, one regression output) with `bs_common.config_for_scale(scale)` and
`n_out = 1`; identical init rule for all widths. `ew_model.build_model(scale=2)`
must be bitwise identical to `pq_model.build_model` (gate).

Both runs go from epoch 1 through epoch 40 with no mid-run stopping. The
40-epoch LR horizon is never shortened to a trial horizon.

## 5. Budget and run order

* Whole new round: **12 A100 GPU-hours**, including profiling, preflight and
  both formal runs.
* Before the formal runs, the measured throughput of the optimised
  implementation on the real cache must project both runs inside the remaining
  budget. Order: **W3 first**, then W1.
* If only one complete run fits: finish W3, stop, and state explicitly that W1
  is missing; an unfinished run is never presented as a capacity comparison.
* GPU1 preferred; GPU0 only after a live check shows it idle. External jobs are
  never touched. Sequential execution is used so the speed comparison is not
  contaminated by co-tenancy.
* Epochs, validation frequency, batch size, data and precision are never reduced
  to fit the budget.
* The `rr` execution-regime provenance (host, GPU, driver, torch, commit) is
  recorded for every run.

## 6. Reporting and decision rules

`REPORT.md` first page contains the required table (parameters, best Validation
MAE and epoch, Validation@40, monitor@40, GPU-hours, molecules/sec) for W1,
historical W2 and W3, plus the speed comparison, profiling hotspots,
correctness gates, curves, epoch 20/30/35/40 and last-5-epoch means,
monitor-vs-Validation gap, parameter/width vs speed/MAE, and the updated
3,378,606 × 40-epoch projection (marked as projection, never as a measurement).

Width decision rule (resource allocation only, not a significance test): if
Validation MAE @40 improves by at least **0.005 eV** versus W2 and the late
trend is consistent, label `PROMISING_WIDTH_SIGNAL`; a smaller change keeps the
exact number and claims no stable advantage. With one seed per arm no
confidence interval is claimed.

Supplementary read-only diagnostics on the W2 best checkpoint, never used to
select a model: exact-SMILES overlap between Validation and the official Train
split, MAE in the duplicate vs non-duplicate group, and absolute error by atom
count, target-gap bin and UNK-element presence. The official metric is never
recomputed after dropping duplicates.

## 7. Out of scope

No full 3,378,606-molecule run, no architecture search, no hyper-parameter
search, no pretraining, no extra seeds, no AMP/TF32, no batch-size change of
the formal recipe, no CUDA graphs, no compiler caches, no model fusion, no
early stopping, no Test-split evaluation or submission, no further width arm.

## 8. Deliverables

`PROTOCOL.md`, `REPORT.md`, optimised code, tests, profiling JSON, benchmark
JSON, correctness-gate JSON, three-model comparison, resource ledger, run
commands, code commit, data hashes and checkpoint hashes. Large binaries
(caches, checkpoints) stay git-ignored with their real paths and SHA256
recorded.

## 9. Amendments (append-only)

- 2026-11-11 — protocol written before the first profiling step. Questions,
  implementations, gates, budget, run order and decision rule come from the
  execution instruction. No model step had been taken at this commit.
- 2026-11-11 — stage-1 profiling completed (400 measured steps after 50
  warm-up, cold and hot pass, commit `67256fb49e6f`; JSON:
  `analysis/profile.json`). Hot-cache step-only time 33.77 ms (p50 32.80,
  p90 41.54), end-to-end 2857.8 molecules/s; cold-cache 44.72 ms / 718.9
  molecules/s. Host time is dominated by collate (hot 9.48 ms/batch, cold
  131 ms) and kernel launches (36,462 launches over 30 steps; `cudaLaunchKernel`
  391 s of the 1046 s profiler CPU window), not by GPU arithmetic. Explicit
  synchronisations cost 0.03–0.06 ms/step; full Validation + monitor +
  checkpoint evaluation costs 0.76 ms/step amortised (1.6–2.4 %). The process
  uses ~8 cores on average and the GPU is the idle partner in the gap-free
  stretches. These numbers set the stage-2 floor and the decision to prefer a
  CPU-side fix.
- 2026-11-11 — stage-2 implementation selection (before any formal capacity
  run). A first single-pass benchmark was discarded as uninterpretable because
  the same frozen configuration measured 36.4 ms in one window and 44.1 ms at
  the end of the same session (shared-host drift). The retained benchmark
  (`analysis/bench.json`) pre-touches every row, runs 11 configurations in
  three rounds with alternating order, and reports medians: frozen 2759 mol/s;
  **E1 3626 mol/s (1.31×)**; E1E3 3262; E1E2E3c16 3399; E1E2 2998; E1E2c8
  2898; E1E2c16 2710; E2 2326; E3 1765; `frozen_t1` 2307; `E1_t1` 2334. All
  configurations reached bitwise-identical weights. E2 (deferred sync) and E3
  (producer thread) were *consistently slower* than the frozen path, and a
  one-thread diagnostic also lost throughput; the block-flush interval sweep
  (64/16/8) did not rescue E2. E1 is the only change with a material gain; its
  mechanism is directly measured (frozen collate 9.5–10 ms vs 1.2–1.5 ms,
  step time unchanged at ~32–35 ms), and the 1.31× ratio survives the ±10 %
  host noise. **Adopted implementation: E1** (vectorised collate only; frozen
  step, per-step synchronisation, order, loss, clip and optimizer untouched).
  Batch-size probes (256: 5113 mol/s; 512: 13342 mol/s) are recorded for
  information only and are not used in any formal run. Stage-1 + stage-2 GPU
  time is ≈0.35 h, inside the 0.75 h allowance.
- 2026-11-11 — pre-run budget check with per-width E1 timing
  (`analysis/bench_scale{1,2,3}.json`, two rounds each, same cache):
  259,831 / 939,079 / 2,038,167 parameters all take ≈33–36 ms per batch-128
  step on this host; the step is launch-bound, so width costs almost no
  throughput. Conservative projection at 38 ms/step (collate + staging +
  step) is 3.3 h per 40-epoch 1M run plus ≈0.08 h evaluation; W3 + W1
  therefore fit inside the remaining budget with margin. W3 starts first;
  W1 is launched only after W3 completes and only if the ledger still allows.
- 2026-11-11 — **execution-regime change forced by infrastructure.** At 11:54
  the `res` host (a100-2) became unreachable: the jump host and ICMP still
  worked, but TCP 22/6003 failed; the interruption persisted. The W3 run on
  `res-gpu1` was stopped mid-training (last visible epoch 18 at 11:18; its
  partial checkpoints remain on that host and are not used). To keep the
  budgeted round moving, the formal runs were moved to `res-2` -> node `c06`
  (A100-PCIE-40GB, driver 525.85.12, torch 2.5.1+cu124, Slurm pool
  `res2-c06` pinned because `--nodelist=c05,c06` collapsed to the busy c05).
  The 1M + Validation caches, labels, subset rows, monitor rows and the
  historical W2 checkpoint were copied byte-exactly (cache tree hashes and the
  W2 checkpoint SHA256 `e45316c2…ed0b1` re-verified on c06). All 19
  correctness gates were re-run **in the new regime** (`analysis/verify_res2.json`,
  status OK), including E1-vs-frozen bitwise equality over 200 steps at W2 and
  W3 and the 200-step checkpoint/resume gate. W3 restarts from epoch 1 on
  c06; W1 will follow on the same node. The changed regime is recorded for
  every timing number: throughput comparisons across hosts are not made —
  the controlled width-vs-speed measurement remains
  `analysis/bench_scale{1,2,3}.json` (same host/conditions). MAE is
  deterministic given the seed/order/labels and is compared across regimes
  with that caveat stated in REPORT.md.
- 2026-11-11 — post-run clarification of the outage record. `res` came back at
  ~20:30 with a fresh boot; the partial W3 directory shows the process wrote
  `checkpoint_last.pt` and the epoch-23 row at **11:45:44**, i.e. the machine
  died then (SSH had already become unreliable). The lost attempt is therefore
  charged **2.03 GPU-hours**, not 2.17, and it is not used. Comparing its curve
  with the completed c06 run, **all 23 completed epochs are bitwise identical**
  in `train_l1_z`, `valid_mae_eV` and `monitor_mae_eV`, strengthening the
  cross-regime determinism statement; the final result still comes solely from
  the completed c06 run.
