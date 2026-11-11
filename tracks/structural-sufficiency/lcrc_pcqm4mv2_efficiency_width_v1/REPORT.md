# PCQM4Mv2 — efficiency + width round v1

Round `lcrc-pcqm4mv2-efficiency-width-v1` · 2026-11-11 · code commits
`6e9d58c` (W3) / `3663e07` (W1, doc-only diff) · historical control `93ae701f`

Two questions on the **same fixed 1,000,000-molecule PCQM4Mv2 Train subset**
and the same 40-epoch recipe:

* **Q1 (engineering):** can the real end-to-end batch-128 training throughput be
  improved without changing order, loss, clip, optimizer or evaluation?
* **Q2 (capacity):** does the LCRC parameter count change Validation MAE?
  W1 (259,831) and W3 (2,038,167) are trained; historical W2 (939,079,
  0.132310904 eV, 5.162 GPU-h) is read-only and was **not** retrained.

**Decision: `NO_WIDTH_SIGNAL`.** W3 reaches Valid@40 `0.127874764` eV versus
W2's `0.132310904` eV — an improvement of **0.004436 eV**, below the
pre-registered 0.005 eV threshold, and the late trend is not monotone
(30→35 rises by 0.00376 eV). The direction is consistently favourable from
epoch ~12 onward, but the pre-registered bar is not met and no stable advantage
is claimed. Q1 succeeds: the vectorised collate (E1) gives **1.31×** end-to-end
at batch 128.

## 1. Result table (first page)

| model | parameters | best Valid MAE (eV) @ epoch | Valid@40 (eV) | monitor@40 (eV) | GPU-hours | molecules/s |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 259,831 | 0.143313990 @ 36 | 0.143362438 | 0.124051795 | 3.354 | 3,407 |
| W2 (historical) | 939,079 | 0.132310904 @ 40 | 0.132310904 | 0.105890654 | 5.162 | 2,189 |
| W3 | 2,038,167 | 0.127809167 @ 39 | 0.127874764 | 0.097213247 | 3.447 | 3,316 |

Regimes: W2 on `res`/a100-2 (A100-SXM4-40GB, frozen code), W1/W3 on
`res-2`/c06 (A100-PCIE-40GB, driver 525.85.12, torch 2.5.1+cu124, E1). MAE is
deterministic given seed/order/labels; every W3 epoch that can be compared
against the aborted `res` run (**all 23 completed epochs**) matched it
bitwise — including `valid_mae_eV` and `monitor_mae_eV` to the last digit —
which supports the cross-host MAE comparison. **Throughput is never compared across hosts**: the
controlled width-vs-speed measurement is the same-host
`analysis/bench_scale{1,2,3}.json`.

Δ(W3 − W2) at Valid@40 = **−0.004436 eV** (−3.35 %); Δ(best) = −0.004502 eV.
Monitor−Valid gap at epoch 40: W2 −0.026420 eV, W3 −0.030662 eV.

**At a glance (same page as the table).**
Speed: frozen 2,759 → E1 3,626 mol/s (**1.31×**); the 1M 40-epoch run costs
3.45 h (W3) / 3.35 h (W1) on the adopted path versus 5.16 h for the historical frozen W2 run (different host; throughput is not compared across hosts).
Profiling hotspots: collate 9.5 ms/step (frozen) → 1.3 ms (E1); 1,215 kernel
launches/step; training GPU only ≈7 % busy; backward 18.8 ms of a 33.8 ms
step; evaluation 0.76 ms/step amortised; explicit syncs 0.03 ms.
Gates: 19/19 (`analysis/verify_res2.json`), E1 bitwise-equal over 200 steps,
checkpoint/resume bitwise, official==NumPy MAE, local recheck Δ7e-9.
Fixed epochs (Valid@20/30/35/40): W1 0.1477412/0.1451665/0.1438796/0.1433624; W2
0.1403157/0.1350242/0.1360498/0.1323109; W3
0.1349600/0.1284196/0.1321793/0.1278748. Last-5-epoch means: W1 0.1436678,
W2 0.1335467, W3 0.1292267. Monitor−Valid gap: W1 −0.019311, W2 −0.026420,
W3 −0.030662 eV. Width/MAE/speed: params ×3.62 then ×2.17, step time
essentially flat (W1/W3 on c06: 37.6 and 38.6 ms), MAE gain 0.0111 eV then
0.0044 eV.
Full-data 3.38M × 40 projection: **11.41 h** (W3), 11.11 h (W1), 17.23 h
(historical frozen path) — projection only, not run.
Code: W3 `6e9d58c`, W1 `3663e07`, historical W2 `93ae701f`; W2 checkpoint
`e45316c2…`; W3/W1 checkpoint hashes in §9; GPU ledger **9.29 h** of 12 h
(§8); all commands in §9; curves in `analysis/curves.png` / `.csv`.

### Answers to the five required questions

1. **Speed-up factor.** **1.31×** end-to-end at batch 128 on the real 1M cache
   (frozen 2,759 mol/s → E1 3,626 mol/s; 46.4 → 35.3 ms/step). Only E1 is
   adopted; E2/E3 and combinations are slower (see §3).
2. **Full-run GPU-hours.** W3: **3.447 GPU-h** measured for 40 epochs on 1M;
   W1: **3.354 GPU-h**; historical W2 5.162 GPU-h. Whole round
   (profiling + gates + both new runs + the lost `res` attempt):
   **9.15 GPU-h** of the 12 h hard budget.
3. **Is W1 redundant with W2?** **No.** W1's Valid@40 is 0.143362438 eV versus
   W2's 0.132310904 eV — +0.011052 eV (W1 worse), far outside the 0.005 eV
   band. Parameter count matters between 1× and 2×.
4. **Does W3 add over W2?** It improves Valid@40 by 0.004436 eV (3.35 %), and
   improves the monitor by 0.008677 eV, but misses the 0.005 eV pre-registered
   threshold and its late curve is non-monotone. So: a consistent small gain,
   **not** a demonstrated width signal.
5. **Full-data investment?** **Not justified by this round** for either width.
   The 3.38M × 40-epoch projection is ~11.4 h (W3) / 11.02 h (W1) on
   this implementation (§7); spending that on a single seed to chase a
   0.0044 eV effect at 1M is premature. If any full-data work happens, it
   should first replicate the 1M W3-vs-W2 contrast with a second seed (≈3.4 h
   each), not start with the widest arm on 3.38M.

## 2. What was measured (Q1)

**Setup.** `res` (a100-2), pool `res-gpu1`, historical W2 builder
(939,079 parameters), the real 1M `pcqm_train1m` mmap cache, batch 128, FP32,
AdamW, 400 measured steps after 50 warm-up steps, cold and hot cache passes,
`torch.profiler` (CPU+CUDA) plus CUDA-event phase spans and wall clock.
Asynchronous CUDA time is never counted as host time; host spans and device
spans are reported separately.

| pass | ms/step (mean) | p50 | p90 | mol/s | wall |
|---|---:|---:|---:|---:|---:|
| cold cache | 44.72 | — | — | 718.9 | 71.2 s |
| hot cache | **33.77** | 32.80 | 41.54 | **2,857.8** | 17.9 s |

Hot-cache host phases per step: **collate 9.48 ms**, pinned-staging issue
1.48 ms, explicit `torch.cuda.synchronize` 0.030 ms (p90 0.037). Device spans
per step: whole step 33.54 ms (34.93 ms including H2D), backward 18.82 ms,
forward+loss span 11.93 ms, AdamW update 1.58 ms, clip 1.17 ms, H2D 1.40 ms.
Cold cache is dominated by collate: **131.14 ms/batch**, i.e. the 1M cache's
first touch costs ~120 ms/batch of page-fault time; it also inflates the
measured step span (44.7 vs 33.8 ms) because the page-fault I/O competes with
the host. Once the rows are resident, the step cost is the hot number below.

**The step is launch-bound, not GPU-bound.** During the measured window the
training device was **≈7 % utilised on average (max 22 %)**, while the process
used ≈7.8 CPU cores (max 24.2) and the profiler counted **1,215 CUDA kernel
launches per step** (36,462 in 30 steps), with `cudaLaunchKernel` alone taking
391 s of the 1,046 s `train_step` CPU window. `aten::mm` (1,230 calls, 259 ms
self CPU), `aten::fill_` (17,280 calls) and the graph-staging copies fill the
rest. The GPU is the idle partner in every non-backward phase.

**Per-epoch amortised overhead** (hot): full Validation 4.44 s (73,545 rows,
official MAE 0.42599 for the untrained model), 20k monitor 1.42 s, checkpoint
0.062 s → 5.93 s/epoch = **0.759 ms/step (2.25 % of the step)**, so evaluation
is not where the time goes.

## 3. Implementations and selection (Q1)

Three candidate changes, measured alone and combined, from the same seed-0
state on the same rows (all configurations must end with identical weights):

| config | mol/s (median of 3 interleaved rounds) | speed vs frozen | notes |
|---|---:|---:|---|
| frozen | 2,759.1 | 1.000 | historical path |
| **E1** | **3,626.0** | **1.314** | vectorised collate; adopted |
| E1+E3 | 3,261.6 | 1.182 | prefetch with fast collate |
| E1+E2+E3 (c16) | 3,399.1 | 1.232 | all three |
| E1+E2 | 2,997.5 | 1.086 | deferred sync |
| E1+E2 (c8) | 2,898.1 | 1.050 | |
| E1+E2 (c16) | 2,709.5 | 0.982 | |
| E2 | 2,325.7 | 0.843 | deferred sync alone |
| E3 | 1,765.0 | 0.640 | producer thread alone |
| frozen, 1 ATen thread | 2,307.3 | 0.836 | diagnostic |
| E1, 1 ATen thread | 2,333.9 | 0.846 | diagnostic |

All configurations reached **bitwise-identical final weights**. The host is
shared and noisy (±10–20 % run to run; the pilot benchmark measured the same
frozen path at 36.4 and 44.1 ms in one session), so the selection rests on a
mechanism that is measured directly rather than on the ranking alone: E1
reduces collate from 9.5–10.2 ms to ≈1.2–1.5 ms per batch while the step time
stays at 32–35 ms, which predicts a 1.25–1.35× end-to-end gain; the measured
3-round median is 1.314×. The other candidates were *consistently slower* — E2
because removing per-step synchronisation lets the launch queue grow without
reducing the host work (the process is CPU-bound, not sync-bound), E3 because
the producer thread competes for the GIL with a host-bound step. Reductions of
the E2 flush interval (64 → 16 → 8 steps) did not rescue it. No "optimised"
machinery beyond E1 is used.

For information only (never a formal recipe): batch 256 → 5,113 mol/s,
batch 512 → 13,342 mol/s. The formal runs stay at batch 128.

## 4. Capacity results (Q2)

Every number below is a full official-Validation MAE after each epoch, plus the
same 20k-row Train-monitor MAE. No early stopping, no selection by monitor; the
"best" checkpoint is the minimum Validation MAE, and Valid@40 is reported
separately.

| epoch | W1 Valid | W2 Valid | W3 Valid | W1 monitor | W2 monitor | W3 monitor |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.1477412 | 0.1403157 | 0.1349600 | 0.1269342 | 0.1183494 | 0.1109407 |
| 30 | 0.1451665 | 0.1350242 | 0.1284196 | 0.1253578 | 0.1098690 | 0.0984256 |
| 35 | 0.1438796 | 0.1360498 | 0.1321793 | 0.1248718 | 0.1105308 | 0.1032885 |
| 40 | 0.1433624 | 0.1323109 | 0.1278748 | 0.124051795 | 0.1058907 | 0.0972132 |
| last-5 mean | 0.1436678 | 0.1335467 | 0.1292267 | 0.1242095 | 0.1073484 | 0.0991742 |

Curves: `analysis/curves.png` (Validation, monitor, train L1, ms/step) and
`analysis/curves.csv` (per epoch, all models). Late behaviour: W3's
Valid@35→40 change is −0.004305 eV but 30→35 is **+0.003760 eV**, so the late
trend required by the decision rule is not consistent. W3's best epoch is 39,
not 40 (0.127809167 vs 0.127874764), another sign of a noisy plateau.

**Parameter / speed / MAE.** Going W1→W2→W3 multiplies parameters by 3.62× and
2.17× (7.84× total). Speed is essentially flat (this host is launch-bound;
all widths are ≈35–42 ms/step), so the width axis is nearly free in compute.
On MAE the ordering is W3 < W2 < W1 at every fixed epoch; the W2→W3 gain
(−0.0044 eV at epoch 40) is far smaller than the W1→W2 gain (0.0101 eV at
epoch 30, 0.0111 eV at epoch 40), i.e. **clear diminishing returns**: the first
doubling of width buys ~2.3× more MAE than the second.

## 5. Correctness and audit

The adopted implementation passed every pre-registered gate on the formal
regime (`analysis/verify_res2.json`, `status OK`, 19/19, 148 s on c06):

* inherited task identity: 1,000,000 Train rows, 73,545 Validation rows, 7,813
  steps/epoch, seed 0, 0 Train/Validation overlap, cache tree hashes unchanged;
* analytic == built parameter counts (W1 259,831 / W2 939,079 / W3 2,038,167)
  and width audit; `ew_model.build_model(scale=2)` bitwise == frozen
  `pq_model.build_model`;
* E1 collate equals frozen collate **field-by-field (value, dtype, shape)** on
  real batches of size 1, 2, 5 and 128; forward is bitwise identical on the
  same batch at all three widths;
* **200 real steps**, frozen vs E1, at W2 and W3: per-step loss and gradient
  norm max |Δ| = 0, final weights and AdamW state bitwise equal, RNG state
  equal, fixed-sample predictions equal;
* checkpoint save → restore → continue reproduces the uninterrupted run
  bitwise (200 + 10 steps);
* atom-renumbering invariance max |Δ| ≤ 7.2e-7 (declared 1e-4);
* official MAE == independent NumPy MAE (Δ = 0) on the full 73,545 rows;
* no Validation row in sampled training batches; no Test access anywhere;
* historical W2 checkpoint SHA256 == `e45316c2…ed0b1` (untouched);
* independent local CPU re-evaluation of the delivered W3 best checkpoint:
  recorded 0.127809167 → recomputed 0.127809174 eV (Δ 7e-9), same rows,
  `analysis/recheck_w3.json`.

The `res` → `res-2` switch (host outage at 11:54) was forced; the caches, row
plans and the historical checkpoint were copied byte-exactly (tree hashes
re-verified) and all gates were re-run in the new regime before the formal
runs. The only code difference between the W3 and W1 commits is PROTOCOL.md
and the verify JSON.

## 6. Read-only diagnostics (best checkpoints)

Full Validation only; the metric is never recomputed after dropping rows. W2
and W3 below use each run's best checkpoint (W3 epoch 39); W1 uses its best
checkpoint. Exact-SMILES overlap is against the **official 3,378,606-row
Train split**; the 1M-subset row is against our 1M Train cache.

| group | W1 | W2 | W3 |
|---|---:|---:|---:|
| Validation MAE (eV) | 0.143313990 | 0.132310906 | 0.127809174 |
| duplicates in official Train | 5,375 (7.308 %) | 5,375 (7.308 %) | 5,375 (7.308 %) |
| MAE, duplicate | 0.138238 | 0.124882 | 0.120822 |
| MAE, non-duplicate | 0.143714 | 0.132897 | 0.128360 |
| dup − non-dup (bootstrap 95 % CI) | −0.005477 [−0.009751, −0.000404] | −0.008014 [−0.012730, −0.003101] | −0.007538 [−0.011671, −0.002762] |
| UNK-containing (230 molecules) MAE | 0.736478 | 0.651691 | 0.604699 |
| UNK-free MAE | 0.141453 | 0.130682 | 0.126313 |

Error by atom count and gap bin (W2 → W3): the ≤20-atom bucket (70,269
molecules) improves 0.128266 → 0.123943; the 21–25 bucket 0.198150 → 0.196309;
26–30 0.270314 → 0.244280; the handful of >30-atom molecules stay very bad
(0.34–0.73 eV) and do not drive the average. By true gap, the largest gain is
in the <4.5 eV bin (16,066 molecules, 0.161779 → 0.150972); 4.5–5.0
0.117667 → 0.113103; 5.0–5.5 0.113221 → 0.110139; 5.5–6.0 0.109716 →
0.107245; ≥6.0 0.143657 → 0.142108. The 230 UNK-containing molecules remain a
catastrophic tail (≈0.6 eV, bias −0.15 to −0.23 eV) and dominate the RMSE of
all three models; they are not a width effect. Exact duplicates are ~0.005–0.008 eV
*easier* than non-duplicates in all three models (bootstrap CIs exclude 0), so the
Validation split does contain a small leak-like component; the official metric
is nevertheless reported unchanged, as required.

## 7. Projection to the full 3,378,606-row split (projection, not a run)

`ceil(3,378,606 / 128) = 26,396` steps/epoch × 40 = 1,055,840 steps. Holding
the measured per-step training cost constant and adding the measured full
Validation + monitor cost per epoch gives:

| path | ms/step (measured, 1M) | projected train | projected + eval | basis |
|---|---:|---:|---:|---|
| historical frozen W2 | 58.48 | 17.15 h | **17.23 h** | measured 40-epoch run |
| frozen path (bench, same host) | 46.4 | 13.61 h | 13.69 h | measured bench |
| E1 at W2 width | 35.3 | 10.35 h | 10.43 h | measured bench |
| E1 at W1 width | 37.57 | 11.02 h | 11.11 h | measured 40-epoch run |
| E1 at W3 width | 38.60 | 11.32 h | **11.41 h** | measured 40-epoch run |

So the engineering work turns the 17.2 h historical projection into **11.4 h
for the widest arm** and **11.02 h for W1** — but this is extrapolation:
the full 3.38M run was **not** executed, as instructed. The per-step cost is
assumed independent of dataset size (it is graph-assembly/launch-bound, not
data-size-bound once the rows are in page cache); the first epoch on a cold
cache would add tens of minutes, not visible in the 1M measurement.

## 8. Resource ledger (A100 GPU-hours)

| item | host | GPU-hours | note |
|---|---|---:|---|
| stage-1 profiling (400 steps ×2 passes) + 5 smokes | res-gpu1 | 0.076 | `analysis/profile.json` |
| pilot benchmark (single pass) | res-gpu1 | 0.041 | discarded as noisy |
| interleaved benchmark (11 configs × 3 rounds) | res-gpu1 | 0.058 | `analysis/bench.json` |
| per-width timing (scale 1/2/3) | res-gpu1 | 0.010 | `analysis/bench_scale*.json` |
| correctness gates on `res` | res-gpu1 | 0.032 | first run (pre-E1 gate) |
| W3 attempt on `res` (aborted by host outage) | res-gpu1 | 2.03 | lost, not used |
| correctness gates on `res-2` (two runs) | c06 | 0.106 | 19/19 final |
| **W3 formal, 40 epochs** | c06 | **3.447** | `runs/PCQM_W3_1M/run.json` |
| **W1 formal, 40 epochs** | c06 | **3.354** | `runs/PCQM_W1_1M/run.json` |
| **total** | | **9.15** | budget 12 h |

The lost 2.03 h is charged to the round: the outage was infrastructure, not a
recipe change, and the rerun was necessary to keep a single complete W3 run.
Even with it, the round stays inside the 12 h hard budget. GPU memory: 309 MB
peak allocated for W3, 146 MB for W1, of 40 GB; process RSS ≈10.0 GB for both
runs (the 5.5 GB mmap cache is resident in RSS); the step is host-bound, so
the GPU is mostly idle (≈7 % utilisation in the profiling window).

## 9. Artifacts, hashes, commands

* Protocol: `PROTOCOL.md` (frozen before the first profiling step; amendments
  in §9 record the E1 selection and the forced regime switch).
* Code (committed): `code/ew_common.py`, `ew_data.py`, `ew_model.py`,
  `ew_collate.py`, `ew_pipeline.py`, `ew_train.py`, `ew_bench.py`,
  `ew_profile.py`, `ew_verify.py`, `ew_recheck.py`, `ew_analysis.py`,
  `ew_diagnostics.py`; tests `tests/test_ew_round.py` (17 passing).
* Analysis (committed, small): `analysis/profile.json`, `bench.json`,
  `bench_pilot.json`, `bench_scale{1,2,3}.json`, `verify.json`,
  `verify_res2.json`, `summary.json`, `summary_table.md`, `curves.csv`,
  `curves.png`, `diagnostics.json`, `diagnostics.md`, `recheck_w1.json`,
  `recheck_w3.json`.
* Runs (git-ignored, paths + hashes recorded): `runs/PCQM_W3_1M/` and
  `runs/PCQM_W1_1M/` — `run.json`, `train_curve.csv`, `checkpoint_best_valid.pt`,
  `checkpoint_last.pt`.
  * W2 historical best: `…/transfer_pilot_v1/runs/PCQM_W2_1M/checkpoint_best_valid.pt`,
    SHA256 `e45316c29d9152adf79e702ebd701bf62e6237065112f8b000b419d1de3ed0b1`.
  * W3 best (epoch 39, 24.5 MB): SHA256 `f4175f141702b9cc10b23404c6689936fed0692aac32086489e0a4bf2492b5ac`;
    W3 last: `7b042fb7a56da9c5071454092635f71535ed49a24be397444b440ee950129e6d`.
  * W1 best (epoch 36, 3.2 MB): SHA256 `d33a6a6b032405e33e5821e839fc7a098dfdb60eec55c54565465f93e6e830da`;
    W1 last: `2fa3b4db080e04827ac7ae6bacc2f5d4f2bc063ffe6056027bbccd3f84384602`;
    W1 epoch-40 primary: `0692c9dda5c37b3e152413e5f0d1071d865c4333b1b140ac90ad34b4b43ea995`.
  * W3 epoch-40 primary: `c67d8590f3c0c5446ac8761ce46b4d89d485ae54b468c721f3ac4562c051d7d1`.
* Data: inherited caches, paths and SHA256 in
  `…/transfer_pilot_v1/analysis/pcqm_cache_manifest.json`
  (`train1m` tree `d8f24e86…`, `valid` tree `2c822540…`), labels
  `29 MB`, subset rows, monitor rows. Never modified.

Commands (representative):

```bash
rr doctor res-2
rr deploy res-2 --pool res2-c06
rr run res-2 ew-verify-r2b --pool res2-c06 --cpus 12 --time 02:00:00 \
  -- python …/code/ew_verify.py --device cuda --steps 200 --out …/analysis/verify_res2.json
rr run res-2 ew-w3-r2 --pool res2-c06 --cpus 12 --mem 32G --time 08:00:00 \
  -- python …/code/ew_train.py --run PCQM_W3_1M --out-dir …/runs/PCQM_W3_1M \
     --implementation E1 --soft-seconds 18000 --hard-seconds 19800
rr run res-2 ew-w1-r2 --pool res2-c06 --cpus 12 --mem 32G --time 07:00:00 \
  -- python …/code/ew_train.py --run PCQM_W1_1M --out-dir …/runs/PCQM_W1_1M \
     --implementation E1 --soft-seconds 16200 --hard-seconds 18000
rr pull ew-w3-r2 ; rr pull ew-w3-r2 --path …/checkpoint_best_valid.pt
python …/code/ew_recheck.py --run PCQM_W3_1M --device cpu
python …/code/ew_diagnostics.py --device cpu
python …/code/ew_analysis.py
```

## 10. Limitations

* One seed per width; the decision is a resource-allocation rule, not a
  significance test. No confidence interval on the width contrast is claimed.
* 1M Train molecules, 40 epochs: the absolute MAE is not comparable to
  full-data literature; the W1/W2/W3 ordering is an internal comparison.
* The `res` outage forced a regime change; MAE matched bitwise on the shared
  epochs, but throughput numbers are reported per host and never mixed.
* The shared host's load made single-window timings unreliable; the retained
  benchmark is interleaved and repeated, and the adopted gain is corroborated
  by the measured collate-time decomposition.
* The full 3.38M projection assumes constant per-step cost; it is clearly
  labelled as a projection and was not executed.
* Diagnostics are descriptive; they were not used to select any model.
