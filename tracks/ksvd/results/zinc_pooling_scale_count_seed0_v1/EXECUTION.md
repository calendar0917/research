# EXECUTION — `zinc_pooling_scale_count_seed0_v1`

Round start (first tool call): **2026-10-04 ~22:26 CST**. Frozen protocol/runner/inputs committed
at **`ddd5bd394fe2`** before any training. Branch `task/zinc-pooling-scale-count-seed0-v1`; no
push, no merge, main untouched.

## 1. Pre-work (local CPU, ≤8 threads)

| step | command | result |
|---|---|---|
| frozen inputs | `--prepare` | copied + SHA256-verified all six source artifacts; `frozen_input_copy.json` |
| pre-checks | `--pre-checks` | `all_ok=true` (12.2 s); init identity max abs 0.0; forward equivalence bitwise 0.0; schedule/RNG hashes match source; 24/24 pooling reference checks; count witness all_ok; `count_snapshot.npz` saved |
| smoke C | `--smoke --arm C --device cpu` | 3 steps, losses finite, save/reload exact, states discarded |
| smoke N | `--smoke --arm N --device cpu` | 3 steps, losses finite, save/reload exact, states discarded |

Identity witnesses (all in `pre_checks.json`): C/N init state byte-identical to source
`M_init_state.pt` (297,499 params, `W_loc ≡ 0`, `A_raw = D_loc_init.T`); build RNG
`a2e8a8ab…`, train RNG `1ccf1725…`, schedule `7b11a529…`; six count coordinates verified at
runtime against independently recomputed counts and against N's per-`n` means.

## 2. Remote attempt (res-2) and fallback

| item | value |
|---|---|
| deploy | `rr deploy res-2` OK, `ddd5bd394fe2` (38 s) |
| submitted | `pool-scale-C` run `pool-scale-C-20261005-000132-45792c8d`, job `55976`; `pool-scale-N` run `pool-scale-N-20261005-000146-a669840f`, job `55977` |
| scheduler | both PENDING, queue reason `(Resources)`, for ~11 min (00:01:32 / 00:01:46 → 00:13) |
| probe | CPU probe job 55978 also PENDING; `res` GPU probe on `res-gpu0` returned `torch.cuda.is_available()=False` (NVML device-handle error, `rr doctor res` WARN) |
| action | cancelled 55976/55977; no GPU allocation was ever held, so GPU UUID/PCI verification is N/A |
| fallback | explicit user authorization to run locally without budget constraints; local CPU execution (16 cores, 2 processes × 8 threads, torch 2.5.1+cu124 CPU, FP32, no AMP/DDP) |

## 3. Formal trajectories (local CPU, commit `ddd5bd394fe2`)

| field | C | N |
|---|---|---|
| launch / finish | 00:15 / 01:16 | 00:16 / 01:19 |
| wall | 3645.8 s | 3667.4 s |
| steps | 15,120 / 15,120 | 15,120 / 15,120 |
| epoch time | ~15.5 s | ~15.6 s |
| position / gid stream | `7b11a529…` / `69187f13…` | same |
| train RNG start | `1ccf1725…` | same |
| init identity vs source M | max abs 0.0 | max abs 0.0 |
| calibration bias (fit median) | −0.031017 | +0.022794 |
| CPU replay (128 fit rows, single file) | 0.0 | 0.0 |

Note: the first N launch failed only because of a shell redirect path error (`/N_train_cpu.log`),
not a scientific failure; it was relaunched cleanly one minute later and completed normally.
Training logs: `C_train_cpu.log`, `N_train_cpu.log`.

## 4. Analysis (local CPU, `--analyze`, 17.6 s)

`classification=NO_CANDIDATE`, gate summary `{N_vs_C: 0/5, N_vs_M: 0/5, C_vs_M: 3/5}`. Source
`M` soup replay with the original C6 path: max abs `9.5e-7` (≤1e-5) and all four source main
metrics reproduced to 0.0. Bootstrap seed `20261006`, 1000 draws; witnesses pass. Outputs:
`analysis.json`, `gains.json`, `gate.json`, `bootstrap.json`, `classification.json`,
`main_table.csv`, `group_table.csv`, `group_gain_table.csv`, `scale_table.csv`,
`mechanism_evidence.json`, `replay_checks.json`, `budget.json`, `manifest.json`.

## 5. Budget / regime honesty

* Pre-registered budget: ≤120 min wall, no new compute after minute 90, ≤0.8 allocation-GPU-h,
  ≤2 concurrent GPUs. Actual: training ran locally on CPU (no GPU allocation) and finished at
  ~02:55 h after round start because GPU resources were unavailable; user authorization at ~00:14
  explicitly waived the budget constraint.
* `training_gpu_hours_upper_bound = 2.0315` in `budget.json` is the sum of the two CPU wall
  times expressed in GPU-hour-equivalent units; it is **not** GPU consumption (no GPU was used).
  Actual GPU allocation-hours: **0**.
* The required CPU/GPU replay-equality check could not be performed (no GPU); CPU single-file
  replay is exact (0.0), and the source M soup replay is 9.5e-7.
* Both remote jobs were cancelled while PENDING; final self-created job states: C/N/alloc-probe
  submitted then cancelled, gpu-probe completed, res-2 queue clean. No running/pending job
  remains.

## 6. Repro commands

```bash
uv run python -m tracks.ksvd.experiments.luyin16.zinc_pooling_scale_count_seed0_v1 --prepare
uv run python -m tracks.ksvd.experiments.luyin16.zinc_pooling_scale_count_seed0_v1 --pre-checks
uv run python -m tracks.ksvd.experiments.luyin16.zinc_pooling_scale_count_seed0_v1 --smoke --arm C --device cpu
uv run python -m tracks.ksvd.experiments.luyin16.zinc_pooling_scale_count_seed0_v1 --train --arm C --device cpu
uv run python -m tracks.ksvd.experiments.luyin16.zinc_pooling_scale_count_seed0_v1 --train --arm N --device cpu
uv run python -m tracks.ksvd.experiments.luyin16.zinc_pooling_scale_count_seed0_v1 --analyze
uv run python -m tracks.ksvd.experiments.luyin16.zinc_pooling_scale_count_seed0_v1 --budget --manifest
```

(`--train` on a GPU host follows the same entry point with `--device cuda`; the frozen protocol
intended res-2 `res2-cu124`.)
