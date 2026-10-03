# EXECUTION — zinc-full-decomposition-valid-test-confirmation-v1

## Wall clock

First tool call (reconnaissance) ≈ `2026-10-03T09:47Z`; first authored artifact
(protocol) `2026-10-03T10:06:38Z`; branch created `2026-10-03T10:01:40Z`.
All new training compute stopped at `2026-10-03T11:29:50Z` (last Full run exit).
Frozen evaluation and delivery finished `≈2026-10-03T11:55Z`.  Active wall
≈ `2 h 05 m`, below the 180-minute cap; the last ≈25 min were analysis / report
only (no training).

## Resource regime (all remote jobs through `rr`, host `res-2`, pool `res2-cu124`)

* Skill path: `~/.pi/agent/skills/remote-research-runner/SKILL.md`
  (historical path `/home/calendar/.pi/agent/skills/remote-research-runner/SKILL.md`;
  identical file, read in full at round start).
* Device: `NVIDIA A100-PCIE-40GB` on node `c05`; driver `525.85.12`;
  torch `2.5.1+cu124`; python `3.12.14`.  All four Full runs on the same node /
  driver / torch / pool.
* FP32; no AMP; no DDP; `torch.set_num_threads(8)`; local CPU work capped at 8.
* GPU budget: **4 Full jobs, 1.732 GPU-hours, max 2 concurrent** (≤4 GPU-h,
  ≤2 GPU).  `Q` heads and all evaluation/analysis ran on local CPU.
* Two waves, Y/H of the same seed in parallel: wave 1 `zfdtc-Y-s0` + `zfdtc-H-s0`;
  wave 2 `zfdtc-Y-s1` + `zfdtc-H-s1`.

## Steps

| # | step | artifact | result |
|---|---|---|---|
| 0 | isolated branch `task/zinc-full-decomposition-valid-test-confirmation-v1` | — | created from `c11342e` |
| 1 | freeze protocol + runner + analysis, commit before fitting | `f9fb109` | frozen |
| 2 | all-10k train-only prep (`build_fold_inputs` semantics, `fit_idx=all`) | `all_train_prep.npz`, `prep_meta.json` | K-SVD `296 s`, subspace, scalers |
| 3 | topology25 snapshot (all 10000) | `T25_all.npz` | `(10000,25)`, `2129ec2f…` |
| 4 | smoke (≤8 steps, discarded) + adapter parity (128 train rows) | `smoke_checks.json`, `adapter_checks.json` | both pass |
| 5 | 4 Full trajectories, 10k rows, 240 epochs, soup 236–240 | `Y/H_seed{0,1}_*` | exit 0, replay ≤1.5e−6 |
| 6 | 2 independent `Q` heads, 300 epochs, soup 296–300 | `Q_seed{0,1}_*` | 12.8 s each |
| 7 | single train-only calibration | `calibration.json` | `b_Y/b_H/b_P/b_K` |
| 8 | deploy wrapper sanity | `wrapper_checks.json` | ≤1.7e−6, label delta 0 |
| 9 | **freeze manifest before any held-out read** | `frozen_eval_manifest.json` (`3878d12`) | `official_*_loaded=false` |
| 10 | one frozen official-valid/test prediction pass | `heldout_*`, `heldout_access.json` | n=1000 each, label align ≤4e−15 |
| 11 | tables / bootstrap / gate (cached predictions only) | `analysis_summary.json`, `bootstrap.json`, `*_table*.csv` | valid gate FAIL |
| 12 | reports + budget + run_meta | `REPORT/DECISION/EXECUTION/ERRATA`, `budget.json` | — |

## Execution-regime deviation (recorded, non-recipe)

* Wave-1 Full jobs ran commit `f9fb10989660`; wave-2 ran `b88ab82145de`.
* The only difference between the two commits is a one-line fix in the
  **cycle-head** path (`state_hash(head)` → `state_hash(head.state_dict())`),
  which is not on the Full training path.  The four Full trajectories therefore
  execute byte-identical Full code; only the (local, CPU) head-hash logging
  changed.  Both wave-1 and wave-2 `Y_s`/`H_s` are paired within their commit
  (identical init hash, identical batch order).
* No result was re-run for score; no seed/horizon change.

## Access status

* `official_valid_loaded = false`, `official_test_loaded = false` throughout all
  fitting, calibration and freeze.
* First official-valid read `2026-10-03T11:41:13Z`; first official-test read
  `2026-10-03T11:41:26Z`, both **after** the frozen manifest commit `3878d12`.
* The historical `test_access: blocked` file was not modified.
* Test source: official ZINC subset test (PyG), built through the repository's
  terminal test path with an **explicit** `test` mapping (no `valid→val else
  train` fallback); verified by `n=1000` and exact label alignment
  (`y_max_abs_diff ≤ 9e−16`).

## Analysis-code deviation after freeze (post-prediction only)

`frozen_eval_manifest.json` pins `analysis_source` at
`sha256 5150912454c09186…`; the committed `analyze.py` is
`sha256 7a200b76bc31b569…`.  The difference is **analysis-only bookkeeping**
added after the held-out predictions were written: a `median_c` lookup fix in
`_train_predictions`, the corrected constant-shift invariance check, and the
paired bootstrap/CSV writers.  The held-out prediction path (`heldout`,
`_load_pair`, `_predict_arms`, `_save_split`, `_access_log`) is byte-identical
to the frozen revision, so the saved predictions are from the frozen code; only
post-prediction tables changed.  `runner_source`, `deploy_wrapper_source`,
`all_train_prep` and `target_decomposition` hashes still match the manifest.

## `rr` job status / exit codes

| experiment | commit | node | exit | window (local) |
|---|---|---|---:|---|
| `zfdtc-Y-s0` | `f9fb109` | c05 | 0 | 18:33:52 → 18:57:13 |
| `zfdtc-H-s0` | `f9fb109` | c05 | 0 | 18:34:06 → 19:01:27 |
| `zfdtc-Y-s1` | `b88ab82` | c05 | 0 | 19:01:43 → 19:29:50 |
| `zfdtc-H-s1` | `b88ab82` | c05 | 0 | 19:01:57 → 19:27:04 |

Full per-job provenance in `run_meta/*.meta.json`; budget in `budget.json`.

## Missing / not done (by design)

* No third seed, no head-width/λ/WD/loss/optimizer search, no Full fine-tune.
* No dictionary-vs-MLP ablation, no new graph descriptor, no oracle column in the
  test main table.
* The official-valid gate failed; the official-test result is reported but does
  not alter the gate.  The rare extreme cycle tail (`k≤−3`) remains the dominant
  open error source.