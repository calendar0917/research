# EXECUTION — zinc-full-cycle-target-decomposition-v1

Wall-clock start `2026-10-03 05:46:55 UTC` (first tool call).  Compute cutoff
`08:16:55 UTC`; hard stop `08:46:55 UTC`.  Local CPU capped at 8 threads /
one heavy worker; remote compute only through `rr` on `res-2`, pool
`res2-cu124`, ≤ 2 GPUs.  Cumulative GPU cap 3.5 GPU-hours.

## 0. Orientation and Phase 0 (local CPU)

| step | command | result |
|---|---|---|
| orient | skill read, `git status`, `STATE.yaml`, prior prep/audit sources | base commit `7528377` (main) |
| branch | `git checkout -b task/zinc-full-cycle-target-decomposition-v1` | isolated from `7528377` |
| phase 0 | `python -m ...zinc_full_cycle_target_decomposition_v1 --mode phase0` | `checks.json`, `input_manifest.json`, `target_decomposition.npz` |

Phase 0 (`checks.json`) reproduced every frozen input:

* split `fit_idx` / `dev_idx` SHA-256 equal `165e87ef…` / `fb8b7806…`; dev
  strata `1926 / 65 / 9`; no canonical group straddles fit/dev.
* `y` from `encoded_train.pt` equals the handoff target (`max_abs = 0`); `c` and
  `g` are finite; `y = g + c` and `c = (snapped - mu_cycle)/sigma_cycle` to
  `1.8e-15`; `k` counts `9628 / 325 / 47` (all-train).
* train-only loading: `encoded_train.pt` + `env_train.pt` only.  The official
  validation split is not loaded; the official test split is never
  instantiated.
* the frozen 8000-fit prep was reapplied and the all-train X175 normalised hash
  reproduced `1ca3163…` exactly (only the train branch of
  `apply_prep_blob` runs).

## 1. Frozen smoke (≤ 8 optimizer steps)

Local CPU smoke first, then remote.  Three remote smoke jobs were used; two
failed on **engineering** bugs that were fixed and re-frozen before any formal
run:

| # | experiment | commit | outcome |
|---|---|---|---|
| 1 | `zftd-smoke` | `837a900` | failed: smoke batch not moved to device (CUDA `phi @ U` mismatch) |
| 2 | `zftd-smoke2` | `fb393d8` | failed: paired-init forward compared with exact equality, CUDA float32 gave `6.3e-7` |
| 3 | `zftd-smoke3` | `2cab640` | **completed**, `mechanism_ok = True` |

`smoke_checks.json` (local) and the remote `smoke/smoke_checks.json` confirm:

1. seed-0 `Y`/`O` initial state hashes identical (`408,651` params); seed-1
   differs; initial paired forward within float32 tolerance.
2. the two training targets differ exactly by `c` (`max_abs 5.8e-8`).
3. `O` task gradient reaches the reader, `D_L`, `V_L` and structural `D`; the
   reconstruction term (λ = `33.95873017865987`) reaches structural `D` and does
   **not** touch the task dictionary.
4. oracle identity `y-(h+c) = g-h` exact (`5.6e-17`); control identity exact;
   paired gain of identical predictions is `0`, a swapped arm flips sign.
5. encoded `y` and the cached labels are unchanged.

## 2. Formal paired trajectories

Deployed commit `2cab640be142` (clean worktree, `rr deploy res-2`).  Four paired
trajectories, submitted together and queued by Slurm (2-GPU quota):

| experiment | arm | seed | layer |
|---|---|---|---|
| `zftd-Y-s0` | Y | 0 | control |
| `zftd-O-s0` | O | 0 | oracle diagnostic |
| `zftd-Y-s1` | Y | 1 | control |
| `zftd-O-s1` | O | 1 | oracle diagnostic |

Each run: canonical Full `408,651`, `scale_seed = 0`, fresh seed-specific init,
Adam coupled-L2 `lr 1e-3` / `wd 1e-5`, batch `128`, clip `5`, FP32, no
scheduler/AMP/DDP, 240 epochs, fixed last-5 soup `236–240` parameter average,
fit-only external bias.  Provenance per run is in `run_meta/`.

(Results, timing, and GPU-hour accounting are appended in the delivery
summary.)

## 4. Outcome

All four trajectories `completed`, exit `0`, on `c05` (A100-PCIE-40GB, driver
`525.85.12`, torch `2.5.1+cu124`).  Wall clock per run `1,307–1,362 s`; total
formal GPU time `5,362 s = 1.489 GPU-hours` (cap 3.5).  Both seeds ran in full
(`15,120` optimizer steps each), no run was shortened or re-bought.

Deployed formal commit: `2cab640be142c9203fb0cf6c0de88b2eddaefef5`, clean
worktree (`git_dirty = false`).  Per-run `node`/`driver`/`torch`/`commit` are in
`run_meta/summary.json` and the raw `run_meta/*.meta.json`.

Result: route signal **met**, branch **B1**; mean cal gain `+0.022697`, mean
`G0` MAE gain `+0.003599`, severe group `78.7 %` of the gain.  See `REPORT.md`
and `DECISION.md`.

## 3. Stop

After the four trajectories finished, all compute stopped.  No official-valid
evaluation, no official test, no cycle head, no extra seed, no `lambda` / WD /
loss / optimizer search, no node rescue, no coord perturbation.