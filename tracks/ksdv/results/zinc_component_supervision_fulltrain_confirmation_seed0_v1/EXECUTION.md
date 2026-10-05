# EXECUTION — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

## Timeline (wall clock)

* `2026-10-05T06:05:09Z` round start (phase freeze / init-check / smokes already completed locally).
* `2026-10-05T05:55:49Z` SUM training submitted (Slurm job 56026, c05, res2-cu124).
* `2026-10-05T05:55:49Z` COMP training submitted (Slurm job 56025, c05, res2-cu124).
* `2026-10-05T06:02:29Z` SUM completed (18,960 steps, 759.4 s).
* `2026-10-05T06:08:12Z` COMP completed (18,960 steps, 834.9 s).
* Q head trained locally on CPU (23,700 steps, 13.1 s).
* Calibration + frozen_eval_manifest + heldout_access + analysis completed locally.

Both formal trajectories were submitted before the minute-90 gate and well inside the 120-minute
wall and 0.9 GPU-hours budget.

## Resource accounting

| arm | wall_s | steps | GPU | job | node | driver | torch/cuda/python |
|---|---|---|---|---|---|---|---|
| SUM | 759.4 | 18960 | 1×A100-PCIE-40GB | 56026 | c05 | 525.85.12 | 2.5.1+cu124 / 3.12.14 |
| COMP | 834.9 | 18960 | 1×A100-PCIE-40GB | 56025 | c05 | 525.85.12 | 2.5.1+cu124 / 3.12.14 |
| Q | 13.1 | 23700 | 0 (local CPU) | — | — | — | 2.5.1+cu124 / 3.12.14 |

* GPU budget used: `0.443` allocation-GPU-hours (limit 0.9).
* Both arms ran FP32, no AMP, no DDP, on the same deployment commit
  `947d2837c493 (task/zinc-component-supervision-fulltrain-confirmation-seed0-v1)`.
* Thread caps respected: 8 local CPU threads during prep/analysis/Q, 4 remote during training.

## Physical GPU identity (not the local CUDA index)

`CUDA_VISIBLE_DEVICES=0` was the in-allocation local index for **both** jobs, but the two jobs
held separate one-GPU Slurm allocations and the captured UUIDs differ, confirming they never
shared a card (no oversubscription), even though they sat on the same node (c05) with different
PCI addresses:

* SUM: PCI `00000000:16:00.0`, UUID `GPU-afab53a3-3153-cf22-9c18-43d754ad0fac`.
* COMP: PCI `00000000:0D:00.0`, UUID `GPU-ba05543b-868e-8605-9dec-a8e4f2a82bc3`.

The allocation probes are stored in `SUM_meta.json` / `COMP_meta.json` (`allocation_probe`) and
in `alloc_targets.json`.

## Parallel vs serial

The two arms occupied two distinct GPU allocations and were run **in parallel** (they overlap in
wall-clock). They are not serialised. The Q head ran on the local CPU in parallel with the body
training.

## Recovery / restarts

One early SUM job (run `zinc-fulltrain-SUM-20261005-132614`, Slurm job 56017) failed with
`FileNotFoundError: full_train_payload.npz` because the `.npz` freeze artifacts were not yet
force-added/committed (they are gitignored by default). The `.pt`/`.npz` files were force-pushed
and deployed. No scores from that failure were read or recorded.

A second early pair (run IDs ending `-132954`) failed because the remote run metadata path
contained a typo `tracks.ksdv` instead of `tracks.ksvd` in the experiment invocation, causing
`ModuleNotFoundError: No module named 'tracks.ksdv'`. No scores from those failures were read.

The two formal runs started from the committed init on the same seed and schedule as the
local smoke checks. No checkpoint/RNG resume was needed because both completed cleanly within
budget. Total failed-fragment engineering runs: 4 (2 pre-freeze, 2 path-typo); all discarded;
zero dev scores consumed.
