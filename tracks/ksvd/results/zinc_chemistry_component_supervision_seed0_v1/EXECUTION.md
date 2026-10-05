# EXECUTION — `zinc_chemistry_component_supervision_seed0_v1`

## Timeline (wall clock)

* `2026-10-05 11:45:15` round start (phase-targets / init-check / smokes already completed locally).
* `2026-10-05 11:49:17` SUM training submitted (Slurm job 56009, c05, res2-cu124).
* `2026-10-05 11:49:31` COMP training submitted (Slurm job 56010, c05, res2-cu124).
* `2026-10-05 11:58:26` SUM completed (15120 steps, 641.6 s).
* `2026-10-05 11:59:08` COMP completed (15120 steps, 669.7 s).
* Analysis + budget + manifest completed locally.

Both formal trajectories were submitted before the minute-90 gate and well inside the 120-minute
wall and 0.8 GPU-hours budget.

## Resource accounting

| arm | wall_s | steps | GPU | job | node | driver | torch/cuda/python |
|---|---|---|---|---|---|---|---|
| SUM | 641.6 | 15120 | 1×A100-PCIE-40GB | 56009 | c05 | 525.85.12 | 2.5.1+cu124 / 3.12.14 |
| COMP | 669.7 | 15120 | 1×A100-PCIE-40GB | 56010 | c05 | 525.85.12 | 2.5.1+cu124 / 3.12.14 |

* GPU budget used: `0.364` allocation-GPU-hours (limit 0.8).
* Both arms ran FP32, no AMP, no DDP, on the same deployment commit
  `e271f67c3fe9 (task/zinc-chemistry-component-supervision-seed0-v1)`.
* Thread caps respected: 8 local CPU threads during prep/analysis, 4 remote during training.

## Physical GPU identity (not the local CUDA index)

`CUDA_VISIBLE_DEVICES=0` was the in-allocation local index for **both** jobs, but the two jobs held
separate one-GPU Slurm allocations and the captured UUIDs differ, confirming they never shared a
card (no oversubscription):

* SUM: `GPU UUID GPU-ba05543b-868e-8605-9dec-a8e4f2a82bc3`, PCI `00000000:0D:00.0`.
* COMP: `GPU UUID GPU-c7067be6-0979-516e-45c4-27f69628e1df`, PCI `00000000:1D:00.0`.

The allocation probes are stored in `SUM_meta.json` / `COMP_meta.json` (`allocation_probe`) and in
`alloc_targets.json`.

## Parallel vs serial

The two arms occupied two distinct GPU allocations and were run **in parallel** (they overlap in
wall-clock). They are not serialised.

## Recovery / restarts

No training job failed or restarted. The earlier device-placement failure (jobs 56007/56008 on
commit `17546fee`) was a device-placement bug caught during smoke-level execution on CUDA, fixed
before the two formal runs; those failed attempts produced no dev scores and their fragments were
discarded. The two formal runs started from the committed init and the exact fresh-fold RNG stream;
no checkpoint/RNG resume was needed because both completed cleanly.

## Science vs execution status

* **Science status: COMPLETED.** Two formal trajectories (SUM, COMP), one seed, matched recipe and
  fold/data/labels; the comparison is valid (no dev-driven configuration selection).
* **Execution status: CLEAN.** Exactly two formal trajectories, both completed within budget and
  before the minute-90 new-work cutoff; the two early failed jobs are engineering failures before
  the formal pair and are excluded from the scientific comparison (their dev scores were never read).
