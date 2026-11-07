# LCRC PCBA Backbone Scaling v1 — how to run this round

Round id `lcrc-pcba-backbone-scaling-v1`. Read [`PROTOCOL.md`](PROTOCOL.md)
first: it is frozen before the first formal step and defines every width,
threshold and rule used here.

## What this round is

Two models, one seed, one formal 40-epoch from-scratch run each:

| model | scale | trunk | attention readout | total params |
|---|---:|---:|---:|---:|
| `W2` | 2× | 848,096 | 146,228 | **994,324** |
| `W3` | 3× | 1,809,408 | 311,436 | **2,120,844** |

They are the historical LCRC-A computation (one atom encode, one bond encode,
one-shot S1/S2 local statistics, per-atom `f`, global `rho`, four-head attention
pool readout) with *only* the widths scaled. The reference A0 (287,644 params,
epoch-40 official Validation AP 0.2463560956) is never retrained; it is replayed
read-only from
`../lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A`.

## Code map

| file | role |
|---|---|
| `code/bs_common.py` | frozen constants: widths, parameter targets, recipe, identities, eval-cadence rule, budget, GPU/occupancy helpers |
| `code/bs_model.py` | `BSModel` (scaled `TULCRC` `C1` + unchanged `AttentionPoolReadout`), FULL→C1 remap at arbitrary width, the frozen initialisation, the width audit, the atom-renumbering helper |
| `code/bs_train.py` | the single training step (shared with the cost probe), the 40-epoch loop, per-epoch instrumentation, resumable + final checkpoints, `run.json` |
| `code/bs_audit.py` | `gpu-preflight` / `model` / `a0-replay` / `eval-probe` audit modes |
| `code/bs_compare.py` | comparison, per-task deltas, resource labels, GPU ledger |

Inherited **read-only**: `tu_model.TULCRC`, `nmp_model.AttentionPoolReadout`,
`sg_common.SCRATCH_RECIPE`/`scheduled_lr`, `sg_data` (data, plan, official OGB
evaluator), `pc_metrics`, `pc_model`. `tu_common.MODEL_CONFIG` is never touched.

## Commands

```bash
# 0) local correctness (25 tests, ~6 s)
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/tests

# 1) pre-training audit + cost probe on the remote A100 (two jobs, seconds each)
rr run res bs-preflight --pool res-local --result <round>/analysis -- \
  uv run --no-sync python <round>/code/bs_audit.py --modes gpu-preflight
rr run res bs-audit --pool res-gpu1 --result <round>/analysis -- \
  uv run --no-sync python <round>/code/bs_audit.py --modes model,a0-replay,eval-probe

# 2) formal runs (one job each, separate GPUs; eval cadence from §6.1)
rr run res bs-w3 --pool res-gpu1 --result <round>/runs/W3 -- \
  uv run --no-sync python <round>/code/bs_train.py --model W3 --seed 0 --device cuda \
  --eval-every <cadence> --out-dir <round>/runs/W3 --soft-seconds 14400 --hard-seconds 18000
rr run res bs-w2 --pool res-gpu0 --result <round>/runs/W2 -- \
  uv run --no-sync python <round>/code/bs_train.py --model W2 --seed 0 --device cuda \
  --eval-every <cadence> --out-dir <round>/runs/W2 --soft-seconds 14400 --hard-seconds 18000

# 3) analysis
rr pull bs-w2 ; rr pull bs-w3 ; rr jobs --json > <round>/analysis/rr_jobs.json
uv run --no-sync python <round>/code/bs_compare.py --rr-jobs <round>/analysis/rr_jobs.json
```

## Artifacts

```
analysis/model_scaling_audit.json   gates 1-2, 4-13 (params, widths, one-shot, no re-propagation,
                                    renumbering, normalisation, forward/backward, Test isolation)
analysis/a0_replay.json             gate 3 (historical epoch-40 A AP reproduced within 1e-4)
analysis/gpu_preflight.json         device/NVML/occupancy facts used for the scheduling decision
analysis/evaluation_cost_probe.json real step time, full 43,793-row Validation time, projection,
                                    resolved cadence, budget label
analysis/w2_epoch40.json            the primary epoch-40 record for W2
analysis/w3_epoch40.json            the primary epoch-40 record for W3
analysis/scale_comparison.json      main table, curves, trends, per-task aggregates, efficiency
analysis/per_task_deltas.csv        128 rows: AP and Δ per task with Validation-support stratum
analysis/decision.json              labels, budget status, answers, conclusions
analysis/gpu_ledger.json            GPU-hours per job incl. preflight and any failed attempt
runs/W2/, runs/W3/                  curves, per-task AP, checkpoints, RNG/optimizer state, run.json
```

## Notes

* `runs/` is git-ignored; everything needed to re-derive `analysis/` is the
  committed code plus those local run directories.
* The primary comparison is the fixed epoch 40. The best-Validation checkpoint
  is kept for diagnostics only and never substitutes for it.
* No official Test row is indexed anywhere in this round.
* `eval-every` must match the cadence resolved in `PROTOCOL.md` §6.1 (1 or 2);
  the trainer records the cadence in `run.json` and refuses a formal run that
  disagrees with `bs_common.EVAL_CADENCE` (`formal_run: false` otherwise).
