# LCRC PCBA Backbone Scaling v1 — how to run this round

Round id `lcrc-pcba-backbone-scaling-v1`. Status: **complete** — two formal runs
executed, see [`REPORT.md`](REPORT.md) for the answers to the 15 questions.
Read [`PROTOCOL.md`](PROTOCOL.md) first: it was frozen before the first formal
step and defines every width, threshold and rule used here.

Result: Val AP@40 = 0.246356 (A0) → 0.261867 (`W2`, +0.0155) → 0.259004 (`W3`,
+0.0126). Both scaled widths are labelled `PROMISING_SCALE_SIGNAL`; the 2×→3×
step is not monotone (−0.0029). Total spend 2.91 A100 GPU-hours.

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
#    (note: the argv after -- is exec'd directly, so wrap multi-command lines in bash -c)
rr run res bs-preflight --pool res-local --result <round>/runs/audit -- \
  bash -c "uv run --no-sync python <round>/code/bs_audit.py --modes gpu-preflight --out-dir <round>/runs/audit"
rr run res bs-audit --pool res-gpu1 --result <round>/runs/audit -- \
  bash -c "uv run --no-sync python <round>/code/bs_audit.py --modes model,a0-replay,eval-probe --out-dir <round>/runs/audit"

# 2) formal runs. The cadence resolved to every epoch (§6.1). Run-time artifacts
#    go under the git-ignored runs/ so the remote checkout stays mergeable.
#    The two runs are serial on GPU1 whenever GPU0 is under sustained external load
#    (§5 co-tenancy rule); on an idle pair of GPUs they can use res-gpu0/res-gpu1.
rr run res bs-w3 --pool res-gpu1 --result <round>/runs/W3 -- \
  bash -c "uv run --no-sync python <round>/code/bs_train.py --model W3 --seed 0 --device cuda \
  --eval-every 1 --out-dir <round>/runs/W3 --soft-seconds 14400 --hard-seconds 18000"
rr run res bs-w2 --pool res-gpu1 --result <round>/runs/W2 -- \
  bash -c "uv run --no-sync python <round>/code/bs_train.py --model W2 --seed 0 --device cuda \
  --eval-every 1 --out-dir <round>/runs/W2 --soft-seconds 14400 --hard-seconds 18000"

# 3) analysis (pull lands in .rr/pulled/<run_id>/result/..., copy into runs/)
rr pull bs-w2 ; rr pull bs-w3 ; rr jobs --json > <round>/analysis/rr_jobs.json
uv run --no-sync python <round>/code/bs_compare.py --rr-jobs <round>/analysis/rr_jobs.json
```

Actual execution of this round: `bs-preflight-20261107-171300`, `bs-audit-…-172954`,
`bs-w3-…-173626` (17:36→18:59 CST), `bs-w2-…-190526` (19:05→20:29 CST), all on
`res` (= `a100-2`, 2×A100-SXM4-40GB), both formal runs on `CUDA_VISIBLE_DEVICES=1`.

## Artifacts

```
analysis/model_scaling_audit.json   gates 1-2, 4-13 (params, widths, one-shot, no re-propagation,
                                    renumbering, normalisation, forward/backward, Test isolation)
analysis/a0_replay.json             gate 3 (historical epoch-40 A AP reproduced within 1e-4)
analysis/gpu_preflight.json         device/NVML/occupancy facts used for the scheduling decision
analysis/gpu0_occupancy.json        the 60 s GPU0 utilisation sampling behind the serial-on-GPU1 call
analysis/gpu0_occupancy_samples.csv the raw nvidia-smi samples
analysis/rr_jobs.json               raw rr job records (the ledger is derived from these)
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
* Formal runs must start from a clean local revision; a dirty tree makes
  `rr run` refuse. Run-time outputs therefore live under `runs/` (git-ignored)
  and `analysis/` is written locally when the round is finalised — otherwise the
  next deploy trips over untracked files that a commit later tracks.
* `eval-every` must match the cadence resolved in `PROTOCOL.md` §6.1 (1 or 2);
  the trainer records the cadence in `run.json` and refuses a formal run that
  disagrees with `bs_common.EVAL_CADENCE` (`formal_run: false` otherwise).
