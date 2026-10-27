# SCSB v1 — support-conditioned structural backoff

Round: `zinc_support_conditioned_backoff_v1`
Protocol: `ssrc-zinc-support-conditioned-backoff-v1`
Frozen config: `configs/scb_v1.json`
Full pre-registration: `PROTOCOL.md`

## Question

Does a per-bond, label-free radius-2 training-support weight `alpha = s/(s+4)`,
applied as `Wl -> alpha*Wl + (1-alpha)*Wg` (and the same for `Vl/Vg`) before the
frozen DCC-L local composer, improve raw-y MAE on Fit-unseen structures?  The
placement control **P** permutes each molecule's alpha vector across its bonds,
preserving the per-molecule alpha distribution exactly.

## Status

- Phase 0 CPU gate: **PASS (13/13)** — `analysis/cpu_checks.json`.
  Includes leave-one-out support, Fit-only table, permutation invariance,
  bitwise L reproduction (max |Δ| 1.9e-6 vs the frozen DCB prediction),
  alpha=0/1 synthetic endpoints, gradients, alpha distribution, group counts,
  cache/checkpoint hashes, no-test scan, and a one-epoch training-loop match to
  the frozen DCB L curve (online/monitor/lr/step).
- Phase 1: **not yet run** (S/P two seeds, 47,400 updates each; L is the frozen
  DCB Fit8k checkpoint, zero new GPU hours).
- Phase 2: locked until Phase 1 PASS.  Official test is never loaded.

## Phase 0

```bash
uv run python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_checks.py
```

## Phase 1 (after commit + `rr deploy res`)

```bash
# S/P, two seeds, one job per GPU pool; inspect with `rr jobs/status/logs`.
rr run res scb-S-s0 --pool res-gpu0 --gpus 1 --time 2:00:00 --result tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/runs -- \
  python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_train.py \
  --arm S --fit fit8k --seed 0 --device cuda:0
rr run res scb-P-s1 --pool res-gpu1 --gpus 1 --time 2:00:00 --result tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/runs -- \
  python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_train.py \
  --arm P --fit fit8k --seed 1 --device cuda:0
# ... S-s1 / P-s0 likewise, then one frozen Locked scoring read:
rr run res scb-score-p1 --pool res-local --cpus 8 --time 1:00:00 --result tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/analysis -- \
  python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_score.py --device cpu
```

Then locally:

```bash
uv run python tracks/structural-sufficiency/zinc_support_conditioned_backoff_v1/code/scb_analyze.py
```

## Files

- `configs/scb_v1.json` — frozen protocol, gates, budget, no-rescue list.
- `code/scb_common.py` — support table, alpha, groups, bootstraps, no-test scan.
- `code/scb_model.py` — `SCBDualChannel` (same parameters as DCC-L, one mixed input).
- `code/scb_checks.py` — Phase 0 CPU gate.
- `code/scb_train.py` — one arm/seed, frozen recipe, 47,400 updates.
- `code/scb_score.py` — one frozen Locked read, scores all arms/checkpoints.
- `code/scb_analyze.py` — paired deltas, bootstraps, pre-registered decision.
- `analysis/` — Phase 0 evidence; Phase 1 scores/decision/report land here.
