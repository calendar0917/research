# LCRC-CPC v1 — reproduction guide

One new model (`LCRC-CPC`, seed 0, 40 epochs): the frozen LCRC-W2 seed-0
architecture plus one **one-shot rooted two-bond path residual** between
`drop(local(...))` and the unchanged W2 `post_local` readout.

Read `PROTOCOL.md` first; it is the frozen definition.  `REPORT.md` contains the
single formal result and the one research decision.

## Layout

```
code/cpc_common.py     frozen identities, recipe, parameter budget, decision rules
code/cpc_paths.py      exhaustive rooted two-bond enumeration, cache, collate, shuffle
code/cpc_model.py      CPC module and W2-CPC model, usage metrics, prediction
code/cpc_audit.py      structure audit + CPU/GPU preflight + non-additive witness
code/cpc_train.py      the one formal 40-epoch from-scratch training run
code/cpc_test.py       pre-label Test freeze then the one gated Test score pass
code/cpc_analyze.py    Validation-only mechanism interventions + decision/per-task
code/cpc_uncertainty.py post-freeze task and molecule bootstrap intervals
tests/                 path, bond, witness, renumbering, isolation, init, non-recursion
analysis/              audit, profile, curve, branch usage, freeze, scores, decision
runs/                  git-ignored checkpoints, logs and the 658 MB CPC path cache
```

## Local checks

```bash
cd tracks/structural-sufficiency/lcrc_pcba_cpc_v1
uv run pytest -q tests -m "not slow"
uv run python code/cpc_audit.py --mode structure --cache runs/cpc_cache
uv run python code/cpc_audit.py --mode preflight --device cpu
```

The structure cache is not committed.  It is rebuilt deterministically from the
frozen PCBA cache in ~44 s on this repository's CPU (437,929 molecules,
34,369,144 paths, 658 MB).

## Remote execution

From the repository root, with a clean committed worktree:

```bash
rr doctor res
rr deploy res
# 1) rebuild the label-free structure cache and audit it (CPU, ~1 min)
rr run res cpc-structure --result tracks/structural-sufficiency/lcrc_pcba_cpc_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_cpc_v1/code/cpc_audit.py \
     --mode structure --cache tracks/structural-sufficiency/lcrc_pcba_cpc_v1/runs/cpc_cache
# 2) GPU mechanism / numerical / 100-batch cost preflight
rr run res cpc-preflight --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_cpc_v1/analysis \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_cpc_v1/code/cpc_audit.py \
     --mode preflight --device cuda \
     --cache tracks/structural-sufficiency/lcrc_pcba_cpc_v1/runs/cpc_cache
rr pull cpc-preflight
# 3) the one formal run (hard wall 3.2 h)
rr run res cpc-train --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_pcba_cpc_v1/runs/LCRC_CPC \
  -- uv run python tracks/structural-sufficiency/lcrc_pcba_cpc_v1/code/cpc_train.py \
     --seed 0 --device cuda --epochs 40 \
     --out-dir tracks/structural-sufficiency/lcrc_pcba_cpc_v1/runs/LCRC_CPC \
     --soft-seconds 10000 --hard-seconds 11520
rr pull cpc-train
```

If a formal run is interrupted, resume only from its `checkpoint_last.pt` with
the same `--out-dir` and `--resume`; never change the recipe.

## Freeze, Test and analysis

Run these only after the formal run has status `COMPLETED`:

```bash
# Validation-only mechanism interventions (before any Test label read)
uv run python code/cpc_analyze.py --mode mechanism --device cuda

# Freeze Test logits (labels still NaN), then score once through the gated reader
uv run python code/cpc_test.py --mode freeze --device cuda
uv run python code/cpc_test.py --mode score --device cpu

# Per-task / support-stratified analysis and the one decision label
uv run python code/cpc_analyze.py --mode decision

# Descriptive task/molecule bootstrap intervals (no model changes)
uv run python code/cpc_uncertainty.py --row-reps 300 --task-reps 20000
```

Artifacts:

```
analysis/structure_cache.json       full-cache identity + count/endpoint audit
analysis/preflight.json             init parity, architecture/witness gates, GPU profile
analysis/gpu_profile.json           real 100-batch cost and 40-epoch projection
analysis/mechanism.json             Validation-only real/zero/pairing-shuffle
analysis/test_prediction_manifest.json  sealed Test logits, labels still unread
analysis/test_scores.json           official + independent Test AP after the freeze
analysis/per_task_analysis.json     per-task AP and support strata
analysis/decision.json              pre-registered label and observed numbers
analysis/uncertainty.json          task/molecule bootstrap intervals
analysis/gpu_ledger.json           all remote jobs and GPU allocation
analysis/report_tables.md          generated result tables
runs/LCRC_CPC/train_curve.csv       per-epoch loss/AP/timing/branch usage
runs/LCRC_CPC/branch_usage.json     per-epoch gate/token/path/gradient/weight stats
```

Test values are an exposed engineering comparison set, never a blind estimate
and never a model-selection signal.
