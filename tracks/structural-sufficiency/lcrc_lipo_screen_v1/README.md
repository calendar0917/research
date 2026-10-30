# lcrc_lipo_screen_v1 — LCRC-A Lipophilicity screen

Second-round engineering package for the structural-sufficiency track: a controlled
Lipophilicity screen of LCRC-A (A) vs an S2-off ablation (B) vs GINE+global42 (C) vs
ECFP4+XGBoost (D). This package is **frozen and CPU-validated but has not been run on
GPU**.

- **Round status:** `READY_FOR_GPU_AUTHORIZATION`
- **GPU authorization:** `NOT_AUTHORIZED` (no `analysis/gpu_authorization.json`)
- **Protocol:** [`PROTOCOL.md`](PROTOCOL.md)
- **Report:** [`REPORT.md`](REPORT.md)
- **Decision:** [`DECISION.md`](DECISION.md)

## What exists

| path | contents |
|---|---|
| `code/screen_common.py` | frozen constants, hashes, config validation, protocol freeze |
| `code/access_guard.py` | runtime deny-list + one-shot label grant |
| `code/fit_labels.py` | selective Fit-only label scanner (never materialises Monitor values) |
| `code/monitor_labels.py` | manifest-gated Monitor label entry point |
| `code/screen_data.py` | cache loading, collation, batch order, target normalisation, label-blind prediction |
| `code/screen_models.py` | LCRC-A, S2-off controller, GINE, parameter/soup helpers |
| `code/screen_training.py` | frozen recipe, snapshots, last-5 soup, run manifest |
| `code/screen_predict.py` | label-blind prediction + hashed prediction freeze |
| `code/baseline_ecfp_xgb.py` | frozen ECFP4 + XGBoost baseline |
| `code/evaluate.py` | frozen metrics, bootstrap, paired tests, decision rules |
| `code/screen_gates.py` | CPU gates G0–G8 → `analysis/cpu_gates.json` |
| `code/run_screen.py` | CLI for every stage |
| `configs/lcrc_lipo_screen_v1.yaml` | frozen config (`FROZEN_PRE_GPU`) |
| `tests/` | 53 pytest checks |
| `analysis/` | audit, parameter table, budget, freeze, gates, smoke, D predictions |

`analysis/runs/` (training artifacts) is git-ignored. Everything else in `analysis/`
is committed and hash-referenced.

## Environment

Local CPU environment (used for every command below):

```bash
cd /home/calendar/code/research
uv run --group lipo python -c "import torch, torch_geometric, rdkit, xgboost; print(torch.__version__, torch_geometric.__version__, rdkit.__version__, xgboost.__version__)"
# 2.5.1+cu124 2.6.1 2026.09.1 2.1.3
```

RDKit lives in the local-only `lipo` dependency group. The remote A100 hosts do not
install it, so arm D must be run locally (CPU). `torch.cuda.is_available()` is `False`
here; no GPU was used.

All commands are run from the round directory:

```bash
cd tracks/structural-sufficiency/lcrc_lipo_screen_v1
```

## Reproduce the pre-GPU state

```bash
# 1. data audit: hashes, row alignment, no labels read
uv run --project /home/calendar/code/research --group lipo \
  python code/run_screen.py verify-data

# 2. Fit-only target normalisation, parameter table, budget, protocol freeze
uv run --project /home/calendar/code/research --group lipo \
  python code/run_screen.py freeze-protocol

# 3. CPU correctness gates (writes analysis/cpu_gates.json)
uv run --project /home/calendar/code/research --group lipo \
  python code/run_screen.py cpu-gates

# 4. pytest suite
uv run --project /home/calendar/code/research --group lipo \
  python -m pytest tests -q

# 5. arm D: label-blind ECFP4/XGBoost predictions for Monitor504
uv run --project /home/calendar/code/research --group lipo \
  python code/run_screen.py baseline-ecfp

# 6. current status
uv run --project /home/calendar/code/research --group lipo \
  python code/run_screen.py status
```

Expected: `cpu-gates` ends at `READY_FOR_GPU_AUTHORIZATION`, pytest reports
`53 passed`, and `status` reports `"state": "READY_FOR_GPU_AUTHORIZATION"`.

## Formal stage (blocked)

```bash
# These commands only work after the researcher creates a valid
# analysis/gpu_authorization.json and commits a clean worktree.
python code/run_screen.py train --arm A --seed 0 --device cuda
python code/run_screen.py predict-nn --arm A --seed 0 --device cuda
...
python code/run_screen.py freeze-predictions
python code/run_screen.py evaluate
```

`train` refuses to run if any code/config hash has drifted, if the worktree is dirty,
if the run is not listed in the authorization, or if `max_gpu_hours` exceeds the frozen
hard cap. See `PROTOCOL.md` §9 for the authorization schema. Use the repository's
`rr` remote runner for GPU execution; do not start a manual SSH session.

## Frozen recipe (summary)

- Fit 2352 only; seeds 0/1/2; AdamW; LR 1e-3 → 1e-5 per-step cosine; wd 1e-5;
  batch 128; 360 epochs = 19 steps/epoch = **6840 updates/run**; clip 5.0; dropout 0;
  MSE on Fit-standardised target; checkpoint = mean of last 5 epoch-end state dicts.
- No early stopping, no Monitor-based selection, no arm-specific epochs, no SSL.
- A/B: 247,106 nominal parameters; B zeroes the 200-d S2 block of the LocalMLP input
  (`[320:520)`) in train and eval and has 176,514 effective parameters. This unequal
  effective capacity is disclosed.
- C: GINE, 3 layers, hidden 170, 245,305 parameters, real physical-bond adjacency.
- D: frozen BJOC ECFP4 (Morgan r=2, 2048 bits) + XGBoost, refit on Fit2352.

## Frozen evaluation

Primary raw-scale RMSE; MAE and paired per-molecule metrics secondary; 2000-resample
molecule bootstrap (seed 0); all seeds reported with mean/spread; materiality threshold
0.03. Predictions for all 12 arm×seed keys must be frozen and hashed before any Monitor
label is read. The decision output is `CASE_1/2/3` under the rules in `PROTOCOL.md` §6.

## Current artifacts

| artifact | note |
|---|---|
| `analysis/protocol_freeze.json` | config + code + artifact hashes, recipe, registry |
| `analysis/norm_target.json` | Fit-only mean/scale |
| `analysis/parameters.json` | A/B/C parameter and GINE width search tables |
| `analysis/budget_estimate.json` | 0.734 GPU-h estimate vs 1.5 h hard cap |
| `analysis/cpu_gates.json` | 9/9 gates PASS |
| `analysis/smoke.json` | 3 CPU smoke curves (256 Fit × 12 epochs), no Monitor |
| `analysis/predictions/D_r0..2.{npz,json}` | label-blind baseline predictions, not scored |

## Important boundaries

- No Monitor, Reserved or official valid/test label has been read or scored in this
  package. Only `fit_labels.py` (Fit 2352) and, after a completed prediction freeze,
  `monitor_labels.py` may read the label artifact.
- Monitor504 is a development set previously used by SSRC; it is not a fresh blind test.
- CPU gates and smoke runs are engineering checks, not evidence of predictive quality.
- Do not edit code/config after the freeze without regenerating
  `analysis/protocol_freeze.json`; formal runs will refuse stale hashes.
