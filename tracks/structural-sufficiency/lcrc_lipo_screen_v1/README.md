# lcrc_lipo_screen_v1 — LCRC-A Lipophilicity screen

Second-round engineering package for the structural-sufficiency track: a controlled
Lipophilicity screen of LCRC-A (A) vs an S2-off ablation (B) vs GINE+global42 (C) vs
ECFP4+XGBoost (D). This package is **frozen and CPU-validated but has not been run on
GPU**.

- **Round status:** `HALTED_IMPLEMENTATION_DEFECT` (formal GPU stage attempted 2026-10-30)
- **GPU:** authorized for 1.5 A100 GPU-hours; 0.0139 h used; 0/9 trainings completed
- **Halt record:** [`analysis/gpu_stage_halt.json`](analysis/gpu_stage_halt.json)
- **Protocol:** [`PROTOCOL.md`](PROTOCOL.md) (frozen, unedited after the freeze)
- **Report:** [`REPORT.md`](REPORT.md) — no performance result
- **Decision:** [`DECISION.md`](DECISION.md) — no scientific verdict

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
| `analysis/gpu_authorization.json` | committed researcher authorization (9 runs, 1.5 h) |
| `analysis/gpu_stage_halt.json` | failed formal run ledger, root cause, required amendment |

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
install it, so arm D must be run locally (CPU). Locally `torch.cuda.is_available()` is
`False`; the formal GPU stage ran remotely on `res` (2× A100-SXM4-40GB, driver
550.163.01, torch 2.5.1+cu124) through `rr`, and used 0.0139 GPU-hours before halting.

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
`52 passed, 1 failed` (the pre-GPU-state test asserts the authorization file is absent
and is superseded by the 2026-10-30 authorization; it was not edited), and `status`
reports the halted stage.

## Formal stage (attempted 2026-10-30; halted)

The researcher authorized the nine runs, and the package was deployed and run through
`rr`. Both attempted runs (`A_s0`, `A_s1`) failed at the first optimizer batch:

```
ValueError: row index outside cache   # screen_training.py:165 -> lcrc_data.py:658
```

`screen_training.py` passes global split row ids (`0…4199`) to a collator that indexes
cache positions (`0…2351`); every epoch-1 batch contains out-of-range ids
(1162/2352 slots, 19/19 batches), and in-range slots would pair the wrong label
(1188/1190). All nine runs share the loop. No `train.json`, soup, prediction or
manifest was written; Monitor labels were never opened.

Frozen code was **not** patched. A continuation requires a recorded amendment, a
full-Fit first-epoch alignment gate, a regenerated `analysis/protocol_freeze.json` and a
new authorization. See `DECISION.md` and `analysis/gpu_stage_halt.json`.

```bash
# still refused / still failing under the current frozen hashes
python code/run_screen.py train --arm A --seed 0 --device cuda
```

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
| `analysis/gpu_authorization.json` | researcher authorization (committed 2026-10-30) |
| `analysis/gpu_stage_halt.json` | run ledger, failure trace, root cause, required amendment |

## Important boundaries

- No Monitor, Reserved or official valid/test label has been read or scored. Only
  `fit_labels.py` (Fit 2352) may read labels; `monitor_labels.py` stays locked while no
  prediction manifest exists.
- Monitor504 is a development set previously used by SSRC; it is not a fresh blind test.
- CPU gates and smoke runs are engineering checks, not evidence of predictive quality.
  In particular, gate G5 used a 256-row Fit prefix and did not exercise the full-Fit
  index range; that gap is disclosed in `REPORT.md` §2.3.
- Do not edit code/config after the freeze without regenerating
  `analysis/protocol_freeze.json` and obtaining a new authorization.
