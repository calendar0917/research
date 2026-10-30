# lcrc_lipo_screen_v1 — LCRC-A Lipophilicity screen

Second-round engineering package for the structural-sufficiency track: a controlled
Lipophilicity screen of LCRC-A (A) vs an S2-off ablation (B) vs GINE+global42 (C) vs
ECFP4+XGBoost (D). The v1 GPU stage halted on a Fit-index defect; the defect is fixed
and CPU re-verified, and the package is **ready for re-authorization**, not running.

- **Round status:** `AMENDMENT_CPU_VERIFIED` / `READY_FOR_REAUTHORIZATION`
- **GPU:** 0 h under the amended freeze; 0.0139 h used by the halted v1 stage
- **Amendment:** [`AMENDMENT-001-fit-batch-index.md`](AMENDMENT-001-fit-batch-index.md)
- **Halt record (preserved):** [`analysis/gpu_stage_halt.json`](analysis/gpu_stage_halt.json)
- **Protocol:** [`PROTOCOL.md`](PROTOCOL.md) (frozen design, unedited)
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
| `code/screen_gates.py` | CPU gates G0–G9 → `analysis/cpu_gates.json` |
| `code/run_screen.py` | CLI for every stage |
| `configs/lcrc_lipo_screen_v1.yaml` | frozen config (`FROZEN_PRE_GPU`) |
| `tests/` | 59 pytest checks (incl. full-Fit alignment) |
| `analysis/` | audit, parameter table, budget, freeze, gates, smoke, D predictions |
| `analysis/gpu_authorization.json` | committed researcher authorization (9 runs, 1.5 h) |
| `analysis/gpu_stage_halt.json` | failed formal run ledger, root cause, required amendment |
| `AMENDMENT-001-fit-batch-index.md` | index fix, new gate G9, before/after semantics |

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

Expected: `cpu-gates` ends at `READY_FOR_REAUTHORIZATION` (10/10 gates PASS), pytest
reports `59 passed`, and `status` reports `READY_FOR_REAUTHORIZATION` with
`gpu_authorization_matches_current_freeze = false`.

## Formal stage (v1 halted; amendment CPU-verified)

The v1 GPU stage was authorized and run through `rr`. Both attempted runs (`A_s0`,
`A_s1`) failed at the first optimizer batch:

```
ValueError: row index outside cache   # screen_training.py:165 -> lcrc_data.py:658
```

`screen_training.py` passed global split row ids (`0…4199`) to a collator that indexes
cache positions (`0…2351`); every epoch-1 batch had out-of-range ids, and in-range slots
paired the wrong label. All nine runs shared the loop; no `train.json`, soup, prediction
or manifest was written and no Monitor label was opened.

**Amendment-001** fixes the index semantics and adds the full-Fit alignment gate G9. The
amended freeze (sha256 `7fffba42…`) is CPU-verified: 10/10 gates PASS, 59 tests pass, new
smoke on the real path, and seeds 0/1/2 full-epoch coverage proven against an independent
`global_row_id → label` map. The v1 authorization pins the old freeze and is refused.

```bash
# still refused until a new authorization matches the amended freeze
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
| `analysis/cpu_gates.json` | 10/10 gates PASS (G9 added by amendment-001) |
| `analysis/smoke.json` | regenerated 3 CPU smoke curves on the fixed path (256 Fit × 12 epochs) |
| `analysis/predictions/D_r0..2.{npz,json}` | label-blind baseline predictions, not scored |
| `analysis/gpu_authorization.json` | researcher authorization (committed 2026-10-30) |
| `analysis/gpu_stage_halt.json` | v1 run ledger, failure trace, root cause |
| `analysis/smoke_invalidated_pre_amendment.json` | byte-identical old smoke, marked invalid |
| `AMENDMENT-001-fit-batch-index.md` | index fix, new gate, before/after semantics, verification |

## Important boundaries

- No Monitor, Reserved or official valid/test label has been read or scored. Only
  `fit_labels.py` (Fit 2352) may read labels; `monitor_labels.py` stays locked while no
  prediction manifest exists.
- Monitor504 is a development set previously used by SSRC; it is not a fresh blind test.
- CPU gates and smoke runs are engineering checks, not evidence of predictive quality.
  G5 uses a 256-row Fit prefix; the full-Fit index range is covered separately by G9,
  which was added after the v1 halt.
- Do not edit code/config after the amended freeze without a new amendment,
  regenerating `analysis/protocol_freeze.json` and obtaining a new authorization.
