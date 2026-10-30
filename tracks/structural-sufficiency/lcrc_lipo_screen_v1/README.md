# lcrc_lipo_screen_v1 — LCRC-A Lipophilicity screen

Second-round engineering package for the structural-sufficiency track: a controlled
Lipophilicity screen of LCRC-A (A) vs an S2-off ablation (B) vs GINE+global42 (C) vs
ECFP4+XGBoost (D). The v1 GPU stage halted on a Fit-index defect; the defect was fixed,
CPU re-verified, re-authorized and executed. The formal stage is complete.

- **Round status:** `FORMAL_GPU_COMPLETE_MONITOR_EVALUATED`
- **Scientific result:** `CASE_1_S2_INCREMENT_SUPPORTED` (frozen rule; A RMSE 0.91002,
  B 1.04912, C 0.92284, D 0.97439 on Monitor504)
- **GPU:** 0.5400 h in the amended stage; 0.0139 h historical; 0.5539 h cumulative of
  the 1.5 h round cap
- **Amendment:** [`AMENDMENT-001-fit-batch-index.md`](AMENDMENT-001-fit-batch-index.md)
- **Halt record (preserved):** [`analysis/gpu_stage_halt.json`](analysis/gpu_stage_halt.json)
- **Protocol:** [`PROTOCOL.md`](PROTOCOL.md) (frozen design, unedited)
- **Report:** [`REPORT.md`](REPORT.md) — full results, hashes and interpretation boundaries
- **Decision:** [`DECISION.md`](DECISION.md) — `CASE_1_S2_INCREMENT_SUPPORTED`, with caveats

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
| `analysis/` | audit, parameter table, budget, freeze, gates, smoke, GPU ledger, predictions, Monitor results |
| `analysis/gpu_authorization.json` | active amendment-001 researcher authorization (9 runs, 1.4861 h) |
| `analysis/gpu_authorization_v1_superseded_pre_amendment.json` | byte-identical v1 authorization (superseded, preserved) |
| `analysis/gpu_authorization_history.json` | hashes/commits of both authorization records |
| `analysis/gpu_ledger.json` | per-run GPU wall time, rr ids, train/soup hashes, totals |
| `analysis/gpu_stage_halt.json` | v1 failed-run ledger, root cause, required amendment |
| `analysis/predictions/MANIFEST.json` | frozen 12-key prediction registry + all hashes |
| `analysis/monitor_{metrics,paired,diagnostics}.json` | the single frozen Monitor evaluation |
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

Expected: `cpu-gates` ends at `READY_FOR_REAUTHORIZATION` (10/10 gates PASS; the gate
label is the historical gate-set state), and `status` reports
`state = AUTHORIZED_UNDER_CURRENT_FREEZE`, 12 prediction files and
`monitor_evaluated = true` after the completed stage. The full pytest suite passed
59/59 on the pre-execution snapshot; after the stage, four checks that assert the
pre-execution state (no manifest, no training artifacts) fail by design and must not be
re-run — see `REPORT.md` §3.4. Use the prediction-manifest verification for
post-execution integrity.

## Formal stage (v1 halted, then completed under amendment-001)

The v1 GPU stage was authorized and run through `rr`. Both attempted runs (`A_s0`,
`A_s1`) failed at the first optimizer batch:

```
ValueError: row index outside cache   # screen_training.py:165 -> lcrc_data.py:658
```

`screen_training.py` passed global split row ids (`0…4199`) to a collator that indexes
cache positions (`0…2351`); every epoch-1 batch had out-of-range ids, and in-range slots
paired the wrong label. All nine runs shared the loop; no `train.json`, soup, prediction
or manifest was written and no Monitor label was opened.

**Amendment-001** fixed the index semantics and added the full-Fit alignment gate G9.
The amended freeze (sha256 `7fffba42…`) was CPU-verified (10/10 gates, 59 tests, full-Fit
alignment for seeds 0/1/2), then re-authorized for 1.4861 A100 GPU-hours. The formal
stage subsequently ran on `res` (2× A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124):

- `A_s0` passed the stage-1 gate (360 epochs / 6840 updates, all 360 epoch fingerprints
  independently reproduced); the other eight runs then completed the frozen recipe.
- 9/9 runs completed; 3240/3240 replayed epoch fingerprints matched; no non-finite
  loss/gradient; last-5 soups saved and hash-verified; no Monitor/Reserved/official label
  was read during training.
- Nine label-blind neural predictions plus the existing `D_r0..D_r2` were frozen
  (`MANIFEST.json` sha256 `ba8ad0be…`), then one Monitor evaluation was run.
- Frozen result: `CASE_1_S2_INCREMENT_SUPPORTED` (A 0.91002, B 1.04912, C 0.92284,
  D 0.97439 RMSE). Details and caveats: `REPORT.md`, `DECISION.md`.
- GPU: 0.5400 h this stage, 0.5539 h cumulative including the 0.0139 h v1 halt.

The v1 authorization (`48cb9a28…`) is preserved byte-identically and remains refused
against the amended freeze; the active authorization is `a2183bdf…`.

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
| `analysis/predictions/D_r0..2.{npz,json}` | label-blind baseline predictions (hashes unchanged) |
| `analysis/predictions/{A,B,C}_s{0,1,2}.{npz,json}` | nine label-blind neural predictions |
| `analysis/predictions/MANIFEST.json` | frozen 12-key registry, row order and all hashes |
| `analysis/monitor_{metrics,paired,diagnostics}.json` | the single post-freeze Monitor evaluation |
| `analysis/gpu_ledger.json` | per-run GPU time, rr ids, train/soup hashes, budget totals |
| `analysis/gpu_authorization.json` | active amendment-001 authorization (1.4861 h, nine runs) |
| `analysis/gpu_authorization_v1_superseded_pre_amendment.json` | preserved v1 authorization bytes |
| `analysis/gpu_stage_halt.json` | v1 run ledger, failure trace, root cause |
| `analysis/smoke_invalidated_pre_amendment.json` | byte-identical old smoke, marked invalid |
| `AMENDMENT-001-fit-batch-index.md` | index fix, new gate, before/after semantics, verification |

## Important boundaries

- Monitor504 was read **exactly once**, after all 12 predictions were frozen. Reserved504
  and official valid/test labels were never read or scored. The evaluation audit
  constructed only the 504 Monitor labels (`unrequested_values_constructed=0`).
- Monitor504 is a development set previously used by SSRC; it is not a fresh blind test.
- A vs B is an S2-channel ablation with equal nominal but unequal effective capacity
  (B: 176,514 effective parameters). The measured increment cannot be attributed solely
  to S2 information.
- A vs C and A vs D are `INCONCLUSIVE` on the primary RMSE rule (CIs cross zero);
  `CASE_1` does not assert A beats either reference.
- CPU gates and smoke runs are engineering checks, not evidence of predictive quality.
  G5 uses a 256-row Fit prefix; the full-Fit index range is covered separately by G9.
- Do not edit code/config after the amended freeze without a new amendment,
  regenerating `analysis/protocol_freeze.json` and obtaining a new authorization.
- The round is stopped: no SSL, AQSOL, extra ablation, Reserved/official evaluation or
  further training is authorized.
