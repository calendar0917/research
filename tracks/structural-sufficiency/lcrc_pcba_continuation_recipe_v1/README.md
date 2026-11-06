# LCRC PCBA continuation-recipe round v1

This round tests whether the existing PCBA A/G/V models still improve after the
40-epoch cosine endpoint, and whether a warm learning-rate restart (Recipe R)
beats keeping the low fixed LR (Recipe L).  It then performs one
Validation-frozen official Test evaluation.

## Execution status

Completed on 2026-11-06.  Six Stage-1 runs (A/G/V × L/R, 40→60) and one
Stage-2 run (V-R, 60→80) finished.  A and G stopped at epoch 60 under the
frozen eligibility rule; only V-R reached epoch 80.  Validation selection:
A = epoch40, G = G-L60, V = V-R80.  Official Test was frozen first and scored
once afterwards.  Total GPU use: **5.144 GPU-hours** (target ≤7, hard ≤8; includes a superseded Stage-0 preflight).
See **`REPORT.md`** for the results and the exact answer to Q1–Q8.

## Layout

```
PROTOCOL.md                 frozen protocol (before formal training)
README.md                   this file
code/cc_common.py           frozen source identities, schedules, decisions
code/cc_data.py             frozen data loader, epoch41-80 plan, Test label gate
code/cc_restore.py          strict model/AdamW/RNG restore and checkpoints
code/cc_preflight.py        Stage-0 CPU audit + GPU replay/paired/smoke
code/cc_train.py            Stage-1 (40->60) and Stage-2 (60->80) continuation
code/cc_test.py             Test prediction freeze and post-freeze scoring
code/cc_analyze.py          fixed-epoch comparison, selection, ledger
tests/                      CPU schedule/plan tests and source-restore test
analysis/                   committed audit/comparison/selection/Test JSONs
runs/                       local per-run curves/checkpoints (Git-ignored)
REPORT.md                   final report and Q1–Q8 answers
```

## Reproduce

```bash
# 1. Stage-0 CPU audit, schedules, plan, paired CPU probe
uv run python tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1/code/cc_preflight.py audit

# 2. Stage-0 GPU replay / paired first step / 20-step smoke
uv run python tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1/code/cc_preflight.py gpu --device cuda

# 3. Stage 1, one command per model/recipe (run through rr on res)
python .../cc_train.py --model A --recipe L --start-epoch 41 --final-epoch 60 \
  --resume .../phaseC_seed0_A/checkpoint_epoch40.pt \
  --out-dir .../runs/stage1_A_L --soft-seconds 3300 --hard-seconds 4200

# 4. Analyze / decide Stage 2
uv run python .../cc_analyze.py stage1

# 5. Stage 2 only for eligible models, from their epoch-60 checkpoint
#    In this round only V-R was eligible.
python .../cc_train.py --model V --recipe R --start-epoch 61 --final-epoch 80 \
  --resume .../runs/stage1_V_R/checkpoint_epoch60.pt \
  --out-dir .../runs/stage2_V_R --soft-seconds 3300 --hard-seconds 4200

# 6. Final comparison, Validation selection, ledger
uv run python .../cc_analyze.py all

# 7. Freeze Test predictions, pull them, then score locally
python .../cc_test.py freeze --device cuda \
  --selection .../analysis/validation_selection.json \
  --out .../runs/test_freeze/test_prediction_freeze.json \
  --predictions-dir .../runs/test_freeze/test_predictions
uv run python .../cc_test.py score \
  --freeze .../analysis/test_prediction_freeze.json \
  --predictions-dir .../runs/test_freeze/test_predictions
```

Formal runs are launched only through `rr` from a clean committed revision.
The official Test label file is opened only by `cc_test.py score`, after the
prediction-freeze manifest exists and its seal verifies.

## Important boundaries

- Seed 0 only; no architecture search; no extra seeds.
- Recipe R is a learning-rate restart from epoch 40, not an 80-epoch
  from-scratch control.
- Epoch 50 and 70 are diagnostic evaluation points, never final candidates.
- Test is read once, after training and Validation selection; Test results are
  descriptive and cannot change any model or recipe.
