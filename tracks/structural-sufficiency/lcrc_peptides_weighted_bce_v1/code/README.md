# Execution order

All commands are run from the repository root.  The formal run must be deployed
through `rr` from a committed revision; the smoke run is engineering only and is
never analyzed.

```bash
# 1. Train-only weights (CPU; writes analysis/pos_weight.json)
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_compute_pos_weight.py

# 2. Loss/data gates (CPU; writes analysis/gates.json)
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_check.py
uv run pytest -q tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/tests

# 3. Minimal CUDA smoke (engineering only, one epoch)
rr run res lcrc-pepweighted-smoke --result tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/smoke_weighted_bce_seed0 -- \
  uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_p2_train.py \
  --device cuda --out-dir tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/smoke_weighted_bce_seed0 \
  --max-seconds 240 --smoke-epochs 1

# 4. The single formal run (240 epochs, 4200 s cap)
rr run res lcrc-pepweighted-p2-seed0 --result tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0 -- \
  uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_p2_train.py \
  --device cuda --out-dir tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0 \
  --max-seconds 4200

# 5. Pull, then build comparison tables and gates (local CPU)
rr pull lcrc-pepweighted-p2-seed0
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_analyze.py

# 6. Post-hoc authorized Test evaluation (only after a written user authorization)
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_test_eval.py \
  --device cpu --out-dir tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/runs/test_eval
uv run python tracks/structural-sufficiency/lcrc_peptides_weighted_bce_v1/code/wb_test_report.py
```

The Test entry point is intentionally separate from the training path.  It is
post-hoc only: checkpoints are frozen, no training or selection is performed,
and the original `run.json` keeps `test_evaluated = false`.

`REPORT.md` is generated from `analysis/run_facts.json` and the two CSV tables
after the formal run.
