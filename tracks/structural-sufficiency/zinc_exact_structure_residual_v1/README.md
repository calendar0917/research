# zinc_exact_structure_residual_v1

Bounded development round on top of the frozen P3 system (BJOC-Null +
train-only masked SSL + R3, 413,505 params): add an **exact rooted-patch type
residual** to the atom embedding and ask whether it improves generalisation,
and whether a **real r2→r1 parent-sharing prior** beats an independent per-type
prior at matched capacity.

**Status: CLOSED — negative, phase 2 not triggered.** See `REPORT.md` for the
result and `AMENDMENT.md` for post-freeze issues. In short: B 0.07972, E@1.0
0.09907, H@1.0 0.09453 (seed-mean valid MAE); H consistently beats E by ~0.0045
but both lose to B by 0.015–0.031, and the branch captures the prediction rather
than acting as a small correction.

Official test is never loaded. Official valid is the exposed development set and
its labels are the single existing read from the recipe-tuning round, reused
read-only.

## Layout

```
PROTOCOL.md                  frozen protocol (pre-GPU)
configs/                     frozen config / run matrix / decision rule
code/
  esr_data.py                corrected r1/r2 patch types, train-only vocabulary
  esr_models.py              P3 + per-type low-rank residual, E/H priors
  esr_runner.py              prep -> gates -> smoke -> freeze -> train -> predict
                             -> score -> analyze; ssl / chain / phase2 helpers
  esr_score.py               post-freeze score / analyze over frozen artifacts
  esr_diag.py                post-hoc diagnostics (alignment, zero probe, sweep)
  tests/test_esr.py          lightweight CPU tests (no data, no GPU)
cache/                       git-ignored prep artifacts (vocab, types, scalars)
runs/<run>_s<seed>/          train.json, soup.pt, curve.csv
predictions/                 label-blind valid predictions + hash freeze
analysis/                    metrics, contrasts, decision, groups, mechanism
curves/                      per-run development curves
ssl_pretrain/                (phase 2 only) same-recipe SSL for seeds 2/3
gates.json / smoke.json / protocol_freeze.json / gpu_ledger.json
```

## Commands

Local CPU (no GPU, no valid labels):

```bash
uv run python tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/esr_runner.py \
  --device cpu prep
uv run python tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/esr_runner.py \
  --device cpu gates
uv run pytest -q tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/tests
```

Remote through `rr` (committed revision only):

```bash
rr run res esr-prep   --pool res-local --gpus 0 --time 0:20:00 \
  --result tracks/structural-sufficiency/zinc_exact_structure_residual_v1/cache \
  -- python tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/esr_runner.py --device cpu prep
# ... gates, smoke on res-gpu0, freeze ...
# scoring (post-freeze): python .../esr_score.py score --phase 1 && analyze
rr run res esr-p1-gpu0 --pool res-gpu0 --gpus 1 --time 2:00:00 \
  --result tracks/structural-sufficiency/zinc_exact_structure_residual_v1/runs \
  -- python .../esr_runner.py --device cuda chain --seed 0 --configs B,E_eta0.1,E_eta1,H_eta0.1,H_eta1
```

Every run is a fresh `rr` job; the runner is idempotent and skips a completed
`train.json` + `soup.pt` pair (use `--force` only after a disclosed failure).

## Boundary

Not a blind test, not a confirmation, not an official-test evaluation, not a
novelty claim. The valid split is reused for bounded configuration selection
(two `eta` levels per family), exactly as pre-registered.
