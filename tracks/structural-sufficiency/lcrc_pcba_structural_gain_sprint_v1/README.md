# lcrc_pcba_structural_gain_sprint_v1

Pre-registered PCBA structural-gain sprint. See `PROTOCOL.md` (frozen before
the first formal run) and `REPORT.md` (results).

## Layout

```
code/sg_common.py      frozen arms, thresholds, LR schedule, paths
code/sg_model.py       C1-subclass arms C/F/M/V, init audit, AdamW restore
code/sg_checkpoint.py  warm-start audit, continuation payloads, atomic save
code/sg_data.py        PCBA loader, 40-epoch order plan, AP/BCE evaluation
code/sg_preflight.py   GPU correctness gates + per-arm speed smoke
code/sg_train.py       Phase 1 / 2 / 3 trainer (PinnedStager, fixed epochs)
code/sg_audit.py       CPU source/architecture audit, merges preflight
code/sg_analyze.py     Phase 1/2/3 promotion rules, GPU ledger, run registry
tests/                 CPU correctness tests
analysis/              audit and phase JSONs (Git tracked)
runs/                  checkpoints, curves, per-task artifacts (Git ignored)
```

## Reproduction

Local CPU tests:

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/tests
uv run python tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code/sg_audit.py \
  --preflight tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/analysis/preflight_C_F.json \
              tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/analysis/preflight_M_V.json
```

Remote (all execution goes through `rr`):

```bash
rr doctor res
rr deploy res
rr run res sgs-preflight-gpu0 --pool res-gpu0 --result analysis -- \
  uv run python -u tracks/.../code/sg_preflight.py --arms C,F --out tracks/.../analysis/preflight_C_F.json
rr run res sgs-preflight-gpu1 --pool res-gpu1 --result analysis -- \
  uv run python -u tracks/.../code/sg_preflight.py --arms M,V --out tracks/.../analysis/preflight_M_V.json
# Phase 1 (C/F on gpu0/gpu1, then M/V; same commit)
rr run res sgs-p1-C --pool res-gpu0 --result tracks/.../runs/phase1_C -- \
  timeout 2100s uv run python -u tracks/.../code/sg_train.py --phase 1 --arm C --out-dir .../runs/phase1_C --soft-seconds 1800 --hard-seconds 2000
# Phase 2 (only if Phase 1 promotes a candidate)
rr run res sgs-p2-C --pool res-gpu0 --result tracks/.../runs/phase2_C -- \
  timeout 4000s uv run python -u tracks/.../code/sg_train.py --phase 2 --arm C \
  --resume .../runs/phase1_C/checkpoint_epoch12.pt --out-dir .../runs/phase2_C --soft-seconds 3400 --hard-seconds 3700
# Phase 3 (only if Phase 2 passes; 4 runs, two seeds)
```

Analysis:

```bash
uv run python tracks/.../code/sg_analyze.py phase1
uv run python tracks/.../code/sg_analyze.py phase2   # only if Phase 1 promoted
uv run python tracks/.../code/sg_analyze.py phase3   # only if Phase 2 passed
rr jobs --json > /tmp/rr_jobs.json
uv run python tracks/.../code/sg_analyze.py registry --rr-jobs /tmp/rr_jobs.json
uv run python tracks/.../code/sg_analyze.py ledger
```

Official Test targets are never scored or used for selection in this round. The frozen loader audits that the Test label slice is entirely NaN; no Test logit, AP or BCE is computed or saved.
