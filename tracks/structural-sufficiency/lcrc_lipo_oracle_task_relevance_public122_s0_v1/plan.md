# Plan — oracle task relevance round (public122, seed 0)

Protocol: `PROTOCOL.md`. Source round: `lcrc_lipo_multi_motif_mechanism_public122_v1`
(protocol `0273cad6`, report `a5febe96`). Only a committed revision is deployed.

## Stages

### Stage A — local CPU (no GPU, commit before any training)
* `code/run.py --stages partA` → `analysis/partA_clarifications.json`, `analysis/partA_recovery_pairs.csv`
  (A1 coverage/claim matrix, A2 same-population three tables, A3 conditional scalar/S2 recovery).
* `code/run.py --stages oracle` → `analysis/oracle_labels.csv`, `analysis/{wc_logp,molwt}_oracle.csv`,
  `analysis/cf_isolation.csv`, `analysis/{test,valid}_train_duplicates.csv`,
  `analysis/oracle_manifest.json`, `analysis/freeze_oracle.json`.
* `uv run pytest -q …/tests` (focused checks).

### Stage B — remote `res`, one GPU per task (exactly two trainings)
```
rr doctor res
rr deploy res
rr run res --pool res-gpu0 --experiment lipo-oracle-wclogp-s0 \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/analysis \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/runs \
  -- python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
     --stages train,evaluate --task wc_logp --device cuda:0
rr run res --pool res-gpu1 --experiment lipo-oracle-molwt-s0 \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/analysis \
  --result tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/runs \
  -- python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py \
     --stages train,evaluate --task molwt --device cuda:1
rr jobs ; rr status <exp> ; rr logs <exp> ; rr pull <exp>
```
One process per card, `res-2` unused, other jobs untouched. Planned ≤ 0.2 GPU·h
(soft 0.5, hard 1.0).

### Stage C — local CPU
* `code/run.py --stages analysis,figures` → `analysis/analysis_checks.json`,
  `analysis/summary_tables.json`, the three figures, then `REPORT.md`.

## Deliverables
PROTOCOL, config, code, tests, REPORT; analysis with coverage/three tables/
recovery, oracle values + cache checks, training/CF isolation, both tasks' full
predictions + metrics + intervals + runtime; three main figures.
