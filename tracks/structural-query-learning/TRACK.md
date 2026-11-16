# Track: structural-query-learning

## Charter

Ask how far an **explicit, executable structural query language** plus a plain
tabular reader can go on small molecular graph-property tasks, **without**
iterative hidden-state message passing.

The central object is a discrete query: a Boolean node predicate built from raw
OGB atom/bond attributes and unlabeled structural primitives (`degree`,
`min_ring`, `in_cycle`) with three operators — `AND`, `NBR(P,b,m)` (counting
neighbourhood modality) and `BRANCH(P,Q,b1,b2)` (two *distinct* neighbours).
Queries are never learned embeddings; they are ASTs that can be printed,
counted and replayed.

Two questions are kept separate and reported separately:

1. **Representation**: does supervision-guided query discovery add anything over
   a fixed query bank and over a complexity-matched random bank with the same
   reader and the same raw graph?
2. **Performance**: does this independent predictor come close to strong
   classical (ECFP4+descriptors+XGBoost), neural (GINE) and repository-native
   (C1 / NO_BOND_MEAN) references retrained on the identical official split?

## Non-goals of this track

* No LCRC backbone, no residual head on top of another model, no distillation.
* No GNN/Transformer encoder inside the main predictor (the GINE reference is a
  performance reference only).
* No claim of chemical causality from an exported rule, and no claim of
  state-of-the-art performance from two small tasks and one seed.

## Layout

```
tracks/structural-query-learning/
  TRACK.md                 this charter
  STATE.yaml               current pointer file
  configs/                 frozen run configuration
  protocols/               comparability rules (dataset/split/metric/test policy)
  studies/                 scientific intent
  code/                    sq_data, sq_ast, sq_bank, sq_search, sq_xgb, sq_gine,
                           sq_lcrc, sq_explain, sq_runner, sq_report, sq_common
  tests/                   correctness tests for the language, data and replay
  analysis/                protocol freeze + preparation notes
  data/                    committed LABEL-FREE caches (raw graph, ECFP, lipo_v1 raw)
  cache/                   git-ignored scratch (models, graph caches)
  runs/                    git-ignored run store
  results/sq-pilot-small-v1/  durable per-dataset artefacts, tables, REPORT/DECISION
```

## Entry points

```bash
# local preparation (needs the official OGB data + RDKit)
uv run python tracks/structural-query-learning/code/sq_prepare.py

# correctness tests
uv run pytest -q tracks/structural-query-learning/tests

# remote workers (one per GPU, via rr)
rr run res sq-esol-s0 --pool res-gpu0 --gpus 1 --cpus 4 \
  --result tracks/structural-query-learning/results \
  -- env RESEARCH_TRACK_ROOT=tracks/structural-query-learning \
         RESEARCH_EXTRA_RUNNER_PATHS=tracks/structural-query-learning/code \
         RESEARCH_EXTRA_RUNNERS=sq_runner \
         uv run research run sq-pilot-small-v1 --study sq-pilot-small-v1 \
         --mode screen --set stage=dataset_worker --set dataset=ogbg-molesol

# aggregation (CPU, after both workers finish)
uv run research run sq-pilot-small-v1 --study sq-pilot-small-v1 --mode screen \
  --set stage=aggregate
```
