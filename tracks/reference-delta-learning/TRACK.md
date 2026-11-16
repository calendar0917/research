# Track: reference-delta-learning

## Charter

Ask exactly one question: on the frozen official OGB `ogbg-molesol` /
`ogbg-mollipo` split, does a **joint pairwise delta learner** — one model that
reads two static molecular inputs and predicts their property difference, then
assembles an absolute value from Train references at inference — beat

* **D** direct absolute regression on the same features,
* **N** a Train-neighbour label predictor (no training), and
* **R** the same direct model plus a Train-neighbour residual correction?

The prediction rule of the new arm is

```
delta(G, R) = sigma/2 * [ h(z(G,R)) - h(z(R,G)) ],   z(i,j) = [x_i, x_j, x_i - x_j]
C(G)        = sum_j w_j(G) * [ y_j + delta(G, R_j) ]
```

with `w_j` a fixed Tanimoto/softmax weighting over legal Train references.  The
antisymmetrisation is part of the *definition of inference*, not a post-hoc
trick, and `delta` is a raw-unit difference: it is multiplied by `sigma` and
never gets `mu` added.

Precedents exist (PADRE, DeepDelta, and any kNN-residual stack).  This track
does not claim novelty, chemistry, structure explanation, composition
generalisation or a top-tier result.  A pass only licenses *designing* a
structural comparator next round; it is not evidence that the comparator works.

## Non-goals of this round

* No graph alignment, MCS, edit rules, interpretable comparator, message
  passing, Transformer, pretraining or new dataset.
* No graph/query branch continuation from `structural-query-learning`; the SQ
  track is closed and only its data caches and frozen reference numbers are
  reused.
* No second seed, extra K/temperature, new head, feature augmentation or
  architecture search.  A failed threshold is a negative result for *this*
  recipe, not for pairwise learning in general.

## Layout

```
tracks/reference-delta-learning/
  TRACK.md                 this charter
  STATE.yaml               current pointer file
  protocols/               frozen comparability rules (dataset/split/metric/selection)
  studies/                 scientific intent
  configs/                 engineering knobs of a run
  code/                    rd_common, rd_data, rd_refs, rd_pairs, rd_eval,
                           rd_tree, rd_arms, rd_runner, rd_report
  tests/                   small synthetic correctness tests (no real matrices)
  cache/                   git-ignored scratch (smoke, nothing large committed)
  runs/                    git-ignored run store of the control plane
  results/rd-pilot-small-v1/  durable per-dataset artefacts, tables, REPORT, decision
```

Data are **not** copied: the four frozen `.npz` caches of
`tracks/structural-query-learning/data/` are read in place and verified by
SHA-256 on every run.

## Entry points

```bash
# identity + frozen references + frozen pair list (CPU, no training)
bash tracks/reference-delta-learning/run_remote.sh --purpose "rd prepare" \
  --set stage=prepare --set dataset=all

# correctness tests
uv run pytest -q tracks/reference-delta-learning/tests

# two GPU workers (one per GPU, one per task), via rr
rr run res rd-esol --pool res-gpu0 --gpus 1 --cpus 4 \
  --result tracks/reference-delta-learning/results -- \
  env RESEARCH_TRACK_ROOT=tracks/reference-delta-learning \
      RESEARCH_EXTRA_RUNNER_PATHS=tracks/reference-delta-learning/code \
      RESEARCH_EXTRA_RUNNERS=rd_runner \
      uv run research run rd-pilot-small-v1 --study rd-pilot-small-v1 \
      --mode screen --set stage=dataset_worker --set dataset=ogbg-molesol

# aggregation (CPU, local, after both workers finish)
uv run research run rd-pilot-small-v1 --study rd-pilot-small-v1 --mode screen \
  --set stage=aggregate --set dataset=all
```
