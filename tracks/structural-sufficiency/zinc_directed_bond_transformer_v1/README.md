# zinc-directed-bond-transformer-v1

One single-model ZINC candidate: **DBT** — shared local P3 encoders, a
directed-bond token that keeps source/target roles, four layers of full-graph
relation-conditioned attention (exact simple-path counts + distances of the four
endpoint blocks), one amplitude-preserving reader. 452,523 parameters.
Seed 0, 1,500 epochs / 235,500 updates, best-valid checkpoint, one frozen test
read.

* Design + declared recipe: [`DESIGN.md`](DESIGN.md), [`configs/dbt_v1.json`](configs/dbt_v1.json)
* Results and conclusion: [`REPORT.md`](REPORT.md)
* Pre-training checks: [`analysis/cpu_checks.json`](analysis/cpu_checks.json) (8/8 pass)
* Curves: [`analysis/curves.png`](analysis/curves.png), [`analysis/curve.csv`](analysis/curve.csv)
* Main table: [`analysis/main_table.md`](analysis/main_table.md)
* Frozen test read: [`analysis/test_freeze.json`](analysis/test_freeze.json),
  [`analysis/test_report.json`](analysis/test_report.json),
  [`analysis/test_contrasts.json`](analysis/test_contrasts.json),
  [`analysis/valid_contrasts.json`](analysis/valid_contrasts.json)

## Reproduce

All commands are run from the repository root. `python` below means
`uv run python`; on the A100 host the same commands were launched through
`rr run res <exp> --pool res-gpu0 -- python ...`.

```bash
# 1. label-free path-count / directed-object caches (train, valid; ~20 s CPU)
uv run python tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/dbt_runner.py prep

# 2. pre-training checks: path counts, invariance, junction roles, padding,
#    relation->value pathway, SSL load + optimizer coverage, readout amplitude,
#    data/schedule alignment (CPU, ~5 s)
uv run python tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/dbt_runner.py checks

# 3. timing (2 epochs; the declared projection is recorded, not a stop rule)
uv run python tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/dbt_runner.py timing --device cuda:0 --epochs 2

# 4. the one formal run (resume continues the same cosine; refuses a horizon change)
uv run python tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/dbt_runner.py train --device cuda:0
#    ... --resume   after an interruption

# 5. curves + main table (valid only; reads the historical machine results)
uv run python tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/dbt_runner.py analysis

# 6. terminal scoring of the frozen best-valid checkpoint (writes the freeze
#    first, then reads the official test labels exactly once)
uv run python tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/dbt_runner.py test --device cuda:0
#    --no-test stops right after the valid reproduction + freeze,
#    --freeze-out PATH writes the freeze elsewhere
```

## Files

| path | content |
|---|---|
| `code/dbt_data.py` | exact bounded simple-path counts, distances, directed-bond / junction-role tables, per-split caches, batch assembly |
| `code/dbt_models.py` | relation encoder, relation-conditioned full-graph layer, DBT model, SSL initialisation |
| `code/dbt_train.py` | recipe, trainer, resume state, full-valid evaluation and best selection |
| `code/dbt_checks.py` | the seven pre-training checks |
| `code/dbt_test.py` | freeze + one terminal test read |
| `code/dbt_analysis.py` | curves, main table, descriptive paired bootstraps |
| `code/dbt_runner.py` | CLI for all of the above |
| `analysis/` | committed machine-written results (checks, curves, tables, freeze, test report) |
| `runs/` | git-ignored: `dbt_s0/{best,final,last}.pt`, `train.json`, `curve.csv`, predictions |
| `cache/` | git-ignored, rebuildable: `dbt-path-v1-L12_{train,valid}.pt`, `dbt-objects-v1_{split}.pt` |

## State of the run

| field | value |
|---|---|
| best valid | **0.077768402** @ epoch 949 (LR 9.2e-5) |
| eval-train@best | 0.015964 |
| test MAE (frozen checkpoint) | **0.069401494** |
| final (not selected) | valid 0.081313816, train 0.007638685 |
| parameters | 452,523 (cap 500,000) |
| updates | 235,500 (= 157 × 1,500) |
| GPU time | 3.6918 h training + 37 s timing + 11 s test on one A100-40GB |
| checkpoint sha256 | `5810aaf7b7efa9855e52ccfe920338924d50ffce1d6b9dae8573a09d8933e000` |
| status | complete; no further seeds, grids or follow-ups started |

The official ZINC test split is already spent in this repository; the test row
is a descriptive read of one frozen checkpoint, not a blind evaluation. See
`REPORT.md` §5 for the honest reading of valid vs test.
