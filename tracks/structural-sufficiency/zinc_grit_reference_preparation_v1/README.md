# ZINC GRIT reference preparation v1

This directory prepares a reproducible reference environment for GRIT on the
official PyG ZINC subset, and measures the real A100 cost of the literature
recipe.  It is **not** a new model and it does **not** claim to reproduce the
paper's test MAE.  The 2000-epoch run is deliberately not started here.

## Pinned reference

| item | value |
|---|---|
| upstream | `https://github.com/LiamMa/GRIT` |
| commit | `6c988ea600a606fbb49a2246c64a2d37396b3ab5` |
| config | `configs/GRIT/zinc-GRIT-RRWP.yaml` |
| config SHA-256 | `e60b425badb9ff29de9bb8a973bfa73b9d9d4d97cf6d2ae6cc1a8be77edc00bf` |
| compat patch SHA-256 | `dae522601463ae2c56783b3a0174c64f09f9efb0e1f3af388d52150ef9e5a283` |
| source snapshot | `upstream/GRIT/` (pristine, no `.git`) |
| model params | `473,473`, exactly the paper's ZINC value |

The README's old `zinc-GRIT.yaml` does not exist at this commit; the pinned
config is the actual `zinc-GRIT-RRWP.yaml`.  The config is copied byte-for-byte
and is also present in the pristine snapshot.

## What is in the bundle

```
upstream/GRIT/                 pristine upstream source at the pinned commit
patches/                       one audited compatibility patch (4 files)
compat/                        torch_scatter / torch_sparse shims
deps/                          vendored yacs + opt_einsum (pure Python)
configs/                       byte-identical official config
run_reference.py               timing / formal-train entry point
verify_reference.py            CPU functional checks
audit_data.py                  train-only data audit vs the research cache
reference_common.py            runtime assembly, config, loaders, optimizer
reference_checks.py            RRWP / finite / renumbering checks
data_audit.json                audit result (10,000 train graphs)
functional_checks.json         CPU functional result
timing.json                    A100 timing result (see COST_REPORT.md)
COST_REPORT.md                 measured cost and projections
```

`runtime/` is generated on first run from `upstream/GRIT` + the patch; it is
ignored by git.  The patch is applied only to the generated copy, so the
vendored upstream tree remains auditable.

## Data policy: test is disabled at construction

The official loaders normally build train, valid, and test.  This bundle changes
that at the dataset-construction entry point:

- `grit/loader/master_loader.py::_reference_requested_splits()` accepts only
  `train` or `train,val`; `test` raises before any dataset object is created.
- `preformat_ZINC()` passes the explicit split list into `join_dataset_splits`;
  there is no official test loader and no test graph in the joined dataset.
- Timing mode uses `reference.splits=train`: no valid loader is constructed
  either.  The no-grad 1000-row pass is on **train rows** and computes no score.
- `grit/train/custom_train.py` no longer accesses `perf[2]`; its train-only
  branch logs train loss and, when a valid loader exists, valid loss.
- The upstream assertion that test labels are NaN remains in unused code paths;
  it is never reached because no test loader exists.

The official test split was not constructed, read, scored, or used for
checkpoint selection in this preparation.

## Compatibility patch (no model/optimizer math changes)

The patch touches only:

1. `grit/loader/master_loader.py` — explicit split request, hard test block,
   optional train-graph cap for local smoke, `split_names` carried on the
   joined dataset.
2. `grit/loader/split_generator.py` — use `dataset.split_names` instead of
   inferring three splits from GraphGym's val/test keys.
3. `grit/train/custom_train.py` — timing fields, early stop without changing
   `max_epoch`, budget guard, train-only logging, bare `except` cleanup.
4. `grit/logger.py` — make `torchmetrics` optional; no effect on regression
   loss/MAE paths, only classification metric registration.

Model layers, RRWP, loss (`L1`), optimizer (AdamW), and the 2000-epoch
cosine/warmup scheduler are upstream code and unchanged.  The config is
unchanged; the 2000-epoch schedule is kept even in timing mode, which stops
early rather than shrinking the schedule.

Two small compatibility shims replace missing `torch_scatter`/`torch_sparse`
extensions by equivalent PyG 2.6.1 operations.  RRWP is checked against an
independent dense computation (`max_abs = 0.0` on the checked graph).

## Reproduce the functional checks (CPU)

```bash
cd tracks/structural-sufficiency/zinc_grit_reference_preparation_v1
python audit_data.py
python verify_reference.py --device cpu
```

Expected: 10,000 train graphs, atom types `0..20`, bond types `{1,2,3}`,
labels mean `0.0153`, std `2.0108`, and all checks pass.

## Run the A100 timing measurement

One GPU, one process, seed 41, official batch 32, FP32, official 2000-epoch
scheduler, one warmup epoch + three timed epochs, then a 1000-train-row no-grad
forward as an eval-cost proxy.  The budget guard stops before an epoch that
would exceed 420 s.

```bash
cd /home/calendar/code/research
rr deploy res --pool res-gpu0
rr run res zinc-grit-ref-timing \
  --gpus 1 --pool res-gpu0 --time 00:10:00 \
  --result tracks/structural-sufficiency/zinc_grit_reference_preparation_v1/runs/a100-timing \
  -- python tracks/structural-sufficiency/zinc_grit_reference_preparation_v1/run_reference.py \
       --mode timing --device cuda:0 --seed 41 \
       --out-dir tracks/structural-sufficiency/zinc_grit_reference_preparation_v1/runs \
       --run-name a100-timing \
       --timing-out tracks/structural-sufficiency/zinc_grit_reference_preparation_v1/runs/a100-timing/timing_summary.json \
       --overwrite
```

## Start the formal 2000-epoch run later (not started)

```bash
rr run res zinc-grit-ref-formal \
  --gpus 1 --pool res-gpu0 --time 12:00:00 \
  --result tracks/structural-sufficiency/zinc_grit_reference_preparation_v1/runs/formal-seed41 \
  -- python tracks/structural-sufficiency/zinc_grit_reference_preparation_v1/run_reference.py \
       --mode train --confirm-long-run --device cuda:0 --seed 41 \
       --out-dir tracks/structural-sufficiency/zinc_grit_reference_preparation_v1/runs \
       --run-name formal-seed41 \
       --overwrite
```

Formal mode constructs train+valid (never test), evaluates valid every epoch,
and keeps the best-valid checkpoint.  It does not read the official test split.

## Two different evaluations

- **Literature recipe**: 2000 epochs, official train/valid, best-valid
  checkpoint.  A later test read could be compared to the paper's
  `0.059 ± 0.002` test MAE.  This bundle does not read test and makes no such
  claim.
- **Budget-controlled comparison**: P3 uses 360 epochs and a last-5 soup.  A
  2000-epoch + best-valid GRIT number is therefore not a pure architecture
  comparison against P3; training budget and checkpoint selection differ.

See `COST_REPORT.md` for measured epoch time, A100-hours, projections, and the
limitations of the extrapolation.
