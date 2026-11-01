# LCRC-A seven-task MoleculeNet benchmark pilot v2

**Status: see `REPORT.md`.** One frozen LCRC-A per dataset, one training seed (0),
published hash-pinned splits, one one-shot Test read per *new* dataset, and a
frozen ECFP4 + RDKit-descriptor + XGBoost baseline on the identical row ids.

Sprint 01 (`lcrc_molnet_benchmark_pilot_v1`) established the ESOL /
Lipophilicity / FreeSolv / BACE numbers; their Test labels are therefore
**expired** and this round does not re-read them. This round adds three datasets
whose Test labels were never read — BBBP, Tox21 and `ogbg-molhiv` — and adds the
two controls that make the comparison meaningful: a matched classical baseline
and a Validation-only regularisation probe.

## Datasets

| Dataset | Task | Split | Primary metric | Rows (train/valid/test) | Test exposure |
|---|---|---|---|---|---|
| ESOL | regression | random | RMSE ↓ | 1128 (902/113/113) | expired (Sprint 01) |
| Lipophilicity | regression | random | RMSE ↓ | 4200 (3360/420/420) | expired (Sprint 01) |
| FreeSolv | regression | random | RMSE ↓ | 642 (513/64/65) | expired (Sprint 01) |
| BACE | classification | scaffold | ROC-AUC ↑ | 1513 (1210/151/152) | expired (Sprint 01) |
| BBBP | classification | scaffold | ROC-AUC ↑ | 2039 (1631/204/204) | **first read** |
| Tox21 | 12-endpoint classification | random | macro ROC-AUC ↑ (observed cells) | 7831 (6264/783/784) | **first read** |
| ogbg-molhiv | classification | official OGB scaffold | ROC-AUC ↑ (official Evaluator) | 41127 (32901/4113/4113) | **first read** |

ChemBench data and split indices: `shenwanxiang/ChemBench` at commit
`0b10d4c1ac72b6929503694a5f60446824736391`, vendored byte-for-byte.
`ogbg-molhiv`: the official OGB release v1 as shipped, including its split files.

## Layout

```
lcrc_molnet_benchmark_pilot_v2/
├── BENCHMARK_PROTOCOL.md          # pre-registration: data, gates, recipe, policy
├── REPORT.md                      # results, positioning, probe, next steps
├── README.md
├── analysis/
│   ├── source/chembench/{BBBP,Tox21}/     # vendored CSV + three published split files
│   ├── source/ogb/ogbg_molhiv/           # official mapping, labels, scaffold split, release note
│   ├── source/raw/lcrc_raw_v2/*.npz      # committed label-free raw-attribute artifact
│   ├── source_manifest.json / raw_artifact_manifest.json
│   ├── p0_audit.json / protocol_freeze.json / split_manifest_*.json / dataset_manifest.json
│   ├── checkpoint_manifest.json / test_access_ledger.json
│   ├── baseline/                          # frozen XGB baseline reports, predictions, manifest
│   ├── runs/<dataset>_s0/                 # checkpoint, curve, train.json, test_eval.json, predictions
│   ├── runs/probe_*_s0/                   # P3 Validation-only probes
│   ├── seven_dataset_table.csv / matched_baseline.csv / benchmark_table.csv
│   ├── validation_tuning_probe.csv / probe_manifest.json
│   ├── literature_matrix.csv / literature_sources.json
│   ├── split_scaffold_overlap.json / rr_run_registry.json
│   └── gpu_ledger.json / summary.json
├── code/
│   ├── mb2_common.py              # paths, frozen R3 recipe, probe allowlist, dataset registry
│   ├── mb2_source.py              # hash-pinned ChemBench + OGB loaders
│   ├── mb2_rdkit.py               # verbatim v1 adapter (+ one documented relaxed parse)
│   ├── mb2_raw_artifact.py        # builds/verifies the committed label-free artifact
│   ├── mb2_data.py                # structure caches, Train-only normalisers, targets
│   ├── mb2_model.py               # frozen LCRC-A (+12-output Tox21 heads), locality audit
│   ├── mb2_metrics.py             # regression/binary/multitask metrics, official OGB Evaluator
│   ├── mb2_p0.py                  # G0-G13 pre-flight audit
│   ├── mb2_protocol_freeze.py     # pre-registration freeze
│   ├── mb2_train.py               # P1 formal runs + P3 probes (allowlist-guarded)
│   ├── mb2_freeze.py              # checkpoint manifest, before any Test access
│   ├── mb2_eval.py                # one-shot Test evaluation with its own gate
│   ├── mb2_baseline.py            # P2 matched ECFP4+RDKit+XGBoost
│   ├── mb2_probe.py               # P3 Validation-only table
│   ├── mb2_literature.py          # literature matrix + A/B/C grading
│   ├── mb2_split_audit.py         # post-freeze scaffold-overlap diagnostic
│   └── mb2_tables.py              # seven-dataset tables, GPU ledger, summary
├── tests/                         # 49 CPU tests
└── cache/                         # local structure caches (git-ignored, label-free)
```

## Commands

```bash
cd tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2

# CPU: source integrity, raw artifact, P0 audit, protocol freeze
uv run python code/mb2_source.py --write
uv run python code/mb2_raw_artifact.py --write
uv run python code/mb2_p0.py
uv run python code/mb2_protocol_freeze.py

# CPU: matched baseline (same row ids, frozen configuration, no tuning)
uv run python code/mb2_baseline.py --stage all
uv run python code/mb2_baseline.py --stage freeze

# GPU (through rr, committed revision only)
rr run res molnet-v2-train-BBBP --pool res-gpu0 ... -- python code/mb2_train.py --dataset BBBP --device cuda

# freeze, one Test read per new dataset, then the tables
uv run python code/mb2_freeze.py
uv run python code/mb2_eval.py --dataset BBBP --device cpu
uv run python code/mb2_baseline.py --stage test
uv run python code/mb2_probe.py && uv run python code/mb2_literature.py && uv run python code/mb2_tables.py

# tests
uv run python -m pytest code/../tests -q
```

## Non-negotiables enforced in code

* No new split is generated; the published one is used as published, including
  any cross-split duplicate it contains (recorded, never repaired).
* No molecule is added, removed or silently dropped: eight Tox21 aluminium salts
  use one documented relaxed parse, and a genuine parse failure aborts the task.
* The model is the frozen `lipo_v1` LCRC-A: 247,106 parameters per single-output
  dataset; Tox21 is the only authorised change (+1,430 head parameters, 248,536).
* Tox21 missing labels are `NaN`, never 0; the loss is masked with the
  denominator equal to the observed cells.
* No second seed, no hyperparameter search, no early stopping, no EMA/SWA/SSL.
* Test labels are never decoded before `checkpoint_manifest.json` exists, and
  every Test read is appended to `test_access_ledger.json`.
* The four datasets whose Test labels were read in Sprint 01 are not re-scored;
  their comparisons are labelled `EXPOSED_TEST_EXPLORATORY`.
