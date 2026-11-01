# LCRC-A MoleculeNet benchmark pilot v1

**Status: see `REPORT.md`.** One frozen LCRC-A per dataset, one training seed
(0), published hash-pinned splits, one one-shot Test evaluation per dataset.

This round answers a performance-positioning question, not a mechanism
question: *what does frozen LCRC-A actually score on ESOL, Lipophilicity,
FreeSolv and BACE when the splits are the public ones?*

## Datasets

| Dataset | Task | Split | Primary metric | Rows (train/valid/test) |
|---|---|---|---|---|
| ESOL | regression | random | RMSE ↓ | 1128 (902/113/113) |
| Lipophilicity | regression | random | RMSE ↓ | 4200 (3360/420/420) |
| FreeSolv | regression | random | RMSE ↓ | 642 (513/64/65) |
| BACE | classification | scaffold | ROC-AUC ↑ | 1513 (1210/151/152) |

Data and split indices come from `shenwanxiang/ChemBench` at commit
`0b10d4c1ac72b6929503694a5f60446824736391`, vendored byte-for-byte under
`analysis/source/chembench/` with sha256 pins in `analysis/source/chembench_source_manifest.json`.

## Layout

```
lcrc_molnet_benchmark_pilot_v1/
├── BENCHMARK_PROTOCOL.md      # pre-registration: data, splits, gates, recipe, evaluation
├── REPORT.md                  # results + positioning + next step
├── README.md
├── analysis/
│   ├── source/chembench_source_manifest.json
│   ├── source/chembench/<task>/   # vendored CSV + all three published split files
│   ├── source/raw_attributes_v1.json.gz  # committed label-free attributes (RDKit is local-only)
│   ├── p0_audit.json / protocol_freeze.json / checkpoint_manifest.json
│   ├── benchmark_table.csv / literature_matrix.csv / gpu_ledger.json
│   └── runs/<dataset>_s0/     # checkpoint, curve, train/eval JSON, predictions
├── code/
│   ├── mb_common.py           # paths, frozen declarations, hashing
│   ├── mb_raw_artifact.py     # builds/verifies the committed raw-attribute artifact
│   ├── mb_source.py           # ChemBench loader + split-repeat policy
│   ├── mb_rdkit.py            # SMILES -> frozen raw attributes -> lipo_v1 graph
│   ├── mb_data.py             # structure caches, train-only normalisation, batching
│   ├── mb_model.py            # frozen LCRC-A + dropout-at-h + locality audit
│   ├── mb_p0.py               # CPU pre-flight audit
│   ├── mb_protocol_freeze.py  # pre-registration freeze
│   ├── mb_train.py            # P1: one formal run per dataset
│   ├── mb_freeze.py           # checkpoint manifest (before any Test access)
│   ├── mb_eval.py             # P2: one-shot Test evaluation
│   ├── mb_tables.py           # benchmark table, GPU ledger, summary
│   └── mb_literature.py       # literature matrix + A/B/C grading
├── tests/                     # 44 CPU tests (~2 s)
├── cache/                     # local structure caches (git-ignored, label-free)
└── analysis/
    ├── dataset_manifest.json
    ├── split_manifest_<dataset>.json
    ├── p0_audit.json
    ├── protocol_freeze.json
    ├── checkpoint_manifest.json
    ├── benchmark_table.csv
    ├── literature_matrix.csv
    ├── gpu_ledger.json
    └── runs/<dataset>_s0/     # checkpoint, curve, predictions, train.json, test_eval.json
```

## Commands

```bash
# P0 audit + protocol freeze (CPU)
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_p0.py
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_protocol_freeze.py

# tests
uv run python -m pytest tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/tests -q

# P1 / P2
uv run python .../code/mb_train.py --dataset ESOL --device cuda
uv run python .../code/mb_freeze.py
uv run python .../code/mb_eval.py --dataset ESOL --device cpu
uv run python .../code/mb_tables.py
uv run python .../code/mb_literature.py
```

## Non-negotiables enforced in code

* No new split is generated; the published one is used as published, including
  any cross-split duplicate it contains (recorded, never repaired).
* No molecule is added, removed or silently dropped; a SMILES parse failure
  aborts the task instead of shrinking the benchmark.
* No second seed, no hyperparameter search, no ablation, no extra features.
* Test labels are not decoded during P0 or P1; the Test split is scored once,
  after `checkpoint_manifest.json` is frozen and committed.
* The model is the frozen `lipo_v1` LCRC-A (247,106 parameters); only the
  loss/output/metric/checkpoint criterion adapt to the task.
