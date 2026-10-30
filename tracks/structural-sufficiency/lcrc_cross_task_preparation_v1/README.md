# LCRC cross-task preparation v1

**Status: `READY` for the interface/data/model-construction audit only.**  
This round does **not** train, fine-tune, tune, or evaluate a Lipophilicity model. It does not use the frozen ZINC SSL checkpoint on Lipophilicity, does not read labels, and makes no claim that LCRC-A predicts lipophilicity.

## Purpose

Extend the ZINC-specific LCRC-A implementation into an explicit, schema-driven,
cross-dataset interface and audit it on the frozen Lipophilicity Phase 1a
structure split:

* `Fit` 2352 / `Monitor` 504 / `Reserved` 504 = official train 3360 rows;
* label-free raw graphs from `lipo_raw_v1.json.gz`;
* CPU only; no optimizer step, no backward pass, no GPU.

The round separately replays the frozen ZINC P3 inputs, collate path, and
`runs/lcrcA_s0/best.pt` checkpoint through a `legacy_zinc` adapter. That replay
is a compatibility check of the *interface*, not a Lipophilicity transfer
experiment.

## Layout

```
lcrc_cross_task_preparation_v1/
├── README.md
├── REPORT.md
├── code/
│   ├── feature_spec.py       # legacy_zinc + lipo_v1 specs, serialisation/hash
│   ├── molecular_graph.py    # canonical physical bonds, BFS/ring/path analysis
│   ├── adapters.py           # label-free source loaders + ZINC/Lipophilicity adapters
│   ├── lcrc_data.py          # molecule arrays, cache, Fit-only norm, collate
│   ├── lcrc_model.py         # schema-driven LCRC-A, unchanged mechanism
│   ├── zinc_legacy.py        # read-only ZINC input/collate/checkpoint replay
│   ├── cpu_gates.py          # reusable CPU correctness gates (tests + audit)
│   ├── audit_cpu.py          # main CLI
│   └── lcrc_common.py        # paths, hashing, safe row-index reader
├── tests/                    # pytest gates and compatibility samples
├── analysis/                 # committed evidence JSON
└── cache/                    # local, git-ignored structural caches
```

## Commands

Run from the repository root inside the `uv` environment:

```bash
# Full Lipophilicity audit + ZINC replay (default)
uv run python tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/audit_cpu.py \
  --dataset lipo --scope official-train-only --device cpu

# Lipophilicity only
uv run python tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/audit_cpu.py \
  --dataset lipo --scope official-train-only --device cpu --skip-zinc

# Full ZINC legacy replay only
uv run python tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/audit_cpu.py \
  --dataset zinc-legacy --scope official-train-only --device cpu --zinc-max 10000

# Tests
uv run python -m pytest tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/tests -q
```

`--quick` only shrinks the ZINC replay sample; the Lipophilicity audit always
processes all 3360 in-scope rows. `--device` must be `cpu`. `--scope` must be
`official-train-only`.

## Schemas

| | `legacy_zinc` | `lipo_v1` |
|---|---|---|
| atom feature width | 108 | 51 |
| bond feature width | 7 | 13 |
| atom scalars | 9 | 9 |
| global features | 33 | 42 |
| atom types | 21 | 16 (15 named + UNK) |
| bond types | 3 (one-based) | 5 (SINGLE, DOUBLE, TRIPLE, AROMATIC, UNK) |
| S1 vocabulary | 63 | 80 |
| S2 vocabulary | 189 | 400 |
| structural code dtype | `int32` | `int32` |
| disconnected molecules | reject | supported with finite statistics |

`lipo_v1` is **not** zero-padded or rescaled from `legacy_zinc`. Dimensions,
vocabularies, global statistics and scalar definitions are independent and are
recorded in `analysis/feature_spec_lipo.json`.

Key `lipo_v1` formulas:

* S1: `atom_type[neighbour] * 5 + (bond_type - 1)`.
* S2 category: `atom_type[middle] * 25 + (bond_type_1 - 1) * 5 + (bond_type_2 - 1)`.
* S2 end code: `atom_type[end] * 5 + (bond_type_2 - 1)`.
* S2 closure: `1` iff the end atom is adjacent to the root.
* `N_PATH_CODES = 16 * 5 * 5 = 400 > 255`; therefore path and S1 codes use
  at least `int32`. The audit demonstrates that a synthetic max code `399`
  would become `143` under `uint8`.

The Lipophilicity vocabulary is the fixed label-free source vocabulary
(`atom_class` 0..15, `edge_class` 0..4). The Fit scan observes 0..14 and 0..3;
UNK categories are explicit and are exercised by synthetic edge cases. No
vocabulary is fitted from Monitor/Reserved labels or structures.

## Evidence artifacts

| file | meaning |
|---|---|
| `analysis/feature_spec_lipo.json` | serialisable + hashed `lipo_v1` spec, including RDKit/Python/NumPy/Torch provenance |
| `analysis/feature_spec_legacy_zinc.json` | reconstructed frozen ZINC schema |
| `analysis/source_manifest.json` | source hashes, split/scaffold checks, row-manifest alignment, reader policy |
| `analysis/norm_lipo.json` | Fit-only normalisation, including constant-feature policy |
| `analysis/cpu_audit.json` | full 3360-row CPU audit, coverage, Gates G0–G7, final status |
| `analysis/zinc_compatibility.json` | 10k P3 input replay, collate replay, frozen-checkpoint CPU replay |

The structural caches are local reproducibility artifacts (`cache/*.pt`) and are
ignored by git. They contain no labels.

## Non-goals / prohibitions enforced by the code

* No GPU, training, fine-tuning, hyper-parameter search or optimizer step.
* No reading of `labels_fit_monitor.json`, `labels.npy`, official valid/test
  labels, or the CSV `exp` column.
* No modification of frozen ZINC artifacts, split files, caches or checkpoints.
* No zero-padding/truncation/bridging between the two schemas.
* No extra attention, recursive hidden-state message passing, junction module
  or auxiliary loss in LCRC-A.
* No silent dropping of atoms, molecules, bonds, paths or unknown categories.
