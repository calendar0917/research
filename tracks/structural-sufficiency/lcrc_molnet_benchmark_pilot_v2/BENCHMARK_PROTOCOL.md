# Benchmark protocol — LCRC-A seven-task MoleculeNet pilot (Sprint 02)

Protocol id: `lcrc-molnet-benchmark-pilot-v2`
Machine-readable freeze: `analysis/protocol_freeze.json`
(sha256 `b50137cdd81c5144ab3bc0484b4da56a4db95b93c5894b6f3348bd4b0dea70f5`)
This document is the human-readable statement of that same frozen object; it was
written alongside it, before any formal training run, and no number or rule in it
was changed after a result was seen.

---

## 1. Scope

Seven datasets, **one** training seed (0), **one** formal run each. Four of them
(ESOL, Lipophilicity, FreeSolv, BACE) are carried over from Sprint 01 and are
**not** re-trained and **not** re-scored on Test: their artifacts are read
read-only and their Test comparisons are labelled `EXPOSED_TEST_EXPLORATORY`.
Three are new and their Test labels are read exactly once:
BBBP, Tox21, `ogbg-molhiv`.

| Dataset | Task | Split | Primary metric | Test exposure |
|---|---|---|---|---|
| ESOL | regression | random | RMSE ↓ | expired (Sprint 01) |
| Lipophilicity | regression | random | RMSE ↓ | expired (Sprint 01) |
| FreeSolv | regression | random | RMSE ↓ | expired (Sprint 01) |
| BACE | binary | scaffold | ROC-AUC ↑ | expired (Sprint 01) |
| BBBP | binary | scaffold | ROC-AUC ↑ | first read this round |
| Tox21 | 12-endpoint multitask | random | macro ROC-AUC ↑ (observed cells only) | first read this round |
| ogbg-molhiv | binary | official OGB scaffold | ROC-AUC ↑ (official OGB Evaluator) | first read this round |

Not in scope: a second seed, any hyperparameter search, an S2 ablation, new
architectures, pretraining, class weighting or threshold tuning, an external GNN
training run, a new dataset beyond these seven, or a Test read of any new dataset
more than once.

## 2. Data (fixed, hash-pinned, vendored)

### 2.1 ChemBench (six datasets)

* <https://github.com/shenwanxiang/ChemBench>, commit
  `0b10d4c1ac72b6929503694a5f60446824736391` (2022-03-29).
* Reference API: `load_data(<task>)` → `split_repeats[0]` is the split used.
  The three repeat files (suffixes 122/123/124) are read in lexicographic order
  and repeat 0 (`…122`) is selected; `mb2_source` asserts that the
  lexicographically first file really carries suffix 122.
* BBBP and Tox21 bytes are vendored in this track. The four Sprint-01 datasets
  stay in the Sprint-01 vendor directory (same commit, same bytes, verified by
  both manifests).

| file | bytes | sha256 (head) | upstream git blob |
|---|---|---|---|
| `BBBP/BBBP.csv.gz` | 31355 | `698918057e097537…` | `7348cd24` |
| `BBBP/BBBP.scaffold122.ind.pkl` | 6431 | `b132bccdc116cfea…` | `14f2f6e5` |
| `BBBP/BBBP.scaffold123.ind.pkl` | 6431 | `b132bccdc116cfea…` | `14f2f6e5` |
| `BBBP/BBBP.scaffold124.ind.pkl` | 6431 | `b132bccdc116cfea…` | `14f2f6e5` |
| `Tox21/Tox21.csv.gz` | 99229 | `827b799dc6a18ddc…` | `1ac7ca79` |
| `Tox21/Tox21.random122.ind.pkl` | 23817 | `d6694f6661d0ef2f…` | `52186544` |
| `Tox21/Tox21.random123.ind.pkl` | 23817 | `c86f39bd428f650b…` | `e16c6273` |
| `Tox21/Tox21.random124.ind.pkl` | 23817 | `888bb99bf3433f2f…` | `8e192602` |

For BBBP the three published scaffold repeats are **byte-identical**, i.e.
ChemBench ships one BBBP split three times. That is recorded, not corrected: the
round uses the frozen repeat-0 rule that Sprint 01 used for every dataset.

### 2.2 ogbg-molhiv (OGB release v1)

The official release is used as shipped, including its official split files —
this track does not rebuild the scaffold split:

| file | bytes | sha256 (head) |
|---|---|---|
| `RELEASE_v1.txt` | 71 | `e6f109fcccdf5720…` |
| `mapping/mol.csv.gz` | 404655 | `306c4ad8d54fc6cd…` |
| `mapping/README.md` | 322 | `e6b64bdc76f68f18…` |
| `raw/graph-label.csv.gz` | 1727 | `88f950d08c606556…` |
| `split/scaffold/train.csv.gz` | 73605 | `9f1e46e77897b3de…` |
| `split/scaffold/valid.csv.gz` | 9254 | `eaf4d6b4630f9e68…` |
| `split/scaffold/test.csv.gz` | 9503 | `27485729d83edd6f…` |

41,127 molecules, split 32,901 / 4,113 / 4,113, labels {0: 39,684, 1: 1,443}
(3.51 % positive). The `mol_id` column is entirely empty in the published mapping,
so row identity is the OGB graph index, and `mapping/mol.csv.gz` is the only
source of SMILES. **Only the molecular graph is used**: OGB's 9-dimensional node
features and 3-dimensional edge features are *not* inputs to LCRC (the model
consumes the same `lipo_v1` raw-attribute vocabulary as every other dataset).
`mb2_source.ogb_input_semantics()` records this.

## 3. Pre-work audit (P0). Gate list, all must pass per dataset

`mb2_p0.py` → `analysis/p0_audit.json` (status **PASS** on all seven), one
`analysis/split_manifest_<dataset>.json` per dataset plus
`analysis/dataset_manifest.json`:

| gate | check |
|---|---|
| G0 | source bytes match the pinned manifest (sha256 + upstream blob id) |
| G1 | split integrity: partition is a permutation, no id repeats, frozen sizes |
| G2 | no silent drops: every SMILES parses (strict, or the documented relaxed path) |
| G3 | row alignment: SMILES ↔ label ↔ split id agree |
| G4 | structure cache is label-free (no target array inside it) |
| G5 | vocabulary coverage: atom/bond UNK fractions recorded per dataset |
| G6 | frozen model: 247,106 single-task parameters, `lipo_v1` spec hash, no message passing, radius-2 locality |
| G7 | numerics: finite forward/backward, gradient norms |
| G8 | Train-only normalisation: target and descriptor statistics come from Train |
| G9 | targets: dtype, masks, ranges |
| G10 | Tox21 missing labels: NaN preserved (never coerced to 0), masked BCE |
| G11–G13 | Toy/determinism, duplicate and cross-split overlap report, parameter identity |

Recorded per-dataset facts that matter for interpretation:

* Tox21 has 17.05 % missing label cells (Train 62,358 observed / 12,810 missing;
  Validation 7,749 / 1,647). Missingness is **never** treated as a negative.
* Cross-split duplicate SMILES: only ESOL, 4 groups, all **Train/Validation**
  (this round also carries the Sprint-01 report erratum that said Train/Test).
  Test is duplicate-free in all seven datasets.
* Scaffold overlap (post-freeze diagnostic, `analysis/split_scaffold_overlap.json`):
  BACE/BBBP/HIV have **0.000** Validation and Test scaffolds seen in Train, while
  the random-split sets share scaffolds at 0.52 (Lipophilicity), 0.76 (Tox21),
  0.78 (ESOL) and 0.95 (FreeSolv) of Validation rows.
* Multi-component SMILES fractions: BBBP 4.7 %, Tox21 3.1 %, HIV 7.8 %;
  maximum atom count 222 (HIV).
* Eight Tox21 aluminium-adjuvant salts fail strict RDKit sanitisation; they are
  parsed through one documented fallback (`sanitize=False`,
  `UpdatePropertyCache(strict=False)`) and listed by row id. Nothing is dropped:
  Tox21 keeps 7,831 molecules.

## 4. Frozen model and recipe

Model: `LCRCMulti` — the Sprint-01 `lipo_v1` graph (`mb2_rdkit`, a verbatim port of
the v1 adapter, re-verified byte-equal against v1's raw-attribute artifact on all
four Sprint-01 datasets), a shared encoder, and the unchanged LCRC-A heads **for
every single-output dataset**. The only structural deviation authorised this round
is Tox21, where the two readout heads (`f`, `rho`) widen from 1 to 12 outputs:
247,106 parameters per single-output dataset, **248,536** for Tox21 (+1,430). The
audit asserts that the shared-encoder state hash is identical between the
single-output and the 12-output model at the same seed.

Recipe R3 (unchanged from Sprint 01, per-step cosine):

| item | value |
|---|---|
| optimiser | AdamW |
| peak LR → min LR | 3e-4 → 1e-5, cosine per optimisation step |
| weight decay | 1e-3 |
| batch size | 128 |
| dropout | 0.10, on `h` only |
| gradient clip | 5.0 (global norm) |
| epochs | 360 |
| loss | MSE on the Train-normalised target (regression) / BCE with logits (binary) / masked BCE with logits, denominator = observed cells (Tox21) |
| checkpoint | single best-Validation epoch, strict improvement; multitask tie → lower masked BCE → earlier epoch |
| selection metric | Validation RMSE (regression) / Validation ROC-AUC (binary) / Validation macro ROC-AUC over definable endpoints (Tox21) |
| early stopping, EMA, SWA, SSL, LR search | none |
| inference | argmax-free probability readout; no threshold tuning (ACC@0.5 reported for the record only) |

Data loader shuffle uses seed 0 with a fixed offset (917,000); `torch` deterministic
algorithms where available.

## 5. Two-phase access: freeze, then one Test read

1. `mb2_freeze.py` writes `analysis/checkpoint_manifest.json` **before** any Test
   access, binding for each dataset: checkpoint sha256, best epoch, frozen
   Validation metric, split-file hash, split-row hashes, cache fingerprints,
   Train-only normalisers, code-module hashes, git commit.
2. `mb2_eval.py` refuses to run unless: the dataset is in that manifest, the
   on-disk checkpoint sha256 equals the frozen value, no recorded code module
   changed, rebuilding Train/Validation caches reproduces the frozen
   fingerprints, the reloaded checkpoint reproduces the frozen Validation metric
   within 1e-5, the best epoch equals the Validation-curve argbest, the Test row
   ids equal the P0 split manifest, and no `test_eval.json` already exists.
3. Every actual Test read is appended to `analysis/test_access_ledger.json`
   with timestamp, manifest hash and prediction hash. The ledger is append-only.

The sequence is therefore: P1 train → freeze → (P2 baseline freeze) → one Test
read per new dataset → P3 Validation-only probes → tables. A Test number can
never influence a configuration: the configuration is already hashed.

## 6. P2 — matched ECFP4 + RDKit-descriptor + XGBoost baseline

One frozen configuration, CPU only, no tuning, no Validation-driven choice:
ECFP4 (radius 2, 2048 bits, count-folded) + the 217-entry RDKit descriptor list,
per-descriptor **Train-only** median imputation, Train-only constant-column drop,
`hist` tree method, 500 trees, depth 5, LR 0.03, subsample/colsample 0.8,
`reg_lambda` 5, `random_state` 0. Tox21 gets one model per endpoint, trained only
on that endpoint's labelled Train rows, scored on observed Test cells and
macro-averaged over definable endpoints.

The baseline consumes **the same Train/Validation/Test row ids** as LCRC — the
split manifests are the shared authority — so the comparison is a same-row
comparison, not a same-protocol guess.

## 7. P3 — four Validation-only recipe probes

Fixed before running, never expanded, never evaluated on Test:

| name | dataset | single change vs R3 |
|---|---|---|
| `fs_dropout00` | FreeSolv | dropout 0.10 → 0.00 |
| `fs_dropout20` | FreeSolv | dropout 0.10 → 0.20 |
| `bace_dropout20` | BACE | dropout 0.10 → 0.20 |
| `bace_wd1e2` | BACE | weight decay 1e-3 → 1e-2 |

`mb2_train.resolve_recipe` refuses any other difference, and refuses a probe
registered for a different dataset. Probes never build a Test structure cache and
never write a Test metric; they are excluded from the seven-dataset table. Their
purpose is a regularisation sanity check on the two datasets where Sprint 01
looked weak, not a search.

## 8. Budget

Hard cap **2.0 A100 GPU-hours**, at most **7 new GPU runs** (3 P1 + 4 P3).
Charging rule: one A100 per run, training-process wall clock from `train.json`;
CPU baseline training and CPU Test evaluation are charged 0 GPU-hours and reported
separately. Engineering smokes are reported separately and are not results.

Stop conditions, frozen before the round: if the projected `ogbg-molhiv` run would
exceed the remaining cap, report `BUDGET_BLOCKED` and skip HIV; P3 is dropped
first if the budget gets tight; **no epoch reduction, batch-size inflation or
recipe change may be used to fit the budget**.

HIV projection, recorded before launch: a 2-epoch GPU smoke measured 8.6 s/epoch,
so 360 epochs ≈ 3,100 s ≈ 0.86 h; with BBBP 0.038 h + Tox21 0.179 h + P3 ≈ 0.12 h
the round projects ≈ 1.20 h, i.e. inside the cap. The projection is in the ledger.

## 9. Statistics and what the numbers can support

One seed per dataset. No confidence interval, no significance test, no ranking.
A single-seed difference of one or two points is reported as an observation, not
as evidence of superiority. Literature rows are graded before comparison:

* **A** exact split match (same row ids and metric): only `ogbg-molhiv`, whose
  official split is the one this round consumes;
* **B** protocol match only (same dataset and split method, unpublished indices);
* **C** not comparable (different data version, molecule count, split method or
  metric) — never ranked against this round.

A seed-0 point estimate is never described as beating a published multi-seed mean.

## 10. Deliverables

`analysis/`: `p0_audit.json`, `protocol_freeze.json`, `source_manifest.json`,
`raw_artifact_manifest.json`, `split_manifest_*.json`, `dataset_manifest.json`,
`runs/<dataset>_s0/{train.json,curve.csv,checkpoint.pt,test_eval.json}`,
`baseline/`, `checkpoint_manifest.json`, `test_access_ledger.json`,
`validation_tuning_probe.csv`, `probe_manifest.json`, `seven_dataset_table.csv`,
`matched_baseline.csv`, `benchmark_table.csv`, `literature_matrix.csv`,
`literature_sources.json`, `dump`, `split_scaffold_overlap.json`,
`gpu_ledger.json`, `summary.json`; plus `REPORT.md` and this protocol.

## 11. Reproduction

```bash
cd tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2

# CPU: source integrity, label-free raw artifact, P0 audit, protocol freeze
uv run python code/mb2_source.py --write
uv run python code/mb2_raw_artifact.py --write
uv run python code/mb2_p0.py
uv run python code/mb2_protocol_freeze.py

# CPU: the matched baseline (same row ids, frozen config, no tuning)
uv run python code/mb2_baseline.py --stage all
uv run python code/mb2_baseline.py --stage freeze

# GPU: three P1 runs + four Validation-only probes
rr run res molnet-v2-train-BBBP     --pool res-gpu0 --gpus 1 ... -- python code/mb2_train.py --dataset BBBP --device cuda
rr run res molnet-v2-train-Tox21    --pool res-gpu1 --gpus 1 ... -- python code/mb2_train.py --dataset Tox21 --device cuda
rr run res molnet-v2-train-ogbg-molhiv --pool res-gpu0 --gpus 1 ... -- python code/mb2_train.py --dataset ogbg-molhiv --device cuda
rr run res molnet-v2-probes-validation-only --pool res-gpu1 ... -- python code/mb2_train.py --dataset FreeSolv --device cuda --probe fs_dropout00  # and the other three

# freeze, then exactly one Test read per new dataset, then the tables
uv run python code/mb2_freeze.py
uv run python code/mb2_eval.py --dataset BBBP --device cpu   # Tox21, ogbg-molhiv likewise
uv run python code/mb2_baseline.py --stage test
uv run python code/mb2_probe.py
uv run python code/mb2_literature.py
uv run python code/mb2_tables.py
```
