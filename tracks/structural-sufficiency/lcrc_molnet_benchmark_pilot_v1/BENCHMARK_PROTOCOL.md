# Benchmark protocol — LCRC-A MoleculeNet pilot v1

Protocol id: `lcrc-molnet-benchmark-pilot-v1`
Author: research agent. Created before any GPU training; frozen by
`analysis/protocol_freeze.json` (written and committed before P1).

---

## 1. Question and scope

Four datasets, one training seed, one formal run each:

| Dataset | Task | Split | Primary metric |
|---|---|---|---|
| ESOL | regression | random | RMSE ↓ |
| Lipophilicity | regression | random | RMSE ↓ |
| FreeSolv | regression | random | RMSE ↓ |
| BACE | binary classification | scaffold | ROC-AUC ↑ |

Three questions only:

1. What does frozen LCRC-A score on traceable published splits?
2. Does it transfer from physicochemical regression to a different
   physicochemical property (FreeSolv) and to bioactivity classification (BACE)?
3. Relative to published GNN numbers, is it competitive, clearly behind, or not
   yet fairly decidable?

Out of scope (explicitly forbidden this round): second seed, hyperparameter
search, S2 ablation, PHYSCHEM6, G2L, pretraining, new architectures, external
GNN training, BBBP/Tox21/HIV.

---

## 2. Data source (fixed, hash-pinned, vendored)

* Repository: <https://github.com/shenwanxiang/ChemBench>
* Commit: `0b10d4c1ac72b6929503694a5f60446824736391` (2022-03-29)
* Reference API: `from chembench import load_data; df, split_repeats = load_data("ESOL")`
* Files: `src/chembench/data_and_index/<task>/<task>.csv.gz` and the three
  published split-index pickles per task.

The bytes are vendored under `analysis/source/chembench/` because upstream has no tagged
release and no checksums. `analysis/source/chembench_source_manifest.json` pins every file
by sha256 and by upstream git blob id; `mb_source.load_source_manifest()`
re-verifies all 16 files on every run.

| file | bytes | sha256 (head) | git blob |
|---|---|---|---|
| `ESOL/ESOL.csv.gz` | 10220 | `3eb977a01fd9cf04…` | `399643dd` |
| `ESOL/ESOL.random122.ind.pkl` | 3696 | `ebac56b86a6cad41…` | `bcf75e7b` |
| `ESOL/ESOL.random123.ind.pkl` | 3696 | `e48b49761171b835…` | `939160ae` |
| `ESOL/ESOL.random124.ind.pkl` | 3696 | `07591375f16b3960…` | `db726576` |
| `Lipop/Lipop.csv.gz` | 56116 | `34280f0adde234bc…` | `e1a548e7` |
| `Lipop/Lipop.random122.ind.pkl` | 12918 | `bf7acb177e080caf…` | `dbc3fc7e` |
| `Lipop/Lipop.random123.ind.pkl` | 12918 | `afaed4c6d95b64ac…` | `6535f8d5` |
| `Lipop/Lipop.random124.ind.pkl` | 12918 | `2b48a41ccc39b622…` | `1e4acb99` |
| `FreeSolv/FreeSolv.csv.gz` | 4413 | `46f3df3dbfc65282…` | `b72c7f96` |
| `FreeSolv/FreeSolv.random122.ind.pkl` | 2238 | `8a93ced2f63e6c79…` | `8029f538` |
| `FreeSolv/FreeSolv.random123.ind.pkl` | 2238 | `307655abc9cf4b39…` | `0376bb56` |
| `FreeSolv/FreeSolv.random124.ind.pkl` | 2238 | `7fa0e2e0e3bec941…` | `d418f6ea` |
| `BACE/BACE.csv.gz` | 12736 | `5f0f81e1069babe3…` | `223d9166` |
| `BACE/BACE.scaffold122.ind.pkl` | 4853 | `fe93d61fcf3a0a26…` | `db67f92c` |
| `BACE/BACE.scaffold123.ind.pkl` | 4853 | `fe93d61fcf3a0a26…` | `db67f92c` |
| `BACE/BACE.scaffold124.ind.pkl` | 4853 | `fe93d61fcf3a0a26…` | `db67f92c` |

Full hashes: `analysis/source/chembench_source_manifest.json` and
`analysis/dataset_manifest.json`.

### 2.1 Which split is "repeat 0"

`chembench.load_data` collects `*.pkl` with `glob` and returns them in
filesystem order, i.e. **`split_repeats[0]` is not reproducible across
machines**. This round therefore freezes repeat 0 as *the lexicographically
first published split file*, whose name suffix is **122**, and records the
sha256 of all three repeats (122/123/124) in
`analysis/split_manifest_<dataset>.json` so the choice is auditable and cannot
have been made after seeing a score.

BACE's three repeat files are byte-identical (the scaffold split is unique);
ESOL/Lipophilicity/FreeSolv's three repeats are genuinely different random
draws and only 122 is used.

### 2.2 Alternative sources reviewed (max two, per the task book)

| source | decision | reason |
|---|---|---|
| DeepChem / MoleculeNet loaders with default splitters | **not used** | the published *indices* are what ChemBench ships; current DeepChem loader defaults are not guaranteed to reproduce the published protocol and the round forbids passing a loader default off as the official split |
| `chemprop/splits.tar.gz` (the ChEMBL/BACE scaffold indices ChemBench itself redistributes) | **not used directly** | ChemBench already redistributes those BACE scaffold indices at a fixed commit; a second source adds no information for the four selected tasks and would break the "one stable source for all four" preference |

### 2.3 Targets (measured values only)

| dataset | ChemBench file | label column | meaning / units |
|---|---|---|---|
| ESOL | `ESOL` | `measured log solubility in mols per litre` | measured logS, log10(mol/L) |
| Lipophilicity | `Lipop` | `exp` | experimental logD (octanol/water) |
| FreeSolv | `FreeSolv` | `expt` | experimental hydration free energy, kcal/mol |
| BACE | `BACE` | `Class` | binary bioactivity class (0/1), **not** pIC50 |

No derived column (predicted solubility, pIC50, descriptors) is read. For ESOL
the CSV has no extra columns in this release; for Lipophilicity/FreeSolv/BACE
only `smiles` + the label column are used.

### 2.4 Sample sizes and known differences from the original papers

| dataset | rows used here | original MoleculeNet figure | note |
|---|---|---|---|
| ESOL | 1128 | 1128 | identical |
| Lipophilicity | 4200 | 4200 | identical |
| FreeSolv | 642 | 643 (FreeSolv database) | cleansed MoleculeNet release has 642; **no molecule added or removed by this round** |
| BACE | 1513 | 1522 | cleansed MoleculeNet release has 1513; **no molecule added or removed by this round** |

Splits: ESOL/Lipophilicity/FreeSolv = random 80/10/10 (902/113/113,
3360/420/420, 513/64/65). BACE = scaffold 1210/151/152.

### 2.5 Row-level difference from the old project splits

| dataset | old protocol | new protocol | row overlap (new train ∩ old train) |
|---|---|---|---|
| ESOL | custom Bemis–Murcko scaffold split, seed 20261103 (`lcrc_esol_quickstart_v1`) | ChemBench published random split, repeat 0 | 723 / 902 = 80.2 % |
| Lipophilicity | `deepchem==2.8.0` ScaffoldSplitter (`bjoc_cross_task_lipophilicity_v1`) | ChemBench published random split, repeat 0 | 2681 / 3360 = 79.8 % |
| FreeSolv | no old project equivalent | random repeat 0 | — |
| BACE | no old project equivalent | scaffold (122=123=124) | — |

The SMILES row order of the new ESOL/Lipophilicity CSVs is byte-identical to
the old project's CSVs, so these overlap numbers are exact row matches, not
fuzzy matches. **Old ESOL/Lipophilicity numbers are therefore not comparable to
this round** (different split protocol); they are kept only as historical
context and graded `C — NOT_COMPARABLE` in the literature matrix.

---

## 3. P0 gates (CPU; must all pass before GPU)

| gate | content |
|---|---|
| G0 | every vendored file sha256-verified against the manifest |
| G1 | Train/Valid/Test row ids disjoint, complete, fixed sizes, repeat 0 = file 122, all repeat hashes recorded |
| G2 | zero SMILES parse failures; no molecule silently dropped |
| G3 | `row id → SMILES → graph → cache row → target` 1:1, for Train, Valid *and* Test structures |
| G4 | structure caches contain no labels (targets live in a separate file) |
| G5 | vocabulary coverage recorded (UNK atom/bond counts and fractions) under the frozen UNK policy |
| G6 | frozen LCRC-A identity: 247,106 parameters, no message-passing/attention modules, empirical radius-2 locality |
| G7 | numerics: single-vs-batch consistency, finite forward/backward, every parameter receives a gradient, exactly one local encode per forward |
| G8 | normalisation is Train-only (recomputed from the Train cache; differs from a Valid-only normalisation) |
| G9 | targets: Train-only mean/std for regression; both classes present in Train **and** Valid for BACE |

Test labels are not decoded during P0 (`load_dataset(..., labels="none")` reads
the CSV with `usecols=["smiles"]`). Test *structures* are built and audited so
that a broken Test split would surface before spending GPU time.

---

## 4. Frozen model

`lcrc_cross_task_preparation_v1/code/lcrc_model.py::LCRC` with the `lipo_v1`
FeatureSpec (`7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee`):

```
a_v   = AtomEncoder(atom_feat_v)                       51 -> 128 -> 96
b_e   = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bond_feat_e])   13+288 -> 160 -> 96
S1_v  = sum_{e incident v} b_e ; S1mean_v ; S1joint_v = sum_nbr Emb_nbr[(type, bond)]
S2_v  = sum_{v->w->u} [Emb_path + Emb_end + W_j(U_path*V_end) + Emb_close + gate*a_u]
h_v   = LayerNorm(LocalMLP([a_v, S1_v, S1mean_v, S1joint_v, S2_v, scalars_v]))   529 -> 192 -> 64
y     = sum_v f(h_v) + rho(global, mean_v h_v, mean_v h_v^2, log1p(n_atoms))
```

Frozen: widths, S1/S2 modules, `rho`, one local encode, no iterative hidden-state
message passing, no attention, no junction module, no extra descriptors, no
pretraining. Parameters: **247,106** (identical for all four tasks).

Only allowed task adaptation:

| | regression | classification (BACE) |
|---|---|---|
| loss | MSE on Train-standardised target | `BCEWithLogitsLoss` on the single raw scalar |
| output | denormalised value | `sigmoid(scalar)` = probability |
| metrics | RMSE, MAE, R² | ROC-AUC (primary), PR-AUC (secondary), BCE |
| checkpoint | lowest Validation RMSE | highest Validation ROC-AUC (tie → lower Validation BCE → earlier epoch) |

No class weights, no threshold tuning, no Test-derived anything.

---

## 5. Training recipe (frozen R3 / ESOL-BASE)

| item | value |
|---|---|
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8 |
| peak LR / min LR | 3e-4 / 1e-5 |
| weight decay | 1e-3 |
| schedule | per-step cosine over `steps_per_epoch × 360` |
| batch size | 128 |
| dropout | 0.10 on the final atom-level local representation `h` |
| gradient clip | 5.0 |
| epochs | 360, no early stopping |
| EMA / SWA / soup / SSL / search | none |
| training seed | **0** (initialisation + shuffle generator seed `0 + 917000`) |
| normalisation | Train-only structure standardisation, Train-only target mean/std (regression) |
| formal runs per dataset | **1** |

Split repeat 0 (a property of the public benchmark) and training seed 0 (a
property of this training run) are separate frozen quantities.

---

## 6. Evaluation discipline

1. P1 trains, keeps the Validation-best checkpoint, reloads it, recomputes
   Validation and checks that the recorded best epoch equals the Validation
   curve argmin/argmax.
2. `mb_freeze.py` writes `analysis/checkpoint_manifest.json` (checkpoint
   sha256, curve sha256, code hashes, data identities) and it is committed.
3. Only then `mb_eval.py` scores the Test split **once** per dataset. It
   refuses a dataset that is not in the manifest, a checkpoint whose sha256
   changed, a rebuilt cache whose fingerprint changed, a Validation metric that
   does not reproduce, or an existing `test_eval.json`.
4. Test results never change an epoch, model, split, learning rate, loss,
   threshold or metric definition.

Re-verification tolerance: the reload/reproduction checks compare metric values
with an absolute tolerance of `1e-5` (`mb_common.REPRO_TOL`), because a CUDA
forward pass is not bit-reproducible across kernel-launch orders (a 2.1e-8
Validation-RMSE difference was observed between the training-time and the
reloaded-checkpoint forward pass on the ESOL GPU smoke run, 2026-11-01). The
observed difference is recorded in every `train.json` / `test_eval.json`; the
tolerance was fixed before any formal run and is five orders of magnitude
tighter than the effects these checks guard against.

Reported: Train/Valid/Test primary metric, Test MAE/R² (regression) or Test
PR-AUC (BACE), best epoch, parameters, GPU time, peak memory, and — separately
— data-preprocessing time and inference throughput. CPU and GPU time are never
mixed in one number.

Single seed: no seed variance, no significance test, no "stable improvement"
claim.

---

## 7. Literature comparison grading

| grade | meaning |
|---|---|
| **A — EXACT_SPLIT_MATCH** | same sample set, same label, same Train/Valid/Test row set, same metric. Only then may a single result be compared directly (still noting seed count and extra features). |
| **B — PROTOCOL_MATCH_ONLY** | same dataset, same split method and metric, but the exact split indices are not verified identical. Reference range only; no "beats model X" claim. |
| **C — NOT_COMPARABLE** | different data version, task, metric, split or supervision. Not ranked. |

If no exact match exists for a task, the report says `NO_EXACT_MATCH_FOUND`
instead of filling the table. A seed-0 result is never compared as
"significantly better" than a published multi-seed mean.

---

## 8. Budget and stopping

* Hard cap **1.5 A100 GPU-hours**, charged from measured training-process wall
  clock in `train.json` (plus inference seconds in `test_eval.json`), recorded
  in `analysis/gpu_ledger.json`.
* 2-epoch engineering smoke runs are not results and are excluded from the
  benchmark table.
* Stop after this four-dataset batch. BBBP/Tox21/HIV require a new decision.

---

## 9. Reproduction

```bash
# P0 (CPU, ~20 s)
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_p0.py
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_protocol_freeze.py

# tests (CPU, ~2 s)
uv run python -m pytest tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/tests -q

# P1 (GPU; one command per dataset)
uv run python .../code/mb_train.py --dataset ESOL --device cuda

# freeze, then one-shot P2
uv run python .../code/mb_freeze.py
uv run python .../code/mb_eval.py --dataset ESOL --device cuda
uv run python .../code/mb_tables.py
```
