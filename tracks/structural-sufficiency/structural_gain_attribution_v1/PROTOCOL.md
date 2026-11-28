# PROTOCOL — 显式子图预测增益归因与神经读出迁移研究 v1

Round: `structural-gain-attribution-v1`
Directory: `tracks/structural-sufficiency/structural_gain_attribution_v1/`
Seed: `0` everywhere (single seed by design)
Predecessor: `tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0/` (`MSDS-v0`)

This file is written and committed **before** any model of this round is fitted.
Everything below is pinned: representations, K, filters, arms, recipes, contrasts,
metrics, bootstrap, budget and stop rules. Nothing here may be changed after a
Test number is seen; engineering errors may only be fixed and recorded (§11).

The round is a *progressive* attribution study, not a model search. Explicitly out
of scope: new GNN/architecture, structure-response losses, support-regularisation
search, parent/child hierarchy design, new aggregation modules, KSVD / graph
tokenisation, extra seeds, extra datasets, extra structure scales, Optuna, tree
depth search, vocabulary re-selection, new subgraph sizes.

---

## 1. Questions and what each phase can decide

| Q | question | decided by |
|---|---|---|
| Q1 | what extra value do real subgraphs carry over ordinary fingerprints / local-environment counts? | Phase A contrasts |
| Q2 | same structure information under XGBoost vs MLP readout | Phase B |
| Q3 | does the phenomenon reappear on a ZINC regression task? | Phase C |

Phase 0 repairs the predecessor's *display* layer and splits "out of vocabulary"
into three genuinely different categories; it trains nothing.

Evidence classes (unchanged from MSDS-v0):

| class | meaning |
|---|---|
| `EXACT_MATCH` | same data, features, model, recipe, evaluation (reused verbatim) |
| `SAME_SPLIT_REFERENCE` | same official split and metric, different recipe |
| `HISTORICAL_REFERENCE` | earlier numbers, never a strict ablation |

## 2. Data, splits, index identity

* MolHIV: raw `ogbg-molhiv` (`data/ogb/ogbg_molhiv/raw/*.csv.gz`), official scaffold
  split, graph index = global 0-based row id, sizes **32 901 / 4 113 / 4 113**.
  Split row ids and their sha256 are recorded (`analysis/phase0_input_checks.json`).
* ZINC: PyG `ZINC(subset=True)` official split, **10 000 / 1 000 / 1 000**, raw
  labels as provided by the dataset (`y`), nodes/edges exactly as provided.
* No re-splitting, no Train+Valid refit, no merging of Train and Valid.
* Train-only fitting for every vocabulary, column order, scaling statistic,
  standardisation and class weight.

Test policy: `EXPOSED_DEVELOPMENT_TEST` — Test is evaluated every epoch and
reported, but it never selects a vocabulary, an epoch, a hyper-parameter, a
feature set or a trigger. No design decision in this round is made on Test.

## 3. Reused assets (never retrained, never rewritten)

| asset | source | reuse |
|---|---|---|
| enumeration cache (5 420 485 instances, 71 674 FULL / 13 915 COARSE types) | MSDS-v0 `data/phaseA_instances.npz` | read-only |
| frozen vocabulary (754 COARSE, 3 652 FULL) | MSDS-v0 `data/vocab.npz` | read-only, hashes re-verified |
| design matrices C / F | MSDS-v0 `data/design_CF.npz` | **rebuilt** from the enumeration cache and required to be bit-identical (max abs diff 0) |
| historical `H` (2 258 columns) | `lcrc_molnet_benchmark_pilot_v2` frozen cache + manifest | rebuilt through MSDS-v0 `run_phaseD_xgb.historical_block`; must reproduce `median_sha256` / `keep_sha256` / `n_used_columns` |
| **X0** | historical per-molecule Valid/Test predictions | read directly (evidence class `EXACT_MATCH`, verified in MSDS-v0) |
| **X-CF** | MSDS-v0 Phase-D `XGB_STRUCT` per-molecule predictions | read directly (`SAME_SPLIT_REFERENCE` vs X0) |

X0 and X-CF are **not** refitted. Only six new XGBoost models (MolHIV) + two
(XGB, ZINC) and three new MLPs (MolHIV) + two (MLP, ZINC) are trained.

## 4. Phase 0.1 — chemical attribute decoding (ERRATA, display only)

MSDS-v0's `msds_enum.atom_label` / `bond_label` read raw OGB **category indices**
as physical atomic numbers / charges / bond orders. Decoding must use OGB
`allowable_features`:

| field | correct mapping | predecessor mapping |
|---|---|---|
| `atomic_num` | `possible_atomic_num_list[i]` = atomic number `i+1` (index 5 → C) | `_ELEMENTS[i]` (index 5 → B) |
| `formal_charge` | `[-5..5,'misc'][i]` (index 5 → 0) | used index as the charge |
| `hybridization` | `['SP','SP2','SP3','SP3D','SP3D2','misc'][i]` | ad-hoc shifted table |
| `bond_type` | `['SINGLE','DOUBLE','TRIPLE','AROMATIC','misc'][i]` | ad-hoc shifted table |
| `chirality`, `degree`, `num_h`, `num_radical_e`, `aromatic`, `in_ring`, `stereo`, `conjugated` | `allowable_features` table | partly ignored / mislabelled |

Deliverables: corrected registry + case descriptions under
`analysis/registry_v1/`, independent RDKit cross-check, `ERRATA.md` appended to
the predecessor's interpretation (the predecessor's frozen files are **not**
rewritten). Impact scope must be established explicitly: the corrected functions
are display-only in MSDS-v0 (`describe_key` / `atom_label` / `bond_label` feed the
registry CSV, case JSON and figure titles only). This is verified by
(a) a static call-graph check, (b) recomputing the design matrices and hashes and
comparing with the frozen artefacts, (c) re-verifying the frozen vocabulary
selection from raw indices. If any model-input path is affected, the affected
scope is reported and no old result is silently declared valid; a full retrain is
**not** started without explicit authorisation.

## 5. Phase 0.2 — three kinds of "out of vocabulary" (statistics only)

Using only the MSDS-v0 enumeration cache, the frozen vocabulary and Train document
frequencies, every FULL **instance** falls into exactly one class:

1. `in_vocab` — its FULL type is a frozen FULL column;
2. `train_seen_but_not_frozen` — Train document frequency > 0 but no column; reason
   assigned deterministically by the first matching gate of the frozen selection:
   `parent_coarse_not_retained` → `below_min_document_frequency(<5)` →
   `below_min_scaffold_frequency(<3)` → `top_k_truncated_by_family_cap(1024)`
   (co-occurrence counted separately);
3. `train_never_seen` — Train document frequency 0.

Reported per structural family and per split: instance share, molecule coverage,
and COARSE-fallback coverage (a FULL type whose COARSE parent is a frozen COARSE
column still contributes to the COARSE block). The class-2 split is **not** called
"truly unseen"; §8 forbids that reading.

## 6. Phase A — XGBoost gain attribution (MolHIV)

### 6.1 Blocks

| block | columns | definition |
|---|---:|---|
| `H` | 2 258 | frozen historical ECFP4 binary + RDKit descriptors, historical Train-only preprocessing |
| `C` | 754 | frozen COARSE counts, `count / sqrt(max(n_atoms,1))` |
| `F` | 3 652 | frozen FULL counts, same scaling |
| `MC` | 2 048 | folded Morgan **count** fingerprint, same radius/invariants/chirality/bond-type settings as the binary block, `count / sqrt(n_atoms)`; `MC > 0` must equal the frozen binary support, every mismatch is reported |
| `MU` | 4 406 | **unfolded** Morgan counts (`GetMorganFingerprint`, radius 2), Train document-frequency top-K, K = 4 406, ties by environment id, `count / sqrt(n_atoms)` |
| `WL` | 4 406 | attributed rooted 1-WL over all 9 OGB atom + 3 OGB bond attributes, rounds 0/1/2, top-K = 4 406 `(round, signature)` by Train document frequency, ties by signature bytes, `count / sqrt(n_atoms)` |
| `CF-PRES` | 4 406 | the same 4 406 frozen C+F types **binarised** (`count > 0`) then divided by `sqrt(n_atoms)`; identical vocabulary and identical molecule-size scaling as C+F, repeat multiplicity removed |

Declared boundaries (up front, not post hoc): `MU` vs `MC` differ in *both*
collision behaviour and effective dimensionality (K identical here, but MC is a
folded 2 048-bucket map and MU an unfolded 4 406-entry map); `C` and `F` are not
dimension-matched; all arms share `colsample_bytree = 0.8`, so a different column
count changes the candidate split space; feature gain is an internal usage
statistic and never an AUC contribution.

### 6.2 Arms (six new fits only)

| arm | input | status |
|---|---|---|
| X0 | H | reused (historical predictions) |
| X-MC | H+MC | new |
| X-MU | H+MU | new |
| X-WL | H+WL | new |
| X-C | H+C | new |
| X-F | H+F | new |
| X-PRES | H+CF-PRES | new |
| X-CF | H+C+F | reused (MSDS-v0 Phase D) |

Recipe (frozen, identical for all six): `n_estimators=500, max_depth=5,
learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
min_child_weight=1, tree_method=hist, random_state=0, device=cpu`, dense float32
input (`missing` stays NaN; after the historical Train-median imputation no cell is
NaN, so every explicit zero keeps the value 0 — sparse input would silently treat
un-stored zeros as missing, as measured in MSDS-v0). Train on Train only; predict
Train/Valid/Test; no early stopping, no tuning, no vocabulary change.

### 6.3 Report and contrasts

Per split (Train/Valid/Test): ROC-AUC (sklearn + `ogb.graphproppred.Evaluator`
cross-check), average precision, plus paired **scaffold** bootstrap (1 000
resamples, seed 0, whole Bemis–Murcko scaffold groups) of the ROC-AUC and AP
differences:

`X-MC − X0`, `X-MU − X-MC`, `X-CF − X-MU`, `X-CF − X-WL`, `X-CF − X-C`,
`X-CF − X-F`, `X-CF − X-PRES`.

A single contrast never certifies a unique mechanism. `SPLIT_CONFLICT` (Valid and
Test point in opposite directions) is reported as such, never resolved by picking
the favourable split.

## 7. Phase B — MLP readout on identical inputs (MolHIV)

Arms: `N0` = H, `N-WL` = H+WL, `N-CF` = H+C+F. These inputs are column-identical
to `X0`, `X-WL`, `X-CF`.

Architecture (fixed, one class, no encoder / no GNN / no aggregation):

```
input -> Linear(128) -> ReLU -> Dropout(0.1) -> Linear(128) -> ReLU -> Dropout(0.1) -> Linear(1)
```

For `N-WL`/`N-CF` the first layer is `h = W_H·H + W_Q·Q + b` with `W_Q`
zero-initialised; `W_H`, `b` and every later layer use exactly the same
initialisation across arms (same RNG stream, same draw order → identical shared
tensors). Parameter counts and an initialisation hash are recorded; the extra
branch adds parameters, so capacity is **not** fully matched. `N-WL` and `N-CF`
have near-identical extra-feature budgets (4 406 vs 4 406).

Preprocessing: descriptor columns of `H` (columns 2048:, 210 columns) standardised
with Train mean/std and clipped to `[-10,10]`; the 2 048 fingerprint bits stay
0/1; structural count blocks keep the §6.1 scaling (no per-type renormalisation).
All arms use the same base treatment.

Training: seed 0, 50 epochs, batch 256, AdamW `lr=1e-3`, `weight_decay=1e-4`,
class-balanced BCE with weights computed on Train only and normalised to mean 1,
gradient clipping 5, no scheduler, no early stopping, identical batch order across
arms, full Train/Valid/Test evaluation (ROC-AUC, AP, loss) every epoch. Checkpoint =
best Valid ROC-AUC, ties to the earlier epoch. Curves, best checkpoint and
per-molecule predictions are stored. No architecture or regularisation change is
made after seeing the result.

Core reading (not the absolute level): `X-CF − X0` vs `N-CF − N0`,
`X-WL − X0` vs `N-WL − N0`, and `N-CF − N-WL`; plus whether Train fit improves
while Valid/Test do not, and whether the per-molecule improvements of XGB and MLP
overlap on the same molecules. A fixed-statistics MLP is **not** end-to-end
subgraph representation learning and is never described as such.

## 8. Phase C — minimal ZINC transfer

Data: PyG `ZINC(subset=True)` official 10 000/1 000/1 000. Primary metric: MAE in
the units of the dataset-provided label (`y`; PyG's ZINC label is upstream
log-modulus transformed — stated explicitly, and the same convention as every
historical ZINC reference, so the comparison is unit-consistent).

`Z` (fixed base representation, Train-fitted tables only):
`n_atoms`, `n_bonds`, raw node-class counts, raw edge-class counts, and all-atom
rooted **typed** WL counts for rounds 0/1/2/3 with Train document-frequency
top-2048 per round (all if fewer); count columns divided by `sqrt(n_atoms)`.
Node/edge classes are the dataset's own (21 observed node classes, 3 observed edge
classes); no synthetic OGB-style attribute is invented and no attribute is
back-derived from the label.

`Z-CF` (this round's new structure block): the predecessor's exact enumeration —
all connected induced 2/3/4-node subgraphs and all chordless 5/6-cycles — with
`COARSE` = topology only and `FULL` = topology + the raw ZINC node/edge classes,
FULL→COARSE by deterministic projection. Train-only document-frequency gates and
per-family caps as in MSDS-v0 (COARSE: df ≥ 20, ≤ 256/family; FULL: df ≥ 5,
≤ 1 024/family, parent must be retained); the successor/child scaffold gate is
**dropped** because ZINC has no reliable scaffold identity, keeping only the
document-frequency gate. Counts divided by `sqrt(n_atoms)`.

Arms: `ZX0` (XGB, Z), `ZX1` (XGB, Z+Z-CF), `ZN0` (MLP, Z), `ZN1` (MLP, Z+Z-CF).
XGB recipe = §6.2 with `objective='reg:absoluteerror'`; MLP = §7 architecture and
recipe with MSE-free MAE loss, target standardised with Train mean/std and
evaluated after inverting the standardisation. Every epoch evaluates
Train/Valid/Test MAE; checkpoint = best Valid MAE. No fifth model.

Reported: `ZX1 − ZX0`, `ZN1 − ZN0` (positive = MAE decrease), paired molecule
bootstrap on existing predictions, and the explicit statement that no
training-seed stability evidence exists. Historical patch/path models are quoted
as a position reference only. Cross-dataset statements compare the *phenomenon*
(does an explicitly enumerated structure block help, and for which readout), never
the chemical object identity: MolHIV and ZINC attribute granularities differ.

## 9. Phase D — understanding the signal (no new model)

1. **Which structure classes matter** — combine X-C / X-F / X-PRES with the gain
   and split-threshold statistics of the fitted trees: usage per family
   (size-2/3/4, cycle-5/6), high-frequency vs low-support vs out-of-vocabulary
   coverage, and representative decision paths. Tree importance and single paths
   are descriptive diagnostics, never independent chemical contributions.
2. **Which molecules improve** — per-molecule prediction differences by atom-count
   bin (Train-defined edges), by FULL in-vocabulary coverage, by the share of
   genuinely Train-unseen FULL instances and by the share of Train-seen-but-filtered
   ones; sample sizes, positive/negative counts and paired bootstrap intervals;
   within-size comparisons so that "molecule is large" is not read as "OOV effect".
3. **Cases** — few, not dozens: 3 well-supported specific structure types, 3 real
   coarse/fine families, 3 molecules where explicit structure helps, 3 where it
   hurts, each with molecule id, SMILES, label, prediction, match location and
   correctly decoded attributes. Improvement/degradation cases are post-hoc
   diagnostics, not validation.

## 10. Budget, execution, stop rules

* Wall clock hard limit 4 h; GPU hard limit 4 A100 GPU-hours; new XGB ≤ 8
  (6 MolHIV + 2 ZINC); new MLP ≤ 5 (3 MolHIV + 2 ZINC); each input cache built
  once. Reused arms do not count and are never retrained.
* Execution order: Phase 0 → MolHIV A → MolHIV B → ZINC C → D + report.
  CPU feature preparation and GPU MLP may run concurrently once dependencies are
  satisfied.
* All GPU work runs through `rr` (`res-gpu1` first, `res-gpu0` in parallel when
  free); CPU XGBoost also runs through `rr` on the `res-local` pool so that the
  execution regime is recorded and the local workstation (27 GB RAM, ~9 GB free)
  cannot OOM a 6 664-column dense design. A run is only reported if it completes
  its full fixed epoch/tree budget; a truncated run is never presented as a
  matched contrast.
* If the budget runs short, the complete MolHIV matched contrasts are finished
  before ZINC starts.

## 11. Engineering-deviation log (filled during execution)

Only real implementation bugs may be fixed; each one is recorded here with its
affected runs and the reason it cannot change a reported number.

| # | what happened | fix | affects a reported number? |
|---:|---|---|---|
| 1 | `rr run -- python -m <module>` runs from the repository root, so the round's `code/` was not importable and the first launches died with `ModuleNotFoundError` | commands now run as `env PYTHONPATH=<round>/code python -m <module>` | no — those jobs failed before writing any file |
| 2 | the remote checkout carried an untracked MSDS-v0 `analysis/phaseE/` directory, which blocked `rr deploy` | moved to `/home/hxy/sga1_orphans/msds_v0_phaseE_remote_only_...` (preserved, not deleted) | no |
| 3 | the unfolded-Morgan check initially folded only the Train Top-4406 environments, producing a spurious "MC ≠ fold(MU)" warning | the identity is checked on **all** environments before truncation; it holds exactly | no — only the verification was wrong |
| 4 | `json.dump` failed on `np.bool_` values in the meta files | explicit Python-native casts at the write sites | no |
| 5 | `_family_label`/`canonical_key` returned a tuple where a string was expected in the Phase D case table | fixed before the case table was written | no |
| 6 | the COARSE decoder was fed FULL-length attribute rows | `decode_atom` now takes an explicit granularity and reads only the columns that exist in that representation | no (display only, but it would have mislabelled every COARSE case) |
| 7 | ZINC's processed files are `train.pt` / `val.pt` / `test.pt` (not `valid.pt`) | loader tries both names | no |
| 8 | the ZINC design stored canonical keys as npz object arrays | stored as 64-bit digests of the canonical key, with deterministic tie-breaking | negligible (collision ≈ 1e-8 over 2 335 types; no anomaly observed) |
| 9 | `[dense(b) for b in extra and blocks]` silently dropped the second block when `extra` was a single-element list | rewritten as an explicit loop; verified against the frozen design matrix | yes — it was caught by the C/F rebuild check before any model ran |
| 10 | the ZINC paired bootstrap reported `MAE(a) − MAE(b)`, i.e. the opposite sign from MolHIV | sign convention unified to `delta = MAE(b) − MAE(a)` (positive = arm *a* better) and the analysis re-run from the stored per-molecule predictions | yes — the sign was flipped in one intermediate table; the re-run replaces it, and `REPORT.md` quotes the re-run |
| 11 | the AST call-graph scan for "is this decoding display-only?" initially flagged every caller transitively, including `build_inputs` | the rule was narrowed to "the enclosing function's only outputs are CSV/JSON/plot labels", and the conclusion was re-derived from three independent artefacts (no string arrays in the frozen files, vocabulary re-derived from raw integer indices, design matrix rebuilt bit-identically) | no |
| 12 | the Phase D `X-CF − X-MC` row initially printed the AP delta with the wrong sign in an intermediate table | recomputed from `analysis/contrasts_molhiv.json` | no (REPORT quotes the recomputed value) |

## 12. Pre-flight verification checklist (all machine-checked)

1. official split sizes/ids and graph-index alignment (MolHIV, ZINC);
2. X0 / X-CF per-molecule predictions align row-by-row with the official split;
3. every new vocabulary/column order fitted on Train only; the frozen C/F rebuild
   is bit-identical to the MSDS-v0 design matrix;
4. Morgan binary/count correspondence (`MC > 0` ≡ frozen binary support) and the
   fold rule;
5. existence/count conversion: `CF-PRES` values are exactly `1/sqrt(n_atoms)` and
   its zero pattern equals `(C+F) > 0`;
6. ZINC features use the dataset's own node/edge classes (no synthetic attributes);
7. matched MLP arms share initialisation (tensor-hash comparison) and only the
   zero-initialised extra branch differs;
8. PyG batching never accumulates `graph_index` into an offset (the predecessor
   hit exactly this: `Data.__inc__` adds offsets to any attribute whose name
   contains `index`; SGA-v1 never batches whole `Data` objects — minibatches are
   assembled from dense arrays by row id);
9. every reported metric is recomputed independently from stored per-molecule
   predictions.
