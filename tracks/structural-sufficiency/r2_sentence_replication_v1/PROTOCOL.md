# R2-SENTENCE-REPLICATION-v1 — protocol

Round: `r2_sentence_replication_v1`
Version: 2026-11-23/v1, **frozen before any formal GPU run**
Task: `ogbg-mollipo` whole-molecule lipophilicity regression
Primary metric: whole-molecule held-out RMSE in original label units
Model under test: the existing `SENTENCE_BASE` trunk of `graph_word_incidence_composition_pilot_v0`
Runs: exactly two (`D0_S1`, `D1_S0`). No architecture search, no retries.

This protocol only covers re-running an already frozen model. It introduces no
new model, no new module, no new feature, no new target and no new arm.

## 1. Question

GWIC-v0 measured `SENTENCE_BASE` **once**: D0 split, seed 0, Dev RMSE **0.5506**.
That number is a single training seed on a single frozen scaffold split. This
round asks whether it is a stable coordinate for the trunk:

1. does a second training seed on the frozen D0 split support 0.5506?
2. does the same frozen model on the frozen D1 scaffold split stay close to the
   historical GINE D1 coordinate (0.5534)?
3. is there obvious seed or split sensitivity?
4. can this model now be fixed as the non-GNN baseline for later algorithm
   experiments?

D1 is a **reproducibility check**, not an independent blind test: it reuses
official Train rows only, and the split file was built and frozen in
`structural_path_composition_route_v1` before this round started.

Out of scope (prohibited by the brief): `WORD_REAL`, `WORD_GEOM_NULL`, BBPA,
PSC, any word module, any new feature, any hyper-parameter change, official
Valid/Test labels, a second dataset, ensembles, dictionary learning, extra
architecture screening.

## 2. Frozen model

`SENTENCE_BASE` is **not re-implemented** here. The entry point imports the
GWIC-v0 sources byte-for-byte and only selects the arm flag `use_words=False`
that GWIC-v0 already defines:

* `gwic_model.GwicModel(..., use_words=False)` / `build_model(..., use_words=False)`
* full OGB atom (9 fields) and bond (3 fields) categorical vocabulary
* non-iterative S1 / typed-rooted-S2 / complete-R2 sentence trunk, `h_v in R^96`
* frozen graph readout `[h_sum, h_mean, h_std, a_mean, a_std, b_mean, b_std, g]`
* **416,177** parameters, no word encoder, no `Q`, no incidence geometry
* frozen `init_parameters(seed, last_std=1e-3)`, same embedding table sizes,
  same widths, same dropout 0.1, same code path as GWIC-v0

Frozen sources and their pin (sha256 of the file bytes, checked at runtime by
`r2_common.assert_frozen_sources()`):

| file | sha256 |
|---|---|
| `.../graph_word_incidence_composition_pilot_v0/code/gwic_common.py` | `6d081769940ed00682c95c8a7c8be382f1592763db269c3067fc14bc3f1a97cb` |
| `.../code/gwic_data.py` | `f02863be6e9f37a593e234c4cb7c5e1d9a760088be0c903aad36619a46aa0cef` |
| `.../code/gwic_structure.py` | `97e2e35b272e36043e03a476f58fb6a2edeb3beefc60b15e967c1096c65ecb56` |
| `.../code/gwic_tensor.py` | `f969c1fd47255b3b4b897f540f501671def42ec45ec9515751ca7288204bedd0` |
| `.../code/gwic_model.py` | `66926facafc97935905ba32ae4771f2a15585a34e0749f26cdb2fb8dff187e73` |
| `.../code/gwic_train.py` (reference only) | `fa1031cee6457864c9b99cac387d97dc97004cf4b79f8e329946a5cb089c855d` |
| `.../configs/gwic.json` | `9ed13ab795e7ae8c9e2c447d74a452fa9b6a8e0c3c6c6990b578e1f963a7e090` |

Aggregate `code_sha` of the six training modules: `a58f942ebf4050114987c58c2cfeaec75f2cf42c6fc945c21bb45030813294a1`
(the value recorded in the GWIC-v0 `SENTENCE_BASE_S0` manifest).
The GWIC-v0 directory is **read-only** in this round: no file there is edited,
moved or regenerated, and no GWIC-v0 result is overwritten.

## 3. Frozen training recipe (verbatim from GWIC-v0)

| item | value |
|---|---|
| optimiser | AdamW |
| peak LR / min LR | 3e-4 / 1e-5 |
| weight decay | 1e-4 |
| gradient clip | 5.0 |
| batch | 64 molecules |
| schedule | 5 warmup epochs, cosine to 1e-5 |
| max epochs | 240 |
| minimum epochs | 80 |
| early-stop patience | 45 epochs without Dev improvement |
| objective | whole-molecule standardised MSE only |
| checkpoint selection | best Dev RMSE |
| evaluation batch | 256 |
| search | none |

The D1 run uses this **complete** schedule; the D0 best epoch is **not** used to
shorten or lengthen D1 training, and no best-epoch transfer of any kind is
applied.

## 4. Data and splits

| item | value |
|---|---|
| raw structure cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`, sha256 `fe002287…23c7c` |
| raw members read | whitelist `node_ptr, bond_ptr, atom_feat, bond_feat, bond_u, bond_v, degree, min_ring, in_cycle, graph_n_*, smiles, row_ids, train_idx, valid_idx` — no `labels`, no `test_idx` |
| Train labels | `structural_path_composition_route_v1/data/labels_official_train.npz`, sha256 `a44d29de…14c2` (3360 rows) |
| D0 split | `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json`, identity `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` |
| D0 fit / Dev | **2856 / 504** (fit rows sha `7edd3c68…86c0`, Dev rows sha `eacce8ac…7193`) |
| D1 split | `structural_path_composition_route_v1/data/split_train_sub_dev_d1.json`, identity `c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee`, file sha256 `bdcd3f5884c66015aa099bd9c32b45cd6717b0d08b35a6a6c7ef44635fe61fcc` |
| D1 fit / held-out | **2856 / 504** (fit rows sha `b9f74030…ebbc`, held-out rows sha `342c1de0…30a6`) |
| official Valid | structure only, **labels never loaded**; every Valid label is NaN in the training object |
| official Test | **never touched**: no structure, no label, no statistic, no metric |

D1 identity is re-verified in this round, not assumed:

* both sides are subsets of the 3360 official Train rows and together cover it
  exactly (no Valid/Test row can enter);
* fit ∩ held-out = ∅;
* Bemis–Murcko scaffolds (`includeChirality=False`, acyclic molecules keyed by
  their own SMILES) are **disjoint** between fit and held-out (0 shared keys,
  1318 / 267 scaffold groups);
* the stored `scaffold_by_row` map is recomputed from the SMILES strings and must
  match exactly;
* the split is regenerated from the frozen algorithm (`seed=20261122`,
  `target_heldout=504`, sorted scaffold keys → `default_rng(seed).permutation`
  → greedy whole-group assignment) and must reproduce the same fit/held-out row
  sets.

### 4.1 Per-run statistics

Every run recomputes, on **its own Fit rows only**:

* target mean/std (z-score for labels, inverted for all reported metrics);
* OGB atom/bond categorical support (unsupported embedding rows zeroed);
* graph-level scalar standardisation (8 deterministic graph scalars);
* atom-level local scalar standardisation (5 deterministic atom scalars).

### 4.2 One deliberate, recorded deviation from the v0 *implementation*

GWIC-v0's protocol text says "fit-only-standardised atom/R2 scalars", but its
`gwic_data.load_bundle()` implementation computed the atom-scalar mean/scale over
the atoms of `fit ∪ dev` (both label-free structure sets). This round follows the
v0 **protocol text** and the brief's hygiene rule: the atom-scalar normaliser is
computed on the **current Fit rows only**.

This is the only numerical difference from the GWIC-v0 training path, it is
label-free, and its size is measured and reported (`analysis/checks.json`,
checks `C3`, `D6`): the resulting per-atom standardised scalar shift on Dev atoms
is at most `4.5e-3` standardised units (mean `1.6e-3`). To keep the change
auditable rather than merely asserted, the entry point also keeps the exact v0
mode (`--atom-scalar-mode v0_fit_dev`) and the engineering checks use it to prove
that the new entry reproduces the frozen GWIC-v0 training trajectory and the
frozen checkpoint replay exactly (checks `D5`, `D6`).

No other data statistic, tensor field, module, width, initialisation or
optimiser setting differs.

## 5. Entry point and artefacts

```
code/r2_common.py    pinned hashes, frozen-source import, paths, IO
code/r2_data.py      split-parameterised loader (D0/D1) built on frozen gwic_data helpers
code/r2_train.py     one arm, one split, one seed per process
code/r2_checks.py    pre-training engineering checks -> analysis/checks.json
code/r2_analyze.py   main table + summary + verdict from the pulled manifests
tests/test_r2.py     fast CPU unit tests
```

Per run (`results/<run_id>/`): `manifest.json`, `curves.csv`, `state_best.pt`,
`state_final.pt`, `predictions_best_fit.{csv,npz}`,
`predictions_best_dev.{csv,npz}`, `train.log` — the same schema as GWIC-v0.

Run ids: `D0_S1` (split D0, seed 1), `D1_S0` (split D1, seed 0).

## 6. Run matrix

| run id | split | seed | model |
|---|---|---|---|
| `D0_S1` | D0 (`60ac13c0…`) | 1 | `SENTENCE_BASE` |
| `D1_S0` | D1 (`c94e2d4b…`) | 0 | `SENTENCE_BASE` |

No third run. No retry, no re-run on a bad result, no per-split tuning, no
selection among runs.

## 7. Engineering checks (must pass before the formal runs)

`code/r2_checks.py` writes `analysis/checks.json`; all checks must be PASS.

**A — frozen identity.** A1 GWIC source files byte-identical to the pins and
aggregate `code_sha` equals the GWIC-v0 manifest value; A2 frozen `configs/gwic.json`
sha and its `dims`/`recipe`/`SENTENCE_BASE` spec are the ones actually used;
A3 raw cache, Train-label file, D0 split and D1 split hashes match the pins.

**B — split identity.** B1 D0 sizes / disjointness / exact coverage of official
Train / row hashes equal the GWIC-v0 manifest; B2 D1 the same plus scaffold map
recomputation and frozen-algorithm regeneration; B3 label isolation: all Valid
labels NaN, no `labels`/`test_idx` member read, no finite label outside official
Train.

**C — fit-only normalisation.** C1 target scaler depends only on Fit labels
(mutating Dev labels leaves it unchanged) and differs between D0 and D1;
C2 graph scalar mean/scale equal the Fit-only recomputation and differ from the
all-row statistic; C3 atom scalar mean/scale equal the Fit-only recomputation,
and the deviation from the v0 `fit ∪ dev` statistic is recorded; C4 category
support equals the Fit-row union and mutating a Dev molecule to Fit-unsupported
categories changes neither the support mask nor the (already zeroed) embedding
rows.

**D — unchanged tensor path.** D1 parameter ledger identical to the GWIC-v0
`SENTENCE_BASE` ledger (per module and total 416,177), no word/score/Q module
instantiated, `word_in_dim == 0`; D2 `state_dict` key set and shapes identical to
the frozen `SENTENCE_BASE_S0` checkpoint (44 entries); D3 seed-0 init hashes
identical to the frozen manifest; D4 replay of the frozen seed-0 checkpoint on D0
Dev reproduces **RMSE 0.5505589717057867** and the committed per-molecule
predictions; D5 the new entry point run for 2 epochs on CPU in `v0_fit_dev` mode
matches the frozen `gwic_train.py` reference run bit-for-bit (loss, gradient
norm, fit/Dev RMSE, LR, final state); D6 for a fixed batch the new loader and the
frozen v0 loader produce identical tensors in `v0_fit_dev` mode, and in the
formal fit-only mode *all* tensors except `atom_scalars` are identical.

**E — invariances.** E1 atom renumbering leaves the molecule prediction
unchanged; E2 bond reordering leaves it unchanged; E3 batch invariance: the same
molecule predicted alone, in a different batch position, and inside a different
batch composition gives the same prediction.

**F — gradients.** F1 one real optimiser step: every parameter receives a finite
gradient and every module group has a non-zero gradient norm; F2 a short
`fit`-subset overfit run decreases the loss.

**G — no additions.** G1 the collated batch key set equals the frozen GWIC-v0
key set (no new feature tensor); G2 the instantiated embedding/module names equal
the frozen `SENTENCE_BASE` names; G3 total parameters 416,177.

## 8. Evaluation

Primary: held-out (Dev) RMSE in original label units, at the best-Dev
checkpoint. Secondary: Dev MAE, Fit RMSE at the best epoch, best epoch, epochs
run, wall seconds, GPU hours.

Main table columns:
`split, seed, train_rmse, dev_rmse, dev_mae, best_epoch, epochs_run, wall_seconds, gpu_hours`.

Historical coordinates (different recipes, **not** a matched benchmark):
`SENTENCE_BASE D0 s0 0.5506`, `GINE D0 3-seed mean 0.5266`, `GINE D1 2-seed
mean 0.5534`, `ACRA D0 0.5802`, `ACRA D1 0.6376`.

### 8.1 Frozen verdict rule (engineering criterion, not a significance test)

* `R2_BASELINE_REPLICATED_PRELIMINARY` iff `D0_S1 Dev ≤ 0.59` **and**
  `D1_S0 Dev ≤ 0.61`;
* otherwise `R2_BASELINE_SENSITIVITY_UNRESOLVED`, reporting where degradation
  occurs. No tuning, no extra run, no architecture change may be used to repair
  a failed threshold.

## 9. Budget and degradation

Target ≤ 1 h wall, ≤ 2 GPU-hours (sum of per-run GPU time). Two runs only, one
process per GPU, executed in parallel through `rr` on `res` (`res-gpu0`,
`res-gpu1`). No DDP, no stacking, no retry. If a run fails for an infrastructure
reason, it is relaunched unchanged from the same commit and the relaunch is
recorded. If the budget were exceeded, the second run would still be completed
and the overrun reported — no shortening of the recipe is permitted.

## 10. Deliverables and stop condition

`PROTOCOL.md`, `REPORT.md`, `code/`, `configs/r2.json`, `tests/`,
`analysis/checks.json`, `results/main_table.csv`, `results/summary.json`,
per-run artefacts for `D0_S1` and `D1_S0`.

The round stops after those two runs and the report. Non-claims: no official
Valid/Test metric, no statistical-significance claim, no matched comparison with
the GINE/ACRA historical coordinates, no architecture improvement, no new
mechanism.