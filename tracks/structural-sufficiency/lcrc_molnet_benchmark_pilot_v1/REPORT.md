# LCRC-A standard MoleculeNet benchmark — Sprint 01

Round id: `lcrc-molnet-benchmark-pilot-v1`
Scope: four frozen datasets, one training seed, one split repeat, one checkpoint per
dataset, one one-shot Test read. Nothing else was run.

---

## A. Benchmark table (single seed 0, split repeat index 0)

| Dataset | Task / metric | Split | n (train/valid/test) | Train | Validation | **Test** | Test secondary | Best epoch | Test CI-free note |
|---|---|---|---|---|---|---|---|---|---|
| ESOL | regression / RMSE ↓ (logS) | random, repeat 0 | 902 / 113 / 113 | 0.4430 | 0.5585 | **0.5226** | MAE 0.3961, R² 0.9312 | 136 / 360 | n_test = 113 |
| Lipophilicity | regression / RMSE ↓ (logD) | random, repeat 0 | 3360 / 420 / 420 | 0.2831 | 0.7109 | **0.5727** | MAE 0.4256, R² 0.7714 | 240 / 360 | valid ≫ test |
| FreeSolv | regression / RMSE ↓ (kcal/mol) | random, repeat 0 | 513 / 64 / 65 | 0.6420 | 0.7594 | **1.0870** | MAE 0.7126, R² 0.9199 | 116 / 360 | n_test = 65 |
| BACE | binary classification / ROC-AUC ↑ | scaffold, repeat 0 | 1210 / 151 / 152 | 0.9657 | 0.7761 | **0.8254** | PR-AUC 0.8516, acc@0.5 0.7171 | 102 / 360 | no threshold selection |

All four rows: LCRC-A, 247,106 parameters, frozen `lipo_v1` structure spec
(`7270f767…515ee`), frozen R3 recipe (AdamW, peak LR 3e-4 → min 1e-5 per-step cosine,
wd 1e-3, batch 128, dropout 0.10 on `h`, grad clip 5.0, 360 epochs, no EMA/SWA/soup/SSL,
best-Validation checkpoint). Machine-readable copy: `analysis/benchmark_table.csv`.

BACE numbers come from sigmoid probabilities; accuracy is reported at 0.5 for
completeness only and was never used for any selection.

---

## B. Positioning per dataset

Exact index matching is impossible for every dataset: no published source releases the
per-example split indices behind its numbers, so the strongest achievable grade is B.
**Verdict: `NO_EXACT_MATCH_FOUND`** (`analysis/literature_matrix.csv`,
`analysis/literature_sources.json`; all numbers transcribed from hash-pinned fetched
documents, locators in the CSV).

### ESOL — random split, RMSE
* **B** MoleculeNet (Wu 2018): MPNN 0.58, XGBoost 0.99 — same dataset, same random
  80:10:10 protocol, same metric; indices not published.
* **C** GROVER (Rong 2020): GraphConv 1.068, D-MPNN 0.980, AttentiveFP 0.853,
  GROVER-large 0.831 — GROVER applies **scaffold** splitting to all datasets, so these
  measure a harder regime than this round.
* This round 0.5226 sits below the MoleculeNet random-split references, i.e. the frozen
  local architecture is competitive on the easy regime it was tuned for. Single seed, so
  this is a point estimate, not a distribution.

### Lipophilicity — random split, RMSE
* **B** MoleculeNet: GraphConv 0.655, XGBoost 0.799 — protocol match, indices unpublished.
* **C** GROVER: D-MPNN 0.653, AttentiveFP 0.650, GROVER-large 0.560 (scaffold).
* This round 0.5727 is the most competitive of the four rows against random-split
  references, with a large Validation→Test inversion (0.711 → 0.573) showing the 420-row
  Validation draw is harder than the 420-row Test draw — a split-noise effect that a
  single seed cannot resolve.

### FreeSolv — random split, RMSE
* **B** MoleculeNet: MPNN 1.15, XGBoost 1.74.
* **C** GROVER: AttentiveFP 2.030, D-MPNN 2.177, GROVER-large 1.544 (scaffold).
* This round 1.0870 is below both random-split references, but n_test = 65, so a single
  molecule moves RMSE by ~0.1; treat as weak evidence.

### BACE — scaffold split, ROC-AUC
* **B** MoleculeNet: RF 0.867, Weave 0.806 (scaffold, 1522-compound paper table).
* **B** GROVER (1513 compounds, scaffold, 3 seeds): GraphConv 0.854, D-MPNN 0.852,
  AttentiveFP 0.863, GROVER-large 0.894.
* This round 0.8254 is the honest weak spot: it clears the MoleculeNet graph baseline but
  trails the scaffold-split GNN references by ~0.03–0.07. LCRC-A has no message passing,
  so a scaffold shift removes exactly the signal that neighbourhood aggregation would
  recover. This is the one dataset where the local architecture should not be defended.

---

## C. Next steps (at most two)

1. **Add seeds 1–2 on the same frozen splits and recipe** (≈0.35 A100-hours at the
   measured 0.175 h/seed). Every conclusion above is currently a point estimate; the
   Lipophilicity valid/test inversion and FreeSolv's 65-row test set make error bars the
   single highest-value addition. No config changes, no new datasets.
2. **Only after (1): run one protocol-matched cheap baseline on exactly these split
   indices** (e.g. ECFP4 + gradient boosting trained and evaluated by the same harness).
   This separates "LCRC-A is strong" from "this split draw is easy" at negligible cost,
   and it is the only way to tell whether the BACE gap is architectural or a
   split-difficulty artefact.

Not done, and not to be started without a new decision: BBBP/Tox21/HIV, HP search, extra
descriptors, pretraining, GNN training, a second split repeat.

---

## D. Execution record

### Status
| Item | Value |
|---|---|
| Formal runs | 4 / 4 complete (ESOL, Lipophilicity, FreeSolv, BACE), seed 0 only |
| Retries | 1 engineering failure, no result: the first GPU smoke tripped a `1e-9` reload check (CUDA reduction order, 2.1e-8). Fixed by a pre-registered `1e-5` tolerance *before* any formal run; documented in `BENCHMARK_PROTOCOL.md`. |
| Engineering smokes | 2-epoch ESOL+BACE smoke on CPU (local) and ESOL smoke on GPU; excluded from results |
| Test reads | 1 per dataset, after commit `d5bcba11` froze all four checkpoints |
| Test used for any selection | never (`test_used_for_any_selection: false` in every `train.json`/`test_eval.json`) |
| GPU budget | **0.1747 h of 1.5 h** (ESOL 0.020, Lipophilicity 0.099, FreeSolv 0.011, BACE 0.045) |

### P0 audit (CPU, before any training) — `analysis/p0_audit.json` = PASS
All four datasets pass gates G0–G9:

* **G0 source pinned**: `shenwanxiang/ChemBench` commit
  `0b10d4c1ac72b6929503694a5f60446824736391`, 16 files vendored byte-for-byte
  (`analysis/source/chembench_source_manifest.json`).
* **G1 split integrity**: published index files only, repeat index 0 = the
  lexicographically first published repeat (suffix `122`), frozen before any label was
  read. Partitions disjoint, complete, sizes match (902/113/113, 3360/420/420,
  513/64/65, 1210/151/152). BACE repeats 122/123/124 are byte-identical (sha
  `db67f92c…`), so repeat choice is immaterial there.
* **G2 no silent drops** + raw-artifact fidelity: zero SMILES parse failures; the
  committed label-free attribute artifact equals a fresh RDKit derivation for every row
  and every field.
* **G3 row alignment**: cache row ids ↔ published CSV rows ↔ split indices verified;
  test structures are built label-free and re-checked at P2.
* **G4 label-free caches**: Train/Validation caches are constructed with
  `usecols=["smiles"]`; test labels are not read until P2.
* **G5 vocabulary coverage**: **zero** UNK atoms and **zero** UNK bonds on all four
  datasets.
* **G6 model identity**: 247,106 parameters, no attention/message-passing/conv modules,
  radius-2 locality audit passes, single-vs-batch forward diff 0.0, every parameter
  receives a gradient.
* **G7 numerics**: deterministic init and shuffle generator (seed 917000), fp32.
* **G8 train-only normalisation**: target mean/std fitted on Train only; recomputation
  from Train reproduces, from Validation does not.
* **G9 targets/class balance**: BACE train 695/515 (positive rate 0.4256), valid 67/84 —
  both classes present; targets not z-scored for classification.

### SHA256 of the frozen evidence
```
17229c7b60ad504309c1bfb2883d9e162db99bfe15c3a88fcfbe46493f8d33aa  analysis/dataset_manifest.json
07020e4968a827d8e99cfc206a387e3d025f9f4794de547ec1e7d0cdc3a97a85  analysis/p0_audit.json
62ae11a563fce865a35a4246675e65e374d79230834120d11a63b685320d12a5  analysis/protocol_freeze.json
7daf473b1d049141d59b4f3d64e1363c48d9c40cce681c4eccb3e483ae6b0ba4  analysis/checkpoint_manifest.json
7bc684148c3f5ce7a4f7b4f89bbff01e7d96f2bab447820f99fcd689efcf397c  analysis/benchmark_table.csv
fe00ed298129915e6b7ef444ac3bcd71eff6d84e82e231b61df2987145188f59  analysis/literature_matrix.csv
8155d2a15185946f7d88737e694589af2065977f7e7ad46812f427a81c582dac  analysis/gpu_ledger.json
63ea9a8ec20c28cbe7885720eb19bffb527f799a688a3721036e13241c1aeb09  analysis/summary.json
1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be  analysis/source/raw_attributes_v1.json.gz
0d1a3aff16db514bc81aed555eed966e29f90ceeb4408e33d722cfc83a97466c  analysis/source/raw_artifact_manifest.json
```
Split-row hashes: ESOL `b7459354…/8fbc7db0…/fafa6687…`, Lipophilicity
`4e8d42a9…/31e3f28f…/a4d92ad9…`, FreeSolv `19074e52…/788218be…/229542e8…`, BACE
`5f36ea5e…/b44a8e9e…/ea331b0b…` (train/valid/test; full digests in
`analysis/split_manifest_<dataset>.json`).

Per-run checkpoints: ESOL `894340da…`, Lipophilicity `b1dd1f5a…`, FreeSolv `6ee0f1f2…`,
BACE `f4c434b9…` (full table in `analysis/checkpoint_manifest.json`).

### Execution regime
* Training host `a100-2` via `rr` (`res`, process backend, pools `res-gpu0`/`res-gpu1`),
  one A100-SXM4-40GB per run, driver 550.163.01, torch 2.5.1+cu124, CUDA 12.4,
  Python 3.12.14, fp32. Peak allocated memory 57 MB (ESOL) to 107 MB (Lipophilicity).
* Code revision at run time: `d91e496f` (train) and `d5bcba11` (frozen checkpoints,
  written before P2); worktree clean, `git_dirty_tracked: false` in every artifact.
* P2 Test evaluation ran once per dataset on CPU from the frozen checkpoints
  (device `cpu`, charged 0 GPU-seconds), after `git commit d5bcba11` froze the manifest.
* `res-2` (Slurm pool `res2-cu124`) was tried first and abandoned: jobs sat in
  `PENDING (Resources)` and then reported `ReqNodeNotAvail`, so the round moved to `res`.
  Job ids and states are recorded under `.rr/` and in `analysis/gpu_ledger.json`.

### Known limitations (deliberate, not fixed)
1. **One seed.** Every number is a single draw; no CIs, no seed spread.
2. **ESOL label leakage by duplication.** The published random repeat-0 split places 4
   canonical-SMILES duplicate groups across Train/Test (rows
   `[[147,779],[450,1019],[680,1069],[701,825]]`). The published split was used as-is;
   repairing it would have made the round incomparable to any published ESOL number.
3. **FreeSolv test set is 65 molecules**; one molecule moves RMSE by ~0.1.
4. **Within-protocol comparison only.** All literature rows are grade B or C; nothing in
   this round is an exact split match, and single-seed numbers should not be placed in a
   mean±std column next to 3-seed published means.
5. **Old-track numbers are not comparable.** `lcrc_esol_quickstart_v1` used a custom
   Bemis–Murcko scaffold split (seed 20261103) and
   `lcrc_lipo_mechanism_generalization_ssl_v2` used DeepChem 2.8.0 `ScaffoldSplitter`;
   both are grade C against this round's random repeat-0 split (train overlap ≈ 80.2 %
   for ESOL, ≈ 79.8 % for Lipophilicity). They were read-only inputs and were not reused
   as reference numbers.

---

## Reproduction

```bash
# CPU: source integrity, raw artifact, P0 audit, protocol freeze
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_raw_artifact.py --check
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_p0.py
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_protocol_freeze.py

# GPU: one training run per dataset (frozen recipe, 360 epochs)
rr run res molnet-train-ESOL --pool res-gpu0 --gpus 1 --cpus 4 --mem 16G --time 01:00:00 \
  --result tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/runs/ESOL_s0 \
  -- python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_train.py --dataset ESOL --device cuda

# freeze all checkpoints, commit, then one Test read per dataset
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_freeze.py
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_eval.py --dataset ESOL --device cpu
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_tables.py
uv run python tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/code/mb_literature.py
```
