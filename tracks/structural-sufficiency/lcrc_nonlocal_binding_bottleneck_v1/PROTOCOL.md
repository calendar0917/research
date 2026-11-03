# PROTOCOL — LCRC 远距离化学角色绑定瓶颈验证 v1

Round id: `lcrc-nonlocal-binding-bottleneck-v1`
Track: `tracks/structural-sufficiency/lcrc_nonlocal_binding_bottleneck_v1/`

This file is the **frozen pre-registration** of the round: research question,
definitions, data range, splits, estimator, metrics, decision thresholds and
stopping rules. It was written before the first formal run of `code/nb_run.py`
and nothing in it is chosen from a result. Design choices are the researcher's;
this round only implements, checks, runs and reports them.

* GPU budget: **0 GPU·h**. No model is trained with gradient descent.
* CPU budget: **≤ 2 h** wall clock for the whole pipeline.
* No official Valid/Test index file, label or metric is read at any point.
* Not a hyper-parameter search: one pre-registered grid of three `C` values, one
  pre-registered estimator, one pre-registered split rule.

---

## 1. Proposition under test (fixed, not modified by this round)

LCRC-A's local path covers a bounded structural radius; its 42-d `global` block
contains only *untyped* whole-graph topology, so it does not explicitly keep the
correspondence between atom **types** and **long distances**.

Object to be tested:

```
P[type_pair, distance_bin]
  = count of unordered heavy-atom pairs with the specified element classes
    and shortest-path distance bin
```

Two questions:

1. **Expressivity.** Do two valid, different molecules exist whose outputs are
   indistinguishable for the current LCRC-A input, while the type-distance
   statistics separate them?
2. **Task relevance.** Inside the official MolHIV **Train** split, does the
   type-distance correspondence carry a stable, substantial predictive signal
   beyond a conventional local fingerprint (ECFP4) and the LCRC global
   descriptor (`g42`)?

Interpretation limits fixed in advance:

* A positive answer to (1) proves an expressivity limitation **only**.
* A positive answer to (2) proves the information deserves further testing
  **only**.
* Both together still do **not** prove that adding the information to LCRC
  improves predictive performance.

---

## 2. Frozen code that is executed (not re-implemented)

| role | file |
| --- | --- |
| `lipo_v1` FeatureSpec (hash `7270f767…a515ee`) | `lcrc_cross_task_preparation_v1/code/feature_spec.py` |
| production LCRC-A input assembly (`build_molecule`, `concat_molecules`, `collate`, `fit_norm`) | `lcrc_cross_task_preparation_v1/code/lcrc_data.py` |
| production SMILES → raw attributes → `lipo_v1` graph | `lcrc_molnet_benchmark_pilot_v2/code/mb2_rdkit.py` |
| production `DissectLCRC` (variant A0) + frozen column map to C1 | `lcrc_lipo_model_dissection_v1/code/dissect_model.py` |
| production `PTBLCRC` (single-task forward delegates to the frozen parent) | `lcrc_predictive_trace_benchmark_v1/code/ptb_model.py` |
| frozen OGB source root + registry | `lcrc_molnet_benchmark_pilot_v2/code/mb2_common.py` |
| frozen structure-cache loader / FeatureSpec loader | `lcrc_molnet_benchmark_pilot_v2/code/mb2_data.py` |
| reference pattern for the production-path consistency test | `lcrc_taskaware_representation_pilot_v1/code/ta_collision.py` |

No file above is modified. `analysis/provenance.json` records the commit, the
sha256 of every reused module, the sha256 of every code file of this round, and
the RDKit / PyTorch / scikit-learn / NumPy / SciPy versions actually used.

A0 and C1 are **not** re-implemented mathematically: they are the real frozen
model classes, instantiated with the frozen shared initialisation
(`A0` full state for seed `s`, C1 obtained by the frozen column deletion), in
`eval()` mode, CPU, float32, no training.

---

## 3. Phase A — constructive structural blind spot (CPU, label-free)

### A1. Fixed molecule pair

Two 18-heavy-atom single rings of `SINGLE` bonds, built atom-by-atom with
`Chem.RWMol` in the required index order, then `Chem.SanitizeMol`:

| molecule | atom 0 | atom 6 | atom 8 | other atoms |
| --- | --- | --- | --- | --- |
| **A** | O | **N** | C | C |
| **B** | O | C | **N** | C |

Validity gate: sanitisation succeeds and the formula is exactly `C16H33NO`.
Expected canonical SMILES: A = `C1CCCCCNCCCCCOCCCCC1`, B = `C1CCCCNCCCCCCCOCCCC1`.
Expected N–O shortest graph distance: A = 6, B = 8.

Checks (all must pass, all recorded in `analysis/structural_witness.json`):

1. both molecules sanitise;
2. identical formula;
3. **non-isomorphic** under element labels and bond types — decided by an
   exhaustive labelled backtracking isomorphism search implemented in this
   round (`nb_structural.labelled_isomorphic`), *not* by comparing SMILES
   strings; an independent invariant is recorded too: both graphs are one
   18-cycle, so every isomorphism is a dihedral symmetry of the cycle and
   preserves the cyclic N–O distance (6 ≠ 8);
4. identical multiset of per-atom radius-2 atom/bond-labelled environments
   (RDKit Morgan environment identifiers with `bitInfo` attribution, one
   signature per root atom);
5. identical 2048-bit Morgan radius-2 fingerprint;
6. RDKit N–O shortest graph distances equal 6 and 8 (`Chem.GetDistanceMatrix`);
7. the production `lipo_v1` feature path (SMILES → `mb2_rdkit.smiles_to_raw_row`
   → `raw_row_to_graph` → `lcrc_data.build_molecule`) yields **numerically
   identical 42-d `global` vectors**;
8. the real LCRC model is used to verify the multiset of local `h` rows, the
   global readout and the final output (Section A2).

No property label is read in Phase A. The molecules' true labels are never
looked up and are not part of any claim.

### A2. Production forward equivalence

* variants: `A0`, `C1`; random initialisation seeds `0, 1, 2` (six model
  instances, `PTBLCRC` with a single output, `eval()`, CPU, float32, no
  training);
* normalisation: the **frozen Lipophilicity Train-only structural
  normalisation** of the executed dissection/PTB rounds, recomputed from the
  frozen Lipophilicity structure cache (structures only) and gated on the
  recorded fingerprint `4e62dae7…1cc015`; if the recomputed fingerprint does
  not match, the round stops;
* the two molecules are put in **one** batch, and each molecule is also run
  alone;
* comparisons: batch `global`; local `h` rows as a multiset using a
  **minimum-cost bipartite matching** (Hungarian algorithm on the L1 cost; row
  indices are not comparable because the canonical SMILES re-indexes atoms);
  `contrib`; `rho`; final `y`; plus a random atom-reordering invariance test
  (molecule A renumbered by a seeded permutation, re-expressed as a
  non-canonical SMILES, pushed through the same production path);
* final-prediction equivalence tolerance: **absolute error ≤ 1e-5**.

**Stopping rule (pre-registered).** If the local-`h` multiset or the prediction
equivalence fails for any variant/seed, the round **stops** and reports the
exact failing fields. The molecule pair, the tolerance and the architectures are
not changed to rescue a conclusion.

### A3. Mechanism control

Using the round's own type-distance definition (Section 4.1):

* REAL differs between A and B;
* the N/O type pair appears in the distance-6 bin for A and the distance-8 bin
  for B;
* the independence NULL of Section 4.2 is **identical** for A and B.

Full molecules, elements, bond lists, distance matrices and every check result
are stored in `analysis/structural_witness.json`.

---

## 4. Phase B — MolHIV official-Train-only task relevance

### 4.0 Data range and label discipline

* `ogbg-molhiv`, official release v1 (2020-05-04), local frozen copy under
  `lcrc_molnet_benchmark_pilot_v2/analysis/source/ogb/ogbg_molhiv/`.
* **Only** `split/scaffold/train.csv.gz` is opened (32 901 rows). The official
  Valid and Test index files are never opened.
* SMILES come from `mapping/mol.csv.gz` read with
  `usecols=['smiles','mol_id']`, i.e. the `HIV_active` column is never decoded.
* Labels are streamed line by line from `raw/graph-label.csv.gz` and only the
  positions inside the official Train index set are retained, so no Valid/Test
  label is ever materialised in memory.
* `g42` comes from the frozen `lipo_v1` structure cache for exactly those Train
  rows (the cache is label-free by construction; its content fingerprint must
  equal the one the frozen benchmark recorded, `111d9175…258596`).
* The official Train row-id sha256 must equal
  `4e77289653e41be5d2267f54d9f33fda4b70fd622c19b93a282db710a9318b0a`
  (32 901 rows, 1 232 positives, 31 669 negatives).

All features are computed from 2D molecular graphs derived from the Train SMILES
only. No feature uses a property label.

### 4.1 Element classes and distance features (REAL)

Fixed six heavy-atom classes:

```
C, N, O, S, X, OTHER      with X = {F, Cl, Br, I}
```

Aromatic and non-aromatic forms of an element fall in the same class. Hydrogen
atoms are **not** graph nodes; an explicit `[H]` in the SMILES is removed before
pair counting.

Unordered type pairs are enumerated `i ≤ j`, `p = 6i − i(i−1)/2 + (j − i)`, giving
`6·7/2 = 21` pairs.

Only unordered pairs inside the same connected component with shortest-path
distance `≥ 5` are counted. Distance bins are fixed:

```
5, 6, 7, 8, 9, >=10        (k = min(d − 5, 5))
```

so the matrix is `R[p, k]`, `21 × 6 = 126` dimensions. Pairs in different
components are never counted.

Per molecule:

```
REAL[p, k] = R[p, k] / max(1, total_number_of_pairs_with_distance_>=5)
```

If no such pair exists, all 126 features are zero. No distance summary, atom
role, physicochemical descriptor or extra bin is added.

### 4.2 Same-dimension, margin-matched NULL

On the raw counts (before normalisation):

```
row_margin[p] = Σ_k R[p,k]
col_margin[k] = Σ_p R[p,k]
N             = Σ_p Σ_k R[p,k]
NULL[p,k]     = row_margin[p] · col_margin[k] / N      (all zero if N = 0)
```

`NULL` is then divided by the **same** `max(1, total)` as `REAL`. Consequences
that are unit-tested:

* row margins, column margins and the total pair count of `NULL` equal those of
  `REAL` exactly (up to floating-point tolerance `1e-8` in atom-pair units, with
  exact integer reconstruction of the margins);
* the only programmed difference between the two blocks is the *pairing* of
  element type pairs with long-distance bins.

### 4.3 Common baseline inputs

`BASE` is exactly two blocks:

1. **ECFP4**: Morgan, `radius=2`, `fpSize=2048`, bit vector, `includeChirality=False`.
2. **g42**: the frozen `lipo_v1` 42-d global feature vector (raw, then
   standardised per training partition as in Section 4.5).

Exactly three models are compared; no fourth model is added:

| model | inputs |
| --- | --- |
| `BASE` | ECFP4 + g42 |
| `BASE_NULL` | ECFP4 + g42 + NULL126 |
| `BASE_REAL` | ECFP4 + g42 + REAL126 |

### 4.4 Splits (pre-registered)

Grouping key: Bemis–Murcko scaffold; an empty or uncomputable scaffold falls
back to the canonical SMILES (and to the raw source SMILES if even that fails).

* outer: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261103)`
* inner (inside each outer Train):
  `StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=20261103 + outer_fold)`
* no scaffold may cross the corresponding Train/Validation boundary;
* per fold the sample count, positive/negative counts, scaffold count and index
  sha256 are recorded (`analysis/folds.json`);
* the three outer validation folds must partition the Train set exactly.

If any fold has a single label class the round records `SPLIT_INFEASIBLE` and
stops; the seed is not changed to look for an easier split.

### 4.5 Model and hyper-parameters

`sklearn.linear_model.LogisticRegression`, identical for all three models:

```
penalty="l2", solver="liblinear", class_weight=None,
fit_intercept=True, tol=1e-4, max_iter=4000, random_state=20261103
```

Candidate regularisation: `C ∈ {0.01, 0.1, 1.0}` only.

Per outer fold the single `C` is chosen **using BASE only**, by the mean ROC-AUC
of the two inner folds, ties resolved to the **smaller** `C`. That same `C` is
then applied to `BASE`, `BASE_NULL` and `BASE_REAL` of that outer fold. NULL and
REAL results never enter the choice. Convergence is checked
(`n_iter < max_iter`); a non-converged fit is an error, not a silent result.

Preprocessing (fitted on the training partition only; inner partitions for the
inner folds, the outer training partition for the outer folds):

* ECFP4: kept at 0/1, never standardised;
* g42: per-column mean / population std of that partition;
* REAL and NULL: **one shared** mean/std fitted on the vertical concatenation of
  the partition's REAL and NULL rows, then applied identically to both;
* a zero-variance column gets scaling denominator `1.0`;
* standardised continuous features are clipped to `[-5, 5]`.

### 4.6 Metrics

Primary: arithmetic mean over the three outer folds of the held-out ROC-AUC.

Secondary: mean Average Precision; per-fold `REAL − NULL` and `REAL − BASE`
ROC-AUC differences; per-fold ROC-AUC / AP of every branch; per-fold positive
counts and prediction distributions; non-zero rate and per-type-pair /
per-distance-bin coverage of the type-distance features.

Paired scaffold bootstrap on the **fixed** outer predictions:

* inside each outer validation fold, scaffolds are resampled with replacement;
* the same resampled rows score all three models;
* per replicate the three-fold mean `ROC-AUC(REAL) − ROC-AUC(NULL)` and
  `ROC-AUC(REAL) − ROC-AUC(BASE)` are computed;
* a replicate with a single label class is discarded and redrawn, with at most
  2 000 attempts per replicate; fewer than 1 000 valid replicates ⇒
  `BOOTSTRAP_INCOMPLETE`;
* seed `20261103`, 95 % percentile interval.

Every interval reported by this round is **sampling uncertainty at fixed fitted
models and a fixed split**; it excludes re-splitting and re-training variance.

### 4.7 Pre-registered decision (never relaxed afterwards)

`PROMOTE_INFORMATION_SIGNAL` only if **all** six conditions hold:

1. mean `REAL − NULL` ROC-AUC over three folds `≥ +0.015`;
2. all three folds have `REAL − NULL` ROC-AUC strictly `> 0`;
3. mean `REAL − BASE` ROC-AUC over three folds `≥ +0.010`;
4. mean `REAL − NULL` AP difference `≥ 0`;
5. paired-bootstrap 95 % interval lower bound of `REAL − NULL` `> 0`;
6. all data and feature correctness gates pass.

Otherwise:

* `WEAK_OR_INCONSISTENT_SIGNAL` — some positive evidence, but not all conditions;
* `NO_OBSERVED_INCREMENT` — mean `REAL − NULL` ROC-AUC `≤ 0`;
* `INFEASIBLE` — data, numeric or gate failure makes the result unusable.

All results are **exploratory Train-only evidence**, not independent confirmation.

---

## 5. Implementation details fixed before the run

These are engineering decisions, not scientific choices; each is recorded in the
artifacts and covered by a test.

1. **RDKit parse ladder.** `Chem.MolFromSmiles` (full sanitisation) first; on
   failure, parse without sanitisation and apply
   `SANITIZE_ALL ^ SANITIZE_PROPERTIES` plus `FastFindRings`. This keeps the
   published sample set intact: exactly three official Train molecules are
   invalid-valence metal complexes (Al valence 7, Ge valence 5) that fail
   valence perception but are perfectly adequate for a connectivity-based
   fingerprint. No row is dropped. `analysis/feature_margin_checks.json`
   reports the per-row parse mode.
2. **Pair statistics use a connectivity-only parse** (`sanitize=False`): bond
   orders, charges and aromaticity cannot change a shortest-path distance, so the
   distance features depend only on the 2D connectivity graph. Hydrogen nodes are
   removed. A gate proves
   `n_heavy_atoms + n_hydrogen_nodes_removed == frozen lipo_v1 atom count` for
   every Train molecule, i.e. the same molecular graph is being analysed.
3. **Scaffold ladder.** `MurckoScaffoldSmiles` on the robustly parsed molecule;
   empty/raising → canonical SMILES; canonical SMILES unavailable → raw source
   SMILES.
4. **Floating point.** REAL/NULL are stored as float64; margin conservation is
   asserted at `1e-8` atom-pair units (float64 rounding, not a definitional
   gap). The model forward stays float32 as in production.
5. **Determinism.** Every random operation is seeded: folds (`20261103`,
   `+outer`, `+inner`), logistic regression (`20261103`), bootstrap
   (`20261103`), Phase A renumbering (`0`), model initialisation (`seed ∈ {0,1,2}`).

---

## 6. Stopping rules / what this round will not do

* Phase A failure stops the round before Phase B. No molecule, tolerance or
  architecture substitution.
* No GPU, no gradient training, no new LCRC/GNN/Transformer variant.
* Even with `PROMOTE_INFORMATION_SIGNAL` the round **stops**: no atom-pair
  attention, no distance embedding, no global two-body module, no A0/C1
  retraining, no extra atom-role or bin search, no official Valid/Test scoring.
* No seed shopping, no extra dataset, no post-hoc threshold change.
* The official Valid/Test splits are never used, for any purpose.
