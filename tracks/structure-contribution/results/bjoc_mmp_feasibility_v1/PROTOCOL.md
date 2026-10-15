# PROTOCOL — bjoc-mmp-feasibility-v1 (frozen before any label read)

Protocol id: `bjoc-mmp-feasibility-v1`
Track: `tracks/structure-contribution/`
Frozen date: 2026-10-15
Stage at freeze: Phase 0A complete; `labels_read = false`.

## 1. Research question

For a single legitimate local replacement `R → R'` of a molecule `C+R` in a shared
context `C`, is the property delta `Δy = y(C+R') − y(C+R)` reusable across
different molecular scaffolds, or is there a reliable, learnable context
dependence? Phase 0 only audits **feasibility**: whether the data support the
question at all. No model training, no BJOC modification, no Phase 1.

## 2. Data boundary (hard constraint)

* Read: PyG `ZINC(subset=True, split="train")` — 10,000 molecules — and the
  strictly aligned SMILES table `tracks/ksvd/results/zinc_cssd_basis_reuse_v1/
  train_canonical_smiles.npz` (positional row `i` = official-train
  `subset_index` `i`), plus the official-train label vector `y` inside
  `data/ZINC/subset/processed/train.pt`.
* Never read: official valid / test, internal Monitor / Locked labels,
  ZINC-Full, external labels, existing BJOC predictions / attributions /
  hidden representations.
* Phase 0A reads no labels at all; the module's Phase 0A code path never calls
  `load_train_labels()` (enforced by unit test).
* Verify-data verdict: `PASS_WITH_FROZEN_EXCLUSIONS`; 9,996/10,000 rows
  constitution-identical to the PyG graphs, 4 frozen excluded rows
  (941, 2210, 6071, 9766) whose committed canonical SMILES are constitutional
  isomers of the PyG graph (upstream topology-only pynauty matching cannot
  distinguish constitutional isomers). No RDKit parse failures.

Data hashes (frozen in `phase0a_manifest.json`):

| artifact | sha256 |
| --- | --- |
| smiles table `train_canonical_smiles.npz` | `04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff` |
| PyG `subset/processed/train.pt` | `e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b` |
| module `code/bjoc_mmp_feasibility_v1.py` | `c390ec728af42f974cd893e0acd482623269a42771ac2a57cf00abcb2c0e6000` |
| pair table P0 | `51ee40830c03d0dbda8459ed5366f90079c9445fea7efe6364a9361c19043802` |
| pair table P0s | `d6ec81d4f75814e61637d60c662152ada6c29c2b4839ac373705e2406717638a` |
| pair table P1 | `cbf0a43e3ff8251bb38528ad3ef7b354fbdd364ce3afc5d651de17e00e51f463` |

## 3. Frozen structure extraction (single-position only)

All of the following was implemented and frozen **before** reading `y`:

1. **Cut rule**: single acyclic `SINGLE` bond only. Ring bonds and
   double/triple bonds are never cut. Fragment via RDKit `FragmentOnBonds`
   with one dummy per side (`dummyLabels=[(1, 1)]`); both fragments must be
   connected and carry exactly one dummy; fragments with 0 heavy atoms
   rejected.
2. **Constitution level**: fragment canonicalisation and pair keys use
   `isomericSmiles=False`. A pair whose two molecules share a constitution is
   skipped: a stereo-only difference is not a structural edit. (Audit: 0 such
   pairs exist inside a shared core, because identical constitutions produce
   identical fragment sets; the 4 duplicate rows only inflate symmetric
   cut-plane multiplicities.)
3. **Core / variable fragment**: each cut yields incidences `(core, chain)` in
   both roles. The shared fragment is the core; the two non-shared fragments
   form the replacement `R → R'`.
4. **Direction**: within one core, chains are sorted by canonical fragment
   key; `R` = lexicographically smaller key, `R'` = larger. `row_r` holds `R`,
   `row_rp` holds `R'`. Therefore `Δy(A→B) = −Δy(B→A)` exactly, independent of
   row order (unit-tested).
5. **Ambiguity / duplicates**: multiple cut planes producing the same
   replacement are deduplicated; `pre_dedup_variants` records the number of
   symmetric cut realizations, and coverage reports pre-dedup totals
   (`n_cut_realizations_uncapped`) vs deduplicated pairs (`n_pairs`).
6. **Regimes (frozen)**:
   * `P0` (primary): core ≥ 5 heavy atoms, variable fragment 1–10 heavy atoms.
   * `P0s` (sensitivity): core ≥ 8, fragment ≤ 6.
   * `P1` (broad, capped): core ≥ 3, no fragment cap, per-core pair cap 200
     using deterministic sha256(`SUBSAMPLE_SALT`, core, rows, fragments)
     ascending order. Un-capped totals are still exact
     (`n_uncapped_pair_incidences`).
7. **Paired columns written** (parquet, `phase0a_pairs_<regime>.parquet`):
   `row_r, row_rp, core, r, rp, core_heavy, r_heavy, rp_heavy, env,
   scaffold_r, scaffold_rp, pre_dedup_variants`.
8. **Murcko scaffold**: RDKit `MurckoScaffoldSmiles` (default, non-generic) per
   row. **Attachment environment** `env`: element / aromaticity / formal
   charge / degree / total H of the core-side attachment atom plus sorted
   `(bond type, element, aromaticity, charge)` neighbour list.
9. **Coverage statistics** (all label-free): pairs, unordered molecule pairs,
   distinct molecules / cores / scaffolds, strict rule `(R,R')` counts,
   env-augmented rule counts, exact support distribution with support ≥ 2 / 3 / 5,
   cross-scaffold pair counts, coarse class (env + element-count delta)
   support, molecule participation, context/fragment size histograms.
   Exact rule support counts **distinct cores**; scaffold reuse is counted as
   distinct scaffold-pair groups per rule (a single occurrence does not count
   as reuse).

## 4. A-priori coverage thresholds (frozen, primary regime P0)

| check | threshold |
| --- | --- |
| distinct molecules | ≥ 1,000 |
| MMP pairs | ≥ 3,000 |
| distinct Murcko scaffolds | ≥ 300 |
| exact rules with support ≥ 2 contexts | ≥ 50 |
| exact rules with support ≥ 2 scaffold-pair groups | ≥ 20 |

Verdict: `MMP_COVERAGE_SUFFICIENT` iff all five pass; otherwise
`MMP_COVERAGE_INSUFFICIENT`. (Observed in Phase 0A:
`MMP_COVERAGE_INSUFFICIENT` — the first three pass, both repetition checks
fail. The thresholds are reported as frozen; the verdict is fixed by the
module, not by a judgement call after seeing labels.)

## 5. Label analysis plan (Phase 0B, frozen before the label read)

* `Δy = y[row_rp] − y[row_r]` on the raw ZINC-12k scale; no per-rule
  normalisation, no winsorising, no pair filtering by |Δy| or sign.
* Per-regime `Δy` summary (mean / median / std / quantiles / sign fractions)
  and per-rule cross-scaffold statistics; the statistics units reported are
  raw pairs, unordered molecule pairs, distinct molecules, distinct scaffolds,
  and distinct rules. Unordered-molecule pairs are the independent unit;
  scaffold-pair groups are used for resampling / cross-validation. **No
  pair-level bootstrap or naive pair-level significance test is performed.**
* Reference baselines only, evaluated with 5-fold leave-Murcko-scaffold-pair-out
  grouped CV (seed 20261015, deterministic):
  * `zero` (constant 0),
  * `global_train_mean`,
  * `rule_mean_fallback_zero` (**primary unknown-rule fallback = 0**, frozen),
  * `rule_mean_fallback_global` (sensitivity),
  * `composition_ols` — local composition descriptors only:
    per-element count deltas, ring / aromatic-ring / charge / HBD / HBA /
    rotatable-bond deltas and heavy-atom delta. No fingerprints, no learned
    representations, no NN training, no hyperparameter search.
* **Target formula identifiability audit**: ZINC-12k `y` is defined as
  `(logP − μ_logP)/σ_logP + (−SA − μ_SA)/σ_SA + (−cycle − μ_cycle)/σ_cycle`
  with `cycle = max(0, longest_cycle − 6)`. Read-only recomputation of logP
  (RDKit Crippen), SA (the GVAE `sascorer_gvae.py` shipped with the long-cycle
  audit, reading its own `fpscores.pkl.gz`) and the cycle penalty (NetworkX
  `cycle_basis`); two constant sets are reported (community and dataset
  constants recorded in the long-cycle audit). Per-molecule reconstruction
  quality bounds the audit noise floor (`2 × median |reconstruction residual|`,
  at least 0.01). The formula audit is a **near-oracle diagnostic**, never
  presented as a learned structure result; `composition_ols` is the fair
  structural baseline.
* The Phase 0B runner re-verifies the Phase 0A freeze (module, pair tables and
  SMILES-table hashes) and aborts if anything changed.

## 6. Pre-registered decisions

* **A. Coverage**: `MMP_COVERAGE_SUFFICIENT` / `MMP_COVERAGE_INSUFFICIENT`
  (thresholds in §4, frozen pre-label).
* **B. Mechanism**, on P0 / community constants:
  * `TARGET_FORMULA_DOMINATED` if `R²(d_formula, Δy) ≥ 0.90` **and**
    median |residual| ≤ noise floor;
  * `CONTEXT_DEPENDENCE_PLAUSIBLE` if `R² < 0.50` and
    `R²(composition_ols) < 0.50`;
  * otherwise `TARGET_MECHANISM_UNRESOLVED`.
  If dominated, the conclusion is not softened.
* **C. Next step**:
  * `GO_TO_BASELINE_VALIDATION` only if coverage sufficient **and** data
    integrity OK **and** both repetition checks pass **and** mechanism
    `CONTEXT_DEPENDENCE_PLAUSIBLE`;
  * `NO_GO_ON_ZINC` if `TARGET_FORMULA_DOMINATED`;
  * otherwise `INCONCLUSIVE`.

## 7. Anti-fishing commitments

* Extraction rules, direction convention, regimes, caps and thresholds were
  frozen before the label read; the Phase 0A manifest records
  `labels_read = false` and all hashes.
* No MMP-rule tweaking, no pair dropping/selection based on `y`, no repeated
  significance hunting; one analysis pass after the freeze.
* No training, no GPU, no SSL search, no molecule generation, no Phase 1;
  existing reports and `research_core.md` are not overwritten.

## 8. Reproduction

```bash
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --verify-data
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --phase0a
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --phase0b
uv run python tracks/structure-contribution/code/bjoc_mmp_feasibility_v1.py --summary
uv run pytest -q tracks/structure-contribution/tests/test_bjoc_mmp_feasibility_v1.py
```
