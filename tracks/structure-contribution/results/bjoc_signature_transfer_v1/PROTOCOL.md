# cscl-bjoc-signature-transfer-v1 — Preregistered Protocol

**Status:** PREREGISTERED before any validation outcome computation.
**Track:** `tracks/structure-contribution` · **Namespace:** `bjoc_signature_transfer_v1`
**Compute:** CPU only, frozen BJOC-R3 full-10k soups (`soup_R3_seed0.pt`, `soup_R3_seed1.pt`).
**Scope:** one proposition only — *can Junction structure-signature importance
discovered on one batch of molecules transfer to molecules unseen during model
weight training and predict which local structures the frozen model encodes
more sensitively?*

Explicitly **not** under test: chemical causal laws, complete property
explanation, new models, new GNN training, R3 optimisation, Motif partition
search, signature-vocabulary expansion. No prediction labels are read; the
official test is never loaded; the internal Locked split is never read;
historical reports, models and results are read-only.

---

## 1. Discovery phase (frozen before validation)

**Data.** The original Phase-0 attribution results
(`results/bjoc_source_ownership_v1/phase0_results.json`, `records_c`): 256
`sample_manifest.json` diagnostic train molecules × 2 frozen R3 seeds, with
per-Junction `|IG|`, encoder deviation `dev`, signature, degree, ring state.
No values are recomputed for the table itself; the reuse is guarded by an exact
replay check (Section 5).

**Signature.** The unchanged Phase-0 tuple
`(center_type, sorted(endpoint_types), sorted(bond_types), center_in_ring, n_ring_bonds)`
(225 classes on the discovery set). No new features, no vocabulary change.

**Typical importance.** Within-molecule `|IG|` percentile
`p = (average_rank − 1)/(n_j − 1)` (ties averaged; `n_j = 1 → 0.5`).
Per `(seed, molecule, class)`: mean percentile of that class's Junctions —
**one value per molecule–seed–class**.
Table value `T_c = mean over molecules of (mean over seeds of that molecule's
class value)`. Every molecule carries equal weight; a molecule with many
repeated Junctions of one class cannot dominate the class statistic.

**Frequency floor.** A class is a candidate for validation if it appears in
≥ 20 of the 256 discovery molecules. All classes above the floor are kept; no
top-N cut. Unknown or below-floor classes receive the frozen neutral score
`T = 0.5` (secondary sensitivity: minimum candidate `T`).

**Rarity baseline.** Counts over the discovery 256 molecules only;
`-log2((count + 0.5)/(N + 0.5 (V + 1)))`, `V` = distinct discovery classes,
unknown count = 0.

**Internal positive control.** Diagnostic rows sorted ascending; even index →
fold A, odd → fold B. Fold tables use the same molecule-cap rule with floor 10
molecules. Fold-B molecules are ranked with the fold-A table and vice versa;
the pooled top-4 response contrast vs RANDOM is reported as an
in-distribution pipeline control (not part of the decision).

## 2. Validation phase

**Data.** `objects_valid.pt` — the 1000 official-valid structures. Official
valid is a historically exposed development set (model/recipe selection in
earlier rounds), **not** a fresh blind test; this is stated in every output.
No prediction labels are accessed anywhere in this round.

**Selection is label-free and IG-free.** Ranking a validation molecule's
Junctions uses only the frozen discovery table `T_c` (or a pre-fixed baseline
rule). A molecule's true `|IG|` may never select its target Junctions; it is
computed only as the oracle reference upper bound.

**Intervention.** Identical to Phase 0: nearest frozen reference molecule
(1024 `reference_rows`, L1 on `(n_atoms, n_bonds, n_junctions)`, tie-break
smaller row), natural encoding replacement
`w_j ← w_j + α(w_ref[j mod n_ref] − w_j)`, `α = 1.0`. Response
`Δ_j = predict(replaced) − predict(natural)`. Responses are independent per
Junction, so `Δ_j` is computed once per Junction and reused by every ranking
(exactly the same arithmetic as the Phase-0 `_subset_response` loop).

**Ranking methods.** `SIG` (table `T_c`, unknown 0.5; secondary `SIG_MIN`),
`RING` (center in ring), `DEGREE`, `RING_DEGREE`, `DEV` (encoder deviation),
`RARE`, `RANDOM` (16 uniform draws), `SHUFFLE` (16 permutations of the
candidate `T` values, same permutation across molecules per seed), `ORACLE`
(true `|IG|`, reference only). Ties break by Junction index.

**Metrics.** Primary `k = 4` (sensitivity `k ∈ {1, 2, 8}`):
`A(S) = mean_{j∈S} |Δ_j|`. Statistical unit is the **molecule**; merged
inference uses the per-molecule mean of the two seeds (cross-seed correlation
preserved), bootstrap 5000 (seed 0) and exact one-sided sign test; per-seed
values are reported separately. Also reported: oracle fraction
`(SIG − RANDOM)/(ORACLE − RANDOM)`, per-molecule Spearman ranking quality
against model-internal `|IG|`, class-level Spearman/Pearson between `T_c` and
the validation class percentile (molecule bootstrap), and context stability
for candidate classes across center degree, endpoint ring count, molecule
size tercile and molecule ring-bond tercile. Signed `Δ` is reported as
secondary context only and is never described as positive/negative chemical
contribution.

**Cross-scaffold.** Canonical dataset-graph SMILES and Murcko scaffolds for
train 10k, discovery 256 and valid 1000. Exact-duplicate checks (valid↔train,
valid↔discovery) are recorded; the primary contrast is recomputed on
`scaffold_unseen_discovery` (valid molecules whose Murcko scaffold is absent
from the 256 discovery molecules) and `no_exact_dup_train`. Cache-level
identity limitation documented (the committed ZINC atom dictionary cannot
express uncharged N/O/S H counts).

## 3. Decision rule (exact)

Evaluated on the merged row-unit `k = 4` contrasts; `SHUF` = mean over the 16
permutations.

| Gate | Requirement |
| --- | --- |
| M0 coverage | SIG top-4 known-class fraction ≥ 0.5 |
| G1 transfer | `SIG − RANDOM`: mean > 0, CI95 lower > 0, sign p < 0.01 |
| G2 ring rule | `SIG − RING`: CI95 lower > 0 |
| G3 baselines | `SIG − DEV` CI95 lower > 0; `SIG − RARE`, `SIG − RING_DEGREE`, `SIG − DEGREE` means > 0 |
| G4 cross-scaffold | `scaffold_unseen_discovery` subset: `SIG − RANDOM` mean > 0, CI95 lower > 0 |
| G5 both seeds | per-seed `SIG − RANDOM` mean > 0 |
| G6 specificity | `SIG − SHUF`: mean > 0, CI95 lower > 0 |
| G7 oracle fraction | ≥ 0.25 |
| G8 class level | Spearman(`T_c`, validation class percentile) > 0, bootstrap CI95 lower > 0 (candidate classes present in ≥ 5 valid molecules) |

**Verdicts**

* `TRANSFERABLE_SOURCE_SIGNATURE_SIGNAL` — M0 and G1–G8 all pass.
* `CONTEXT_DEPENDENT_OR_PARTIAL_TRANSFER` — G1 passes but at least one other
  gate fails: transfer exists only partially or in some contexts/scaffolds.
* `NO_SUPPORTED_SIGNATURE_TRANSFER` — G1 or M0 fails: no confirmable transfer
  increment; the current Junction signature should not continue as the primary
  interpretation vocabulary.

## 4. Chemical-semantics check

Class identities are fixed by rule from the discovery table before validation:
top-3 candidate classes by `T` with support ≥ 50, bottom-3 by `T` with
support ≥ 50, and the single most frequent candidate class. For each, up to
three distinct valid molecules (ascending row) show the center/endpoint atom
words, bond-order words, RDKit ring membership of the center and both bonds,
atom degrees, and one counterexample instance (minimum within-molecule `|IG|`
percentile, different molecule when possible). RDKit is used to verify ring
and bond semantics. Two-bond signatures are never named chemical groups; the
check separates (a) signature reuse, (b) context effects, and explicitly
leaves (c) higher-order connected-structure discovery out of scope.

## 5. Discipline, replay guards and amendments

* `freeze` records git commit, config/code hashes and input hashes; every
  computational stage re-verifies the freeze.
* Diagnostic replay guard: recomputing `IG`, `dev`, `rare` and signatures for
  the 256 diagnostic molecules with the frozen soups must reproduce the
  Phase-0 `records_c` values (tolerance 1e-9); the nearest-reference choice
  must reproduce the Phase-0 pairings.
* All outputs written under `results/bjoc_signature_transfer_v1/`; historical
  artifacts are never overwritten. Any forced deviation (e.g. an RDKit
  fallback) is recorded as an `AMENDMENT_*.md` **before** the affected
  analysis runs.
* No GPU purchase, no new training, no search; only the frozen R3 forward
  pass, IG and interventions on CPU.

## 6. Deliverable

One-page research judgment in `REPORT.md` plus `final_decision.json`: whether
the current structure signatures are worth continuing as cross-molecule shared
interpretation units, with sample-support scope, confidence intervals,
cross-seed and cross-scaffold results, and the limitation that every claim is
predictor-internal encoded source attribution, not chemical causality or
property explanation.
