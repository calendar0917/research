# PROTOCOL — `cscl-bjoc-source-ownership-v1`

* namespace: `bjoc_source_ownership_v1` · track `tracks/structure-contribution`
* status: **frozen before the first Phase-0 execution and before any GPU trajectory**
* config (machine-readable freeze): `configs/bjoc_source_ownership_v1.yaml`
* frozen evidence base: `bjoc_structure_explanation_audit_v1` (+ `ERRATUM.md`),
  `bjoc_source_attribution_feasibility_v1`, `bjoc_architecture_breakthrough_v1`,
  `bjoc_cross_bond_interaction_v1`

This round answers, with one preregistered protocol, three questions:

1. **Q1 (Phase 0).** Does the frozen-model source-level Junction ranking (|IG|) carry
   structural signal *beyond* simple encoder-anomaly / structural-rarity baselines,
   under the same independent natural-encoding intervention?
2. **Q2 (Phase 1/2).** With real Junction→Bond ownership available to the local
   pathway (arm `O`) vs an exactly parameter-, init-, data-, SSL- and budget-matched
   non-ownership aggregate (arm `N`), does ownership have independent representation
   value?  A matched ownership-shuffle arm (`S`) separates "ownership" from "any
   count-matched per-bond variation".
3. **Q3 (Phase 2).** Can the ownership-resolved local pathway support a *more
   trustworthy local-structure explanation* at acceptable BJOC-R3 performance parity?

---

## 0. Prohibited this round

* no new random-walk search, no walk-length / community / path search;
* no new Junction feature family search;
* no new IG baseline / reference search;
* no model width / depth / hidden-size search;
* no training-hyperparameter search (recipe R3 and the reused SSL are frozen);
* official test is never loaded; the Locked split is never read;
* no label (`.y`) is read in Phase 0 or in SSL; Monitor labels are read exactly
  once for the seed-0 screen and once (cached) for the conditional seed-1 confirmation.

---

## 1. Phase 0 — frozen-model source specificity (CPU, label-free)

Frozen model: full-10k `soup_R3_seed{0,1}.pt` (413,505 params, `eval()`, no
gradient updates).  Structures: `objects_train10k.pt` (no label field).  Sample:
the *same* hash-rule diagnostic set (rank 0–255) and reference set (rank
256–1279) as `bjoc_source_attribution_feasibility_v1/sample_manifest.json`.

### 1.1 0A — interpretation facts (documented, numerically verified)

**Fact 1 (single-junction IG is a one-dimensional path integral).**  Under the Null
grouping the prediction depends on `w` only through
`mu_G = mean_j w_j` and `nu_G = mean_j w_j^2`.  The integrated gradient of one
junction is therefore the exact telescoping difference

```
IG_j = pred(..., w_j, ...) - pred(..., mu_G, ...)   (up to float32 path error)
```

and equal to `-occ_j` with `occ_j = pred(w_j <- mu_G) - pred(natural)`.  This is
*one-junction baseline replacement*, not a joint-reset share.

**Fact 2 (joint reset).**  Setting *all* `w_j <- mu_G` leaves the graph first
moment exactly unchanged (`mu_G` stays `mu_G`) and collapses the second moment to
`mu_G^2`.  The joint-reset response therefore measures mainly the removal of the
*encoded variance* term, and must not be called a "mean-level contribution".
Numerically: after the joint reset, recomputed `mean_G(w) == mu_G` and
`mean_G(w^2) == mu_G^2` (float32 tolerance).

Both facts are verified on the frozen models and recorded in
`phase0_a.json`; they are *references to existing reports*, which are not modified.

### 1.2 0B — simple-baseline exclusion

Pre-fixed rankings of junctions inside one molecule (no tuning):

| name | score |
|---|---|
| `IG` | `abs(IG_j)`, baseline `mu_G`, K = 32 midpoint steps |
| `dev` | `norm(w_j - mean_G(w))_2` (encoder-space anomaly) |
| `rare` | `-log2(p_hat(signature))`, `p_hat` = signature frequency over the official train 10k junctions |
| `random` | uniform `k`-subset, 16 draws (same RNG convention as the source-attribution audit) |

* `k = 4`; the intervention is the **matched natural-encoding replacement**
  (`w_j <- w_j + alpha (w_ref[j mod n_ref] - w_j)`, `alpha = 1.0`) with the
  position-matched nearest reference molecule (same rule as the source-attribution
  audit).  Ranking and intervention stay independent.
* **Matched control.**  Each `IG`-top-4 junction is compared with up to 8
  same-molecule control junctions matched on
  `(signature, center degree, center-in-ring flag, in-molecule dev-decile)`
  (fallback: drop the dev-decile constraint).  Matching coverage, the matched
  junction count and the dev-decile-exact fraction are reported as diagnostics
  (amended before execution: no fixed coverage threshold, see `AMENDMENT_01.md`).
* **Frozen gates**
  * `P0_BEYOND_DEV`: paired mean `Δ_IG - Δ_dev > 0`, bootstrap CI95 lower > 0,
    one-sided sign-test `p < 0.01`;
  * `P0_BEYOND_RARE`: same against `rare`;
  * `P0_MATCHED`: paired mean `Δ_j - Δ_matched > 0` and CI95 lower > 0 over the
    matched junctions (amendment 01; coverage and dev-decile exactness reported).
* Structural composition of the top-4 sets (signature class, center degree, ring
  state, mean dev) is reported; if the sets differ materially the matched-control
  result governs.

### 1.3 0C — deterministic cross-molecule Junction signature

Signature (real structure only, label-free):

```
sig(j) = (center_type,
          sorted(endpoint_types),
          sorted(two bond types),
          center_in_ring,
          n_ring_bonds)
```

where types are the raw 28-slot atom-type index (`argmax(atom_feat[:28])`),
bond types the raw category index, ring state from the fixed feature blocks.
Corpus frequency is computed over all official-train junctions.

Reported:
* frequency table (corpus and diagnostic sample);
* top-decile `|IG|` signature enrichment with a **within-(seed, molecule)
  `|IG|` permutation null** (1000 draws, RNG 20261015) → `P0C_ENRICHMENT`
  `p < 0.05`.  Statistic: max over classes with corpus count >= 100 and
  expected top-decile count >= 3 of observed/expected top-decile count; the top
  decile is taken within each (seed, molecule) block.
* cross-molecule sign consistency for classes with sample count >= 30, with the
  same permutation null → `P0C_SIGN` `p < 0.05`.  Statistic: weighted mean of
  `|mean(sign(centered IG_j))|` where `centered IG_j = IG_j - mean_molecule(IG)`;
  centering is required because the molecule-level mass term gives all of a
  molecule's junctions the same sign.  Classes with corpus frequency < 30 are
  reported descriptively only.
* classes with insufficient frequency and their descriptive statistics;
* context differences (center degree / ring state) within signature.
* No clustering figure is used as evidence.

### 1.4 Phase-0 verdict (frozen)

```
BEYOND_SIMPLE_BASELINES  iff  P0_BEYOND_DEV and P0_BEYOND_RARE and P0_MATCHED
                              and P0C_ENRICHMENT and P0C_SIGN
EXPLAINABLE_BY_SIMPLE_BASELINES  iff  (not P0_MATCHED) and
                              (mean Δ_dev >= mean Δ_IG or mean Δ_rare >= mean Δ_IG)
INCONCLUSIVE  otherwise
```

---

## 2. Phase 1 — N / O / S ownership model (design + CPU proof)

All arms share one class and one state-dict layout; the grouping function is a
functional switch, so parameter count, init and modules are identical:

```
a_i = AtomMLP(atom_feat_i)
b_e = BondMLP(a_u, a_v, bond_feat_e)
w_j = JuncMLP(a_i, a_c, a_k, b_e1, b_e2, junc_feat_j)

local composer input:  [ b_e | R1_e | R2_e | log1p(n_e) | bond_feat_e ]  -> c_e
  N: R1_e = n_e * mu_G              R2_e = n_e * nu_G
  O: R1_e = sum_{j ∋ e} w_j         R2_e = sum_{j ∋ e} w_j^2
  S: R1_e = sum of n_e deterministic draws from the molecule's own junction
             pool (canonically ordered), same format as O; true incidence destroyed

global content pathway (identical in N/O/S, never broadcast to bonds):
  p = MLP( [mu_G | nu_G | log1p(n_junc)] )                     193 -> 96 -> 96

reader input: [ sum_e c_e | sum_i a_i | sum_i a_i^2 | g | p ]    321 + 96 = 417
```

Init parity: build the historical scratch BJOC-Null init, copy all base modules,
widen the reader first layer to 417 with **zero columns** for `p`, initialise the
global MLP from the same seeded construction RNG.  All arms share the seed's
scratch init, the reused SSL encoder, the batch order and the recipe.  The N arm
reproduces the historical P3 function to `<= 1e-6` (float32 reduction order);
O/S differ at init only through the grouping and are instead certified by the
zero-`p_override` inertness proof (the global block cannot influence any output
in any arm).

CPU proofs (must all pass before any GPU job):

1. **permutation invariance** — atom relabeling, bond-order permutation and
   junction-row permutation leave N/O (and S, by construction from a
   label-invariant canonical junction key) predictions unchanged to float32 noise;
2. **ownership intervention isolation** — in O, editing one junction's incidence
   changes `c_e` only for the two old and the two new bonds (exact), never for
   other bonds; in N the same edit changes nothing (no ownership);
3. **global pathway separation** — randomising `p` leaves every `c_e` unchanged;
   the local pathway cannot read the global block (code audit + numeric witness);
4. **arm parity** — exact parameter count and identical init hashes for N/O/S;
5. **train-loop determinism** — two runs with the same seed/config produce identical
   soup hashes; the supervised loop is a line-for-line R3 copy;
6. **save/load replay**, single-molecule vs batch prediction parity, no
   GNN/attention/iteration modules;
7. **sham invariants** — S preserves `n_e` per bond and the per-bond vector
   format exactly, and (on the audited test set) has no label-invariant-key
   collisions among distinct junctions/bonds.

## 3. Phase 2 — frozen training protocol and decisions

* supervised recipe R3 (360 epochs, cosine, raw-y L1, batch 128, clip 5,
  last-5 soup); masked-SSL encoder reused read-only; no hyperparameter search.
* arms `N`, `O`, `S` for seed 0; seed 1 for `[N, O, S]` only if every seed-0 gate
  passes.  GPU cap 1.5 h.
* Monitor is read once after all seed-0 predictions are frozen and hashed; a second
  (seed-1) scoring round reuses the cached label vector and creates no new label read.
* metrics: paired MAE / RMSE, ordinary (`y > -3`) / tail,
  top-10-removed (removing the rows with the largest `max(err_candidate,
  err_reference)`), molecule bootstrap CI95 (5000 resamples, seed 0); contrasts
  `O-N` (primary), `O-S`, `N-S`, and each arm vs the frozen `P3-D` reference on
  the same rows.
* secondary descriptive comparison: official valid (exposed dev set) of N/O/S and
  `P3-D` vs the frozen full-10k BJOC-R3.  Frozen reference: recipe-tuning R3
  seed-0 valid MAE `0.079101`, seed-1 `0.080586`.  The evaluation path is
  replayed as a guard on the valid set (must reproduce both numbers to 1e-5)
  before the comparison is read; never used for selection.
* mechanism audit on the trained soups (CPU): local-pathway deletion sensitivity,
  per-bond attribution `||c_e||_2` vs single-bond deletion response (faithfulness
  rank correlation, gate mean Spearman >= 0.5), ownership intervention isolation on
  real molecules, cross-seed attribution stability (if seed 1 runs).
* decision vocabulary (frozen): `OWNERSHIP_RELATION_VALUE_CONFIRMED`,
  `OWNERSHIP_LOCAL_VALUE_SEED0_ONLY`, `OWNERSHIP_NOT_MORE_VALUABLE_THAN_SHAM`,
  `OWNERSHIP_NULL`, `OWNERSHIP_HURTS_PARITY`, `INCONCLUSIVE`.

## 4. Writing and interpretation rules

* a pathway being *used* is not an explanation claim;
* only bond-anchored `c_e` may be called local; `p` is a graph-level pathway and
  must never be presented as a sum of local contributions;
* Phase 0 is a frozen-model measurement, not chemical causality;
* Monitor / official valid are exposed data; no blind-test language;
* no historical report or JSON is overwritten.
