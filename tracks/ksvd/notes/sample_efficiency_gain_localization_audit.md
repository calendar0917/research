# Sample-Efficiency Gain Localization Audit

**Track:** `tracks/ksvd` — ZINC `compact-v4-smallhead`
**Protocol:** `sample_efficiency_gain_localization_v1`
**Mode:** zero-new-training, target-free-support → gain association
**One-line verdict:** **Case H — localization INCONCLUSIVE.** No target-independent
support family passes the full pre-registered strong gate; the *strict* primary
well-covered stress test is underpowered (n = 1), so no architecture is authorized.
The strongest *partial* signal is token support (large, seed-consistent,
size-adjusted quartile contrast), but it fails the continuous Spearman gate and is
therefore labelled a **NON-ROBUST ASSOCIATION**. A pre-registered *secondary*,
coarser relation-key diagnostic is powered and is consistent with a **Case E
candidate** (basic coverage insufficient), but the secondary cannot substitute for
the primary stress test.

---

## 1. Objective

Localize which target-independent growth of training-set support best explains the
observed sample-efficiency gain of the frozen `compact-v4-smallhead` model between
nested training subsets N = 1800 → 3600 → 7200. Three pre-registered mechanism
families are compared against each other and against size confounds:

* **F1 token support** — document frequency of the model-visible patch token,
* **F2 relation-context support** — document frequency of the model-visible pair
  relation identity,
* **F3 whole-molecule neighbour density** — frozen PATCH_FULL k-NN density,
* **controls C1/C2** — molecule-size proxies, used only to adjust and never as a
  mechanism candidate.

## 2. Scope, constraints, non-goals

This audit performs **no new training**, adds **no architecture / feature /
tokenizer / probe / OOF model search / HPO / EMA-SWA / checkpoint averaging**, and
compares against **no external data**. It reuses exactly six frozen states
(1800/3600/7200 × seed0/seed1). The 2000-molecule probe is treated as a
**development-mechanism probe**, not a pristine holdout. Official valid and official
test are never loaded.

## 3. Phase firewall and provenance

The module installs a hard firewall: any attempt to extract a non-`train` split
(`zinc_long_range_proxy._load_zinc`, `raw_graph_patch_sufficiency.extract_test_records`,
`…load_train_valid_records`, `post_v4_residual_audit._extract_v4_records`) raises
`FirewallError`. Target/prediction access raises `ProbeAccessError` until Phase U has
hash-locked its support table. Integrity tests 18/19 confirm
`official_valid_loaded = False` and `official_test_loaded = False`; Phase U tests
U16/U17 confirm the same at lock time.

## 4. Frozen inputs

* Six SOUP states (primary) and their raw anchors (robustness), all hashed in
  `model_inventory.json`.
* Nested training subsets from `sample_efficiency_subset_lock.json`
  (`indices_1800 ⊂ indices_3600 ⊂ indices_7200`, D7200 = `optimization_train`).
* Split manifest roles: `adapter_fit` (7200), `adapter_selection` (800),
  `train_probe` (2000). The probe is disjoint from all training subsets.
* Frozen PATCH_FULL block scales read (never refit) from
  `results/raw_graph_patch_system_sufficiency/distance_scale_stats.json`:
  B1 = 0.7720788717, B2 = 1.111203650953604, B3 = 1.1665689833968398,
  B4 = 1.1217517860795814; combined
  `d = sqrt(mean_b (d_b / scale_b)^2)` with k = 8.

## 5. Phase U — target-free support definitions

All support metrics are **document frequencies over unique training molecules**:
each training molecule contributes once regardless of how often a token/relation
occurs inside it. Support metrics never read targets or predictions.

For N ∈ {1800, 3600, 7200} and each probe molecule i:

* `token_unseen`, `token_rare(<5)`, `token_meanlogdf`
* `relation_unseen`, `relation_rare(<5)`, `relation_meanlogdf`
* `nn8_distance` = mean frozen PATCH_FULL distance to the k = 8 nearest D_N molecules
* size controls C1 = log(1+atom_count), C2 = log(1+relation_count)

Deficiency (higher = worse supported), equal weights, frozen before any gain read:

```
D^fam(i) = ( z(unseen) + z(rare) - z(meanlogdf) ) / 3
```

z-scores are computed **within each N** on the 2000-probe distribution (target-free).

## 6. Token family (F1)

Primary token key = `patch.typed_certificate`, the historical
`typed_tokenizer_v1_historical` rooted radius-2 typed incidence certificate
(pynauty). The corrected/later tokenizer is **not** substituted (integrity tests
7/8). The parent certificate (`patch.parent_certificate`) is reported as a clearly
non-authoritative diagnostic only.

Global inventory (10 000 official-train molecules): **6 784 unique typed tokens**
(model vocabulary 6 785 including OOV), **31 unique parent tokens**.

## 7. Relation family (F2)

The model-visible pair object is `(endpoint_i, endpoint_j, relation)` where the
relation descriptor is the 23-D vector built from `distance_one_hot(5) + log-dist +
overlap(5) + boundary(3) + path-bond mean(4) + log-count(1) + adjacent(4)`.

Primary discrete relation key (mode A, authoritative) uses **naturally discrete
source quantities only** (no binning of a continuous descriptor):

```
canonical unordered {typed_certificate_left, typed_certificate_right}
  + exact integer shortest-path distance
  + adjacent bond type when distance == 1 (else -1)
```

with distance recovered as `int(round(expm1(pair_relation[k, 5])))`. This key is
combinatorially large: **655 202 unique relation keys** over the 10 000 training
molecules.

A **secondary, clearly non-authoritative** key replaces the typed endpoints with
their `parent_certificate` ids: **2 069 unique keys**.

## 8. Whole-molecule neighbour density (F3)

Frozen PATCH_FULL distance, normalization frozen on the 7200 reference set and never
refit per N (integrity tests 11/12; reproduction of the saved probe neighbour
manifest is exact). `nn8_distance` is the mean distance to the 8 nearest molecules
of D_N — a whole-molecule density, not a token/relation count.

## 9. Size controls and a documented collinearity

C1 = log(1+atom_count) is the molecule-size proxy (`atom_count == patch_count`
exactly). C2 = log(1+relation_count). Empirically, for **every** probe molecule
`relation_count == C(atom_count, 2)` exactly, so **C2 is a deterministic non-linear
function of C1** and the two controls span an effectively **one-dimensional** size
confound (Spearman(C1, C2) = 1.0). C2 is retained because the protocol mandates two
controls, but it adds no independent size information here. This is recorded in
`size_control_lock.json`. It does not invalidate the size adjustment — the size
surface is still well defined — but it means the adjustment is effectively
single-covariate.

## 10. Phase U inventory findings (target-free)

| N | token unseen | token rare | token singleton frac | relation unseen | relation rare | relation singleton frac |
|--:|--:|--:|--:|--:|--:|--:|
| 1800 | 0.0465 | 0.1324 | 0.4780 | 0.3518 | 0.5958 | 0.7568 |
| 3600 | 0.0307 | 0.0908 | 0.4414 | 0.2825 | 0.5089 | 0.7188 |
| 7200 | 0.0195 | 0.0589 | 0.4158 | 0.2199 | 0.4222 | 0.6845 |

Both token and relation support grow monotonically with N. Relation keys are far
sparser (median document frequency 1 at every N), matching the anticipated
combinatorial explosion.

## 11. Quartiles, well-covered subsets, and lock discipline

Deficiency quartiles (`q_token_*`, `q_relation_*`, `q_neighbor_density_*`) and the
`well_covered_1800` / `well_covered_parent_1800` masks are computed during Phase U
on target-free support only, written to `support_quartile_manifest.json`, and
hash-locked in `support_table_lock.json` (`target_used = false`,
`prediction_used = false`, `quartiles_locked = true`). Phase Y verifies the support
table SHA-256 before unlocking targets.

WELL_COVERED_1800 requires **all** of: U^tok = 0, U^rel = 0, rare^tok = 0,
nn8 ≤ median, atom_count ≤ median. Under the **primary** typed-endpoint relation key
this yields **n = 1** ([UNPOWERED](§21)). Under the secondary coarse key it yields
**n = 361** (powered).

## 12. Phase Y — gain definition

Per molecule and per transition (N → 2N):

```
g(i) = |y_i − ŷ_N(i)| − |y_i − ŷ_2N(i)|      (positive = molecule benefits)
```

Primary estimator = **two-seed mean SOUP**; RAW = seed-matched single raw anchors,
used only for robustness. Recomputed probe MAEs reproduce the frozen prior learning
curve to ≤ 5.6e-17 (all six anchor MAEs 0.225184 / 0.240017 / 0.176633 / 0.175501 /
0.127517 / 0.124268). `gain_summary.json`:

| transition | E_lower (SOUP) | E_upper (SOUP) | G seed0 | G seed1 | G seed-mean |
|--|--:|--:|--:|--:|--:|
| 18→36 | 0.23260 | 0.17607 | 0.048551 | 0.064516 | **0.056533** |
| 36→72 | 0.17607 | 0.12589 | 0.049116 | 0.051234 | **0.050175** |

## 13. Bootstrap provenance reconciliation

The prior `doubling_gain_summary.json` bootstrap used B = 2000, seed = 20260912,
resampling unit = molecule, statistic = mean paired difference, interval =
percentile. Re-running the **same** procedure reproduces every prior point estimate
and 95% CI **exactly** (`mean_diff < 1e-12`): e.g. 18→36 seed0
0.04855078 [0.04139811, 0.05559287], 18→36 seed1 0.06451556 [0.05603414,
0.07270554], 36→72 seed0 0.04911597 [0.04205279, 0.05663624], 36→72 seed1
0.05123362 [0.04238875, 0.06100616]. The reported point estimates lie inside their
intervals. `all_match_prior = true`, `point_estimate_in_ci = true`; **no provenance
discrepancy**. The new audit's own bootstraps use a freshly locked seed 20261006.

## 14. Token family results (primary)

`token_family_results.json` (B = 2000, seed 20261006):

| transition | quartile contrast Q4−Q1 [95% CI] | Spearman ρ [95% CI] | size-adj β₁ [95% CI] | seed0 / seed1 QC |
|--|--|--|--|--|
| 18→36 | **0.02759** [0.00894, 0.04719] | 0.06064 [0.01368, 0.10647] | **0.01259** [0.00414, 0.02208] | 0.03450 / 0.02069 |
| 36→72 | **0.03358** [0.01721, 0.05000] | 0.06098 [0.01389, 0.10495] | **0.01586** [0.00805, 0.02403] | 0.02415 / 0.04300 |

Quartile means show a threshold pattern: the two best-supported quartiles gain
≈0.045 / 0.034 while the two worst-supported gain ≈0.065 / 0.068 (18→36) and
≈0.064 / 0.068 (36→72). The association is **large in the tails, weak and
non-monotone-through-the-middle**, hence the small ρ.

## 15. Relation family results (primary, mode A)

| transition | quartile contrast [95% CI] | Spearman ρ [95% CI] | size-adj β₁ [95% CI] | seed0 / seed1 QC |
|--|--|--|--|--|
| 18→36 | 0.01235 [−0.00659, 0.03154] | 0.03427 [−0.00994, 0.07614] | 0.00642 [−0.00024, 0.01357] | 0.02160 / 0.00309 |
| 36→72 | **0.03930** [0.01937, 0.06378] | 0.06711 [0.02281, 0.10927] | **0.01497** [0.00846, 0.02207] | 0.02813 / 0.05048 |

Relation support is **not** a both-doubling signal: 18→36 fails G1 (CI lower < 0),
so the family is labelled **REGIME-SPECIFIC / PARTIAL**.

## 16. Neighbour-density results (F3)

| transition | quartile contrast [95% CI] | Spearman ρ [95% CI] | size-adj β₁ [95% CI] | seed0 / seed1 QC |
|--|--|--|--|--|
| 18→36 | **0.02818** [0.01006, 0.04530] | 0.05299 [0.00866, 0.09619] | 0.00497 [−0.03179, 0.03556] | 0.02549 / 0.03086 |
| 36→72 | **0.03789** [0.01774, 0.06072] | 0.07846 [0.03333, 0.12338] | 0.01073 [−0.01887, 0.04206] | 0.03460 / 0.04118 |

The quartile contrast survives, but the **size-adjusted β₁ CI crosses zero** in both
doublings. Neighbour density is strongly anti-correlated with molecule size
(Spearman ≈ −0.54), so this is labelled
**SUPPORT_SIGNAL_EXPLAINED_BY_SIZE_OR_WEAK_AFTER_ADJUSTMENT**.

## 17. Cross-family structure

From `support_family_correlation.csv` (Spearman on the probe): token–relation
≈ 0.83–0.85, token–neighbour ≈ 0.43, relation–neighbour ≈ 0.51; token–size
≈ −0.07, relation–size ≈ −0.05, neighbour–size ≈ −0.54. Token and relation support
are therefore **not independent** — a molecule with unsupported tokens usually has
many unsupported relation contexts, which is why their quartile contrasts move
together while neither dominates.

## 18. Fixed 5-fold OLS OOF

`fixed_ols_oof_results.json`, hash-salt folded (5 folds, salt
`sample-efficiency-gain-localization-v1-20261006`):

| transition | size only | +token | +relation | +nn | full |
|--|--:|--:|--:|--:|--:|
| 18→36 | −0.00012 | +0.00400 | −0.00034 | −0.07588 | −0.07229 |
| 36→72 | −0.00541 | +0.00163 | +0.00155 | −0.04996 | −0.04846 |

Out-of-sample R² is at best +0.004 and negative for the full model. Support
deficiency **does not predict per-molecule gain out of sample**; the in-sample
quartile contrast reflects a population-level tail difference, not a per-molecule
predictive relationship.

## 19. Support-increment diagnostics (secondary)

Spearman of the support *improvement* from N to 2N against g (secondary, never part
of the gate): 18→36 token Δ 0.0417, relation Δ 0.0174, nn improvement 0.0482;
36→72 token Δ 0.0461, relation Δ 0.0688, nn improvement 0.0519. All are weak and
consistent with the cross-sectional picture.

## 20. RAW estimator robustness

Using seed-matched raw anchors instead of SOUP, the quartile contrast and Spearman
signs are unchanged for all three families in both doublings
(`raw_direction_conflict = false`). RAW quartile contrasts are similar or slightly
larger (e.g. token 18→36 0.0354 vs SOUP 0.0276). The localization is **not** an
artifact of the SOUP estimator.

## 21. Primary well-covered stress test — UNPOWERED

Primary WELL_COVERED (typed-endpoint relation key) has **n = 1** < 200, so the
stress test is **underpowered** and, per protocol, is reported honestly rather than
rescued by loosening "unseen". The single covered molecule is not interpretable
(R_covered 1.78 / 0.38 across transitions, zero-width bootstrap CI).

## 22. Secondary coarse-relation-key diagnostic (non-authoritative)

Secondary WELL_COVERED (parent-endpoint relation key) is **powered (n = 361)** and
retains:

| transition | G_all | G_covered [95% CI] | R_covered |
|--|--:|--:|--:|
| 18→36 | 0.05653 | 0.04084 [0.03217, 0.04952] | **0.722** |
| 36→72 | 0.05017 | 0.03027 [0.02386, 0.03653] | **0.603** |

Both covered gains are significantly positive and retain 60–72% of the full gain.
This is the signature of **basic structural coverage being insufficient** to explain
scaling (a Case E candidate). Because it uses a coarser relation key it **cannot**
substitute for the primary stress test and is explicitly flagged
`authoritative = false`.

## 23. Gates G1–G4

A family qualifies for **STRONG SUPPORT-LIMITED SIGNAL** only if, in **both**
doublings: G1 quartile contrast ≥ +0.010 with CI lower > 0; G2 Spearman ρ ≥ +0.15
with CI lower > 0; G3 size-adjusted β₁ > 0 with CI lower > 0; G4 both seeds agree in
sign. Results:

| family | G1 | G2 | G3 | G4 | verdict |
|--|--|--|--|--|--|
| token | ✅ both | ❌ (ρ ≈ 0.061) | ✅ both | ✅ both | NON-ROBUST ASSOCIATION |
| relation | ❌ 18→36 | ❌ | ❌ 18→36 | ✅ | REGIME-SPECIFIC / PARTIAL |
| neighbour_density | ✅ both | ❌ (ρ ≈ 0.053/0.078) | ❌ both | ✅ both | SIZE / WEAK AFTER ADJUSTMENT |
| relation_parent (secondary) | ❌ | ❌ | ❌ | ✅ | NO MATERIAL ASSOCIATION |

No family passes G2. Token is the only family that passes G1, G3 and G4 in both
doublings.

## 24. Case determination

The strict pre-registered hierarchy:

* Case A/B/C (single strong family) — **not applicable**: no family strong.
* Case D (≥2 strong families) — **not applicable**.
* Case E (no family strong, primary well-covered R ≥ 0.60 both) — **not claimable**:
  the primary well-covered test is underpowered.
* Case F/G — not triggered.
* Case H (well-covered subset underpowered) — **triggered**.

→ `final_case = "H"`, `verdict = "SAMPLE-EFFICIENCY GAIN LOCALIZATION INCONCLUSIVE"`,
`authorized_for_design = false`, `full_training_authorized = false`.

## 25. Why the verdict is H and not E (strict discipline)

The underpowering is a direct consequence of the *pre-registered primary* relation
key being the exact typed-endpoint relation identity (655 202 keys). The protocol
requires the well-covered subset to be defined on that key and forbids loosening the
"unseen" definition. Using the powered secondary key to claim Case E would be
exactly the loosening the protocol prohibits. The audit therefore reports H as the
strict verdict and E only as a clearly-labelled sensitivity.

## 26. Best-supported reading (not authorized)

`mechanism_decision.json:best_supported_reading` = "Case E candidate": the
best-supported token quartile still retains most of the doubling gain (Q1 means
0.0451 of 0.0565 at 18→36; 0.0344 of 0.0502 at 36→72), and the powered coarser-key
well-covered subset retains 0.72 / 0.60. Together with the OOF R² ≈ 0, the evidence
**suggests** that additional data is not primarily filling a basic token/relation
coverage deficit. This is a *hypothesis*, not an authorization, pending a properly
powered primary-key stress test.

## 27. Falsification and limitations

* The audit **does not** show that support is irrelevant: token quartile contrasts
  are ~2.8–3.4× the materiality gate and size-adjusted β₁ is positive.
* The audit **does not** show a compositional/function-sharing mechanism: that
  hypothesis was not tested here and remains gated behind Case E/F.
* The audit **does** show that no support family is a robust, per-molecule predictor
  of gain (OOF R² ≤ 0.004) and that any relationship is concentrated in the tail and
  largely absent after size adjustment for the whole-molecule family.
* The primary relation key's sparsity is itself a finding: relation-context support
  may simply be too fine-grained at these training sizes to have measurable
  molecule-level leverage.
* Bootstrap intervals use the percentile method, as in the prior artifacts; no BCa
  sensitivity was required.

## 28. Relation to prior NO-GOs

The binding NO-GOs (tokenizer substitution, P1/P2, cell, covariance, triad,
larger-radius patches, compositional patch sharing) all remain in force. This audit
adds no new evidence that changes their premises; in particular it does **not**
reopen the compositional-patch-sharing NO-GO, because the coverage explanation was
not cleanly rejected on the primary key.

## 29. What would change the conclusion

1. A larger probe (or a pre-registered coarser primary relation key) that makes the
   primary well-covered stress test powered (n ≥ 200) — would resolve H → E or F.
2. A continuous support statistic with a more monotone relationship to gain (e.g.
   tail-focused, or a learned-but-pre-registered deficiency) — would test whether the
   token association is genuinely non-robust or merely mis-measured by Spearman.
3. An independent, larger training-size sweep to confirm the threshold shape (Q1/Q2
   flat, Q3/Q4 high) is not a finite-sample artifact.

## 30. Artifact map

* Locks/provenance: `audit_protocol_lock.json`, `model_inventory.json`,
  `subset_inventory.json`, `prediction_provenance.json`,
  `probe_predictions_locked.parquet`, `recomputed_learning_curve.json`,
  `bootstrap_provenance_check.json`.
* Phase U: `support_table_locked.parquet`, `support_table_lock.json`,
  `support_quartile_manifest.json`, `well_covered_subset_manifest.json`,
  `token_document_frequency_summary.json`, `relation_document_frequency_summary*.json`,
  `relation_key_inventory.json`, `size_control_lock.json`,
  `patchfull_metric_inventory.json`, `phaseU_integrity.json`.
* Phase Y: `gain_table.parquet`, `gain_summary.json`, `token_family_results.json`,
  `relation_family_results.json`, `relation_parent_family_results.json`,
  `neighbor_density_results.json`, `size_adjusted_results.json`,
  `fixed_ols_oof_results.json`, `support_family_correlation.csv`,
  `support_increment_diagnostics.json`, `well_covered_stress_test.json`,
  `raw_estimator_robustness.json`, `initial_error_control.json`.
* Decision: `mechanism_decision.json`, `top1_sample_efficiency_mechanism.json`,
  `answers_q1_q24.json`, `final_decision.json`.
* Integrity/tests: `integrity_tests.json` (20/20 pass),
  `tracks/ksvd/tests/test_sample_efficiency_gain_localization_audit.py`.
* Figures: `figures/figure1_mechanism_schematic.png`,
  `figure2_gain_by_deficiency_quartile.png`, `figure3_well_covered_stress.png`,
  `figure4_support_gain_association.png`.
