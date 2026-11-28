# PROTOCOL — 可解释结构规律发现与稀疏预测研究 v0

Round: `interpretable-structural-rule-discovery-v0`
Directory: `tracks/structural-sufficiency/interpretable_structural_rule_discovery_v0/`
Seed: `0` (single seed, no multi-seed)
Predecessors (read-only): `molhiv_multigranular_structure_sharing_v0` (`MSDS-v0`),
`structural_gain_attribution_v1` (`SGA-v1`).
Compute: CPU-only. Wall-clock target 60 min, hard limit 120 min.

This file is written and committed **before** any rule model of this round is fitted.
Everything below is pinned. Engineering errors may only be repaired and logged in
`analysis/provenance.json` (§11); no design element may change after a Test number
is seen.

---

## 1. Question

Can the explicit structural knowledge already used by fitted molecular predictors be
distilled into a *small* number of structurally meaningful, cross-molecule reusable
prediction rules?

* Q1 (phase B) — how much predictive power do a handful of explicit structural rules carry?
* Q2 (phase C, 6.2) — do structural **threshold** rules add non-linear value over linear counts?
* Q3 (phase C, 6.3) — do **joint** conditions found in real tree paths beat matched random pairs?
* Q4 (phase D) — do we obtain 3–5 real structural regularities worth continued study?

Out of scope: new GNN, distillation, structure-response losses, vocabulary extension,
subgraph re-enumeration, rule depth > 2, extra seeds, extra datasets, extra hyper-parameter
search, > 128 single rules, extra structure scales.

Evidence classes: `EXACT_MATCH` (reused verbatim), `SAME_SPLIT_REFERENCE`
(same split/metric, different recipe), `HISTORICAL_REFERENCE` (quoted only).

## 2. Data and identity

* `ogbg-molhiv` official scaffold split, graph index = global 0-based row id,
  **32 901 / 4 113 / 4 113**.
* Frozen structural representation, reused unchanged from MSDS-v0 via
  `SGA-v1/bundle/sav1_inputs.npz`:
  * `C` — 754 COARSE types, `F` — 3 652 FULL types, 4 406 explicit structural columns;
  * values `x_s(G) = count_s(G) / sqrt(max(n_atoms(G), 1))`;
  * COARSE/FULL column order **is** the frozen vocabulary order (no re-derivation).
* Structure descriptions use the corrected OGB categorical decoding of
  `SGA-v1/analysis/registry_v1/` (`structure_registry_{coarse,full}_v1.csv`).
* Test policy: `EXPOSED_DEVELOPMENT_TEST` — evaluated and reported every arm, never
  used to select a rule, a threshold, a capacity, a C, or a design element.
* All rule discovery, filtering, standardisation and parameter fitting use **Train only**.

## 3. Reused assets (never refitted)

| asset | source | role |
|---|---|---|
| C / F design blocks | `SGA-v1/bundle/sav1_inputs.npz` (sha256 pinned) | rule inputs |
| **X-C** booster | `SGA-v1/runs/phaseA/X-C_model.json` (H + C, 500 trees) | rule source |
| **X-F** booster | `SGA-v1/runs/phaseA/X-F_model.json` (H + F, 500 trees) | rule source |
| **X-CF** booster | *not saved* by MSDS-v0 Phase D (`run_phaseD_xgb.py` dumps no model) | unavailable — see §3.1 |
| B2 | MSDS-v0 `data/phaseB_coefficients.npz` + `data/phaseB_predictions.npz`, C=0.03 | frozen offset model |
| B0/B3/X-C/X-F/X-CF/N-CF metrics | predecessor reports | historical reference |

### 3.1 Booster availability (decided before any fit)

X-CF weights do not exist and are **not** retrained (round rule). The rule source is
therefore **X-C (COARSE) and X-F (FULL)**. Consequence, stated up front: every
`PROJECTED_PAIR` is a same-granularity pair (C–C or F–F); no mixed C–F pair can be
mined. This is a coverage limitation of the round, recorded in `REPORT.md` §limitations
and in `analysis/provenance.json`. The degradation branch `TREE_RULE_SOURCE_UNAVAILABLE`
(§4.5 of the task) is **not** taken, because two real boosters exist.

## 4. Phase A — candidate rule discovery (Train only)

### 4.1 Single-structure rules

For every split node of X-C / X-F whose feature is a structural column
(`feature_index >= 2258`, `block_column = feature_index - 2258`) and threshold `t`:

```
r(G) = 1[ x_s(G) >= t ]        # the right-hand ("no") child of the raw tree
complement: 1[ x_s(G) < t ]    # the left-hand ("yes") child
```

The input matrices are dense with no NaN after the historical imputation, so the
XGBoost `missing` branch is never taken; this is checked explicitly (§10.2).
Floating-point comparison is evaluated on the same float32 design XGBoost consumed.

Each candidate records: structure identity (granularity + column + canonical key +
decoded attributes), actual threshold, source arm, tree ids, split-node count, Train
satisfied/unsatisfied counts, per-side scaffold counts and positive/negative counts,
and the Train raw-count range on both sides.

Scaling caveat: because `x_s = count_s / sqrt(n_atoms)`, a threshold on `x_s` is **not**
a statement about the raw occurrence count. Raw-count phrasing is used only when the
two sides are logically equivalent, which is checked and recorded per rule
(`threshold_reading` field).

Single-rule filters (Train only): each side ≥ 30 Train molecules, each side ≥ 5 Train
scaffolds, non-constant, duplicate activation vectors removed (merged sources logged).
Ranking: number of distinct source trees (desc), then stable id. Keep at most 512.

### 4.2 Joint rules from real root-to-leaf paths

Traverse every tree with the real Train design. Each leaf path yields its ordered list
of `(structure, threshold, direction)` conditions. For every pair of conditions on one
path with **different** structural features, emit a `PROJECTED_PAIR`

```
r_AB(G) = 1[ cond_A(G) AND cond_B(G) ]
```

with the original directions and thresholds preserved. Morgan bits / RDKit descriptors
on the same path are ignored (they are not projectable to a structure rule) — the
candidate is therefore **not** the original tree rule and inherits **no** leaf weight.
`EXACT_PATH` counts are recorded but not used for modelling.

Joint filters: both single conditions present in the §4.1 pool; all four 2×2 Train
cells ≥ 20 molecules and ≥ 3 scaffolds; exclude equivalence, deterministic containment
and constant combinations; drop duplicate activation vectors. Ranking: real co-path
occurrence count (desc), stable id. Keep at most 256 (`K_max`). Filtered counts and
reasons are recorded.

### 4.3 Matched random joint rules

From the same single-rule pool, pairs that **never** co-occur on a real tree path.
seed = 0. Match, in priority order, on `(granularity_pair, family_pair)`, the two
marginal Train activation frequencies, the joint activation frequency and the minimum
2×2 cell support; one-to-one, without replacement; the same §4.2 filters apply. Matching
coverage and residual frequency differences are reported. The control answers only
"do tree-derived conjunctions carry more predictive value than statistically similar
ordinary conjunctions?" — it is not a topological random-graph control.

### 4.4 Semantic boundary

Joint rules denote **whole-molecule condition co-occurrence** only. No chemical
interaction, no local connection claim, no causality, no mechanism.

## 5. Phase B — compact rule model

Model (no Morgan, no descriptors, no embedding):

```
z(G) = b + a1*log1p(n_atoms) + a2*log1p(n_bonds) + sum_k w_k * r_k(G)
```

R0 = size terms only. Forward selection on the Train weighted-logistic gradient:

```
g_j = sum_i sw_i (y_i - p_i) r_ij
h_j = sum_i sw_i p_i (1-p_i) r_ij^2 + 1/C
score_j = g_j^2 / h_j
```

`sw` = class-balanced weights (`n/(2 n_c)`, sklearn `class_weight='balanced'`);
selection `C = 0.03` fixed; up to 16 rules per batch then joint refit; at most one
threshold per structure feature; ties broken by candidate rank. All rule inputs and the
two size terms are standardised with **Train-only** mean/std and the score uses the
standardised inputs. Nested sets: 16 / 64 / 128 rules (fewer if the pool is smaller).
Final C ∈ {0.03, 0.3} chosen by highest Valid ROC-AUC, tie → smaller C. Coefficients are
mapped back to raw rule units, including the intercept.

Arms: `R0`, `R16`, `R64`, `R128`. No extension beyond 128 rules.

## 6. Phase C — increment on the frozen B2 prediction

### 6.1 Fixed B2

`z_B2` = the stored raw logit of B2 (C=0.03). B2 is replayed, not refitted
(`z = b + [M|C|F]·w` recomputed from the stored coefficient vector and compared to the
stored logits). New model:

```
z(G) = z_B2(G) + b_delta + sum_k w_k r_k(G)
```

Weighted logistic regression with offset, fitted with an analytic-gradient L-BFGS
(`scipy.optimize.minimize`), objective
`sum_i sw_i BCE(y_i, z_i) + ||w||^2/(2C)`, `b_delta` unpenalised. Gradient checked by
finite differences; zero-correction replay tested. C grid {0.03, 0.3}, Valid-selected.

### 6.2 Linear vs threshold correction (identical structure identities)

`R64` fixes 64 structure identities. Then

| arm | correction inputs |
|---|---|
| `B2-L64` | the 64 structures' frozen scaled **counts** `x_s(G)` |
| `B2-U64` | the 64 **threshold predicates** `1[x_s >= t]` from the R64 rules |

Same coefficient count, same Train-only standardisation rule, same `sw`, same C grid.

### 6.3 Joint-rule test (base = frozen `B2-U64`)

Only pairs whose two single conditions are already represented in U64 (a condition's
complement is allowed) enter. Score with the frozen B2-U64 Train residual, same
`g²/h` score with C=0.03, select at most 16 real pairs; select the same number of
matched random pairs and the same number of *new* single rules (not in U64) with the
same budget. `K = min(real, random, new_single, 16)`; if `K < 8` stop and mark
`INSUFFICIENT_PAIR_SUPPORT`.

| arm | input |
|---|---|
| `B2-U+S` | B2-U64 + K new single rules |
| `B2-U+P` | B2-U64 + K real co-path joint rules |
| `B2-U+N` | B2-U64 + K matched random joint rules |
| `R64+P` | size + U64 + the same K joint rules, fitted directly on y |

All four only refit the correction; B2 itself is frozen. All three contrasts
(`P−U64`, `P−S`, `P−N`) are reported.

## 7. Evaluation

* Primary: OGB official ROC-AUC (sklearn cross-checked). Auxiliary: average precision.
  Train / Valid / Test all reported; Test marked `DEVELOPMENT_TEST`.
* Per-molecule raw logits, labels, graph ids and split saved for every arm.
* Main contrasts: 1 000 paired **scaffold** bootstrap resamples, seed 0, whole
  Bemis–Murcko scaffold groups within the evaluated split, 95 % percentile interval.
  Bootstrap describes sample uncertainty of fixed predictions only.
* Faithfulness: `logit == intercept + size terms + sum(active contributions)` must close
  to floating-point tolerance after converting back to raw units; representative rules
  are switched off/on and the output change is compared to the rule contribution.
  This is *model-internal* faithfulness, not a physical causal claim. B2-augmented
  models are only interpretable in their correction part.
* Complexity: non-zero rules, distinct structures, conditions per rule, active rules per
  molecule, and the distribution of size / single-rule / joint-rule logit contributions.

## 8. Phase D — knowledge

Small case study only, no new model. Priorities: (1) the COARSE 419 / FULL
2078, 2525, 1788, 2275 aromatic-carbon–sulfur family from
`SGA-v1/analysis/case_structures.json` (attributes, local environment, molecule and
scaffold overlap, size confounding, Valid/Test direction, whether B2 and the new rule
models actually use them); (2) 3–5 representative rules as knowledge cards with the
structure object, the exact condition, data support, prediction evidence, a real
supporting molecule and a counterexample, competing explanations, and the current
epistemic status. Model-computation facts, data associations and unsupported hypotheses
are never mixed.

## 9. Budget and stop rules

CPU only. Target 60 min, hard limit 120 min. Priority: identity checks → single rules →
R0/R16/R64/R128 → B2 linear/threshold contrasts → joint contrasts → knowledge cards.
If the budget runs short the matched arm groups are finished first. Forbidden: refitting
any historical model, re-enumerating the library, new vocabularies, depth > 2 rules,
capacity creep above 128 rules, extra seeds, other datasets.

## 10. Required checks (lightweight)

1. split ids / labels / n_atoms identical to the frozen bundle; C and F column counts and
   canonical keys identical to `registry_v1`;
2. booster feature index → block column mapping; every mined threshold re-evaluated on
   the real design reproduces the tree branch (`missing` unused);
3. single-rule candidates computed with Train only (no label access to Valid/Test);
4. joint-rule main effects present in the shared base model, and the support matching of
   real vs random pairs;
5. offset-logistic gradient vs finite differences; `R0` vs sklearn `LogisticRegression`
   on the same design; B2 zero-correction replay;
6. contribution-sum closure for every rule model;
7. all reported metrics recomputed independently from the stored predictions.

## 11. Deviation log

Filled during execution in `analysis/provenance.json` (`engineering_deviations`). Only
real engineering faults may be fixed, each with affected runs and a statement of whether
a reported number changes.
