# Structural Response Learnability Audit v1 — Protocol

Status: read-only audit. No neural-network training, no model modification, no
hyperparameter search, no GPU, no new experiments beyond the analyses below.
Official Valid and Test label values are not used. The label vector is masked
to NaN outside the official Train rows immediately after loading, and all
label-dependent analysis is restricted to the official Train rows.

## 1. Question

The audit asks whether the real matched-pair directory contains enough
independent structural evidence to learn a *conditional structural-response*
model: a model that predicts the label change of a replacement type from the
local structural context, and transfers that response to Dev molecules.

The audit separates four questions:

1. How much independent support does each replacement type have?
2. How large and stable are the real substitution responses within Train-sub?
3. Do two fixed lightweight references transfer from Train-sub to Dev?
4. If a conditional model is not supported, is the limiting gap type coverage,
   context coverage, or supervision strength?

The audit is associational. It does not estimate chemical causal effects.

## 2. Inputs

All inputs are read-only.

| input | path | sha256 |
|---|---|---|
| real matched-pair directory | `tracks/structural-sufficiency/task_response_composition_s0_v1/data/trc_cache_lipo.npz` | `c53fee5d16869b7b7ff721d404a84289d0a315d392440ebfb436cbd307e3abc3` |
| ACRA local identities and split cache | `tracks/structural-sufficiency/acra_relation_generalization_pilot_v1/data/acra_cache_lipo_v2.npz` | `1f6d7a3f870513a83b50f6aa6edbe0143f37c2a883308627f47145e2078fdc3b` |
| Train-sub / Dev assignment | `tracks/structural-sufficiency/acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json` | `675289f6f997008d44fdabb4ab65350459fe5bccc5e1e5703e31e31fd794e845` |

The loader asserts these hashes. The split is the frozen ACRA scaffold split:
Train-sub 2856 molecules, Dev 504 molecules, seed 0, scaffold-disjoint, both
subsets of the 3360 official Train molecules.

## 3. Label guard

The TRC label vector is copied and masked to NaN outside the official Train
rows immediately after loading. Every label-dependent function only uses rows
inside the mask. The audit writes:

- `test_labels_used = false`
- `valid_labels_used = false`
- `valid_used_for_model_selection = false`

No Dev label is used to fit, select, calibrate, or early-stop any model. Dev
labels are used only to compute evaluation metrics after predictions are fixed.

## 4. Unit definitions

- **Pair row**: one real matched pair `(G_a, G_b)` with a fixed cut bond and a
  fixed environment fragment `C`. The canonical direction is the directory's
  `(word_a, word_b)` orientation; it is a bookkeeping direction, not a chemical
  direction.
- **Exact replacement type** `T`: the TRC `pair_transform` identity
  `(word_a, word_b, raw cut-bond attributes)`. This is the primary type.
- **Coarse replacement type**: `(radius-1 identity of word_a, radius-1 identity
  of word_b, raw cut-bond attributes)`. It is a sensitivity level that merges
  exact types; it is not the primary type.
- **Full context** `C`: the exact matched environment fragment of the cut, as
  stored by the pair directory. It is the finest context identity and the
  primary independence unit.
- **Radius-1 / radius-2 contexts**: the ACRA `r1` and `r2` local identities of
  the environment side of the cut. `r2` refines `r1`.
- **Pair component**: a connected component of the matched-pair graph on
  molecules, using pair rows as edges. Two components are used:
  - Train-sub graph: edges with both endpoints in Train-sub (398 components);
  - full Train graph: all 4646 analyzed pairs (458 components).
- **Independent evidence unit**: one distinct full context `C` within a type.
  Raw pair counts are always reported separately from molecules, scaffolds,
  pair components, and type-context cells. No claim of statistical independence
  is made beyond these reported units.

## 5. Frozen settings

| setting | value |
|---|---|
| bootstrap replicates | 2000 |
| bootstrap seed | 0 |
| confidence interval | percentile 95% |
| A shrinkage constant `k` | 1.0 |
| B ridge penalty `lambda` | 10.0 |
| WL fingerprint width | 256 bits per ball |
| WL fingerprint iterations | 2 |
| descriptor vocabulary / standardisation | Train-sub only |

No setting is selected on Dev or Valid.

## 6. Task 1 — support

For every exact and coarse type, report:

- Train-sub pairs, all pairs, and scope counts;
- independent molecules, distinct full contexts, distinct radius-1 contexts,
  distinct radius-2 contexts, scaffolds, and pair components;
- the same counts over all analyzed pairs as a descriptive check.

Aggregate the number of types and pairs at `>=2`, `>=3`, and `>=5`:

- distinct full contexts (primary);
- distinct radius-1 and radius-2 contexts (coarsenings);
- pair components and scaffolds (molecule-level dependence);
- joint context+component and context+scaffold conditions;
- a strict condition requiring `>=k` contexts, `>=2` components, and `>=2`
  scaffolds.

Report de-duplicated unique units over the selected types (types, pairs,
type-context cells, molecules, scaffolds, components). Per-type incidence sums
are labelled as incidences and are not interpreted as unique counts.

Checks:

- duplicate full pair rows and exact reverse rows;
- multi-pairs per molecule pair and per molecule;
- the maximum number of pairs per `(exact type, full context)` cell;
- pair components crossing the Train-sub / Dev boundary.

## 7. Task 2 — response variation

For exact types with at least two distinct full contexts in Train-sub, and for
the analogous coarse types:

- `delta_y = y(G_b) - y(G_a)` for every real pair;
- same-type mean, SD, min, max, range, mean absolute deviation;
- pairwise absolute differences between contexts of the same type;
- direction stability: sign consistency `max(#positive,#negative)/n`, the
  fraction of types with any reversal, and the pair-weighted sign consistency;
- within-type ANOVA: fraction of within-type variance explained by radius-1
  and radius-2 context groups;
- variance decomposition on the multi-context subset:
  - coarse type, exact type, radius-1 alone, radius-2 alone;
  - exact type + radius-1, exact type + radius-2, exact type + radius-1+radius-2;
  - exact type + additive full-context dummies;
  - an in-sample OLS R², adjusted R², and a 1000-replicate permutation null for
    the exact-type effect.

Support classes are `2`, `3-4`, `5-9`, and `>=10` full contexts. The audit does
not interpret per-type `eta2` at two contexts as evidence: with two points the
statistic is effectively binary. Pooled within-type ANOVA is the primary
context-explanation statistic.

Within-Train-sub predictive check: leave-one-pair-out (LOMO) and
leave-one-component-out type-mean prediction with the same shrinkage, evaluated
against the Train-sub global mean. This tests whether the type mean predicts a
held-out pair from the same type without using that pair.

Noise floor: **not estimated**. The audit first checks for duplicate canonical
structures and repeated `(exact type, full context)` cells. If there are none,
within-type across-context variation cannot be separated from measurement noise
and is reported as an indistinguishable residual, not as a noise estimate.

Failure cases list high-support types with low sign consistency, large
within-type SD, or opposite-sign contexts, plus the cross-scope prediction
failures from Task 3.

## 8. Task 3 — transfer

Evaluation scopes:

- **cross** (primary): Train-sub–Dev real matched pairs, 474 pairs, 129 Dev
  molecules.
- **dev–dev** (secondary): Dev–Dev real matched pairs, 611 pairs, 72 full-Train
  pair components.
- **combined**: 1085 evaluation pairs, clustered by full-Train pair component.

No pair containing a Dev molecule is used to fit either reference.

### A. Transformation-only reference

For a seen exact type, predict the Train-sub empirical mean delta of that type,
shrunk to the Train-sub global mean with `k=1`:

```
A(type) = (sum_train_delta(type) + k * global_mean) / (n_type + k)
```

Unseen type: predict the Train-sub global mean.

For evaluation pairs, no Train-sub pair can share a molecule with a Dev pair
because the split is molecule-disjoint, so the leave-molecule-out and full type
means coincide. For fitting B, the Train-sub offset is leave-one-pair-out and
leave-molecule-out to avoid self-inclusion.

### B. Context-conditioned linear reference

`B = A_offset + ridge_correction`, where the correction is a linear function of
fixed context descriptors:

- radius-1 and radius-2 balls around the environment attachment, cut bond
  removed, computed from the raw attributed graph;
- **compact block** per ball: atom/bond category fractions (Train-sub
  vocabulary) plus log atom/bond counts, cycle rank, root degree, root-in-ring,
  cycle indicator, and bond/atom ratio;
- **fingerprint block** per ball: rooted Weisfeiler–Lehman count fingerprint,
  256 hashed bits, 2 iterations, initial labels include atom features and root
  distance;
- concatenated radius-1 + radius-2 blocks: 634 raw features, 583 with non-zero
  Train-sub variance;
- standardisation from Train-sub only.

Ridge target: `delta - A_lomo` on Train-sub pairs. The correction is centred at
zero with component-balanced weights (each Train-sub pair component contributes
equally, weights normalised to mean 1). `lambda=10` is fixed; no search.

Primary B uses the full compact+fingerprint block. The following are **post-hoc
robustness checks, not model selection**:

- compact-only B (81 used features);
- B with `lambda=100`;
- B fitted with a self-included Train-sub type mean (an optimistic offset that
  is not used for the primary result).

### Evaluation and uncertainty

Metrics are delta-level: MAE, RMSE, correlation, sign agreement, bias, and
prediction/truth SD ratio. They are not whole-molecule RMSE and are not
comparable to the parent rounds' whole-molecule tables.

Cluster bootstrap, 2000 replicates, seed 0, percentile 95%:

- cross: resample Dev molecules (129 clusters);
- dev–dev: resample full-Train pair components (72 clusters);
- combined: resample full-Train pair components (103 clusters).

The same resampled clusters are used for all predictors in a comparison.
Subsets report their own cluster counts; a block is flagged `ci_reliable=false`
when it has fewer than 5 clusters.

Reported subsets:

- seen exact type / unseen exact type;
- seen type with at least two Train-sub contexts;
- seen full context / unseen full context;
- seen radius-1 / seen radius-2;
- support bins by number of Train-sub pairs and number of Train-sub contexts
  for the type.

Paired comparisons use `MAE(B)-MAE(A)` and `MAE(A)-MAE(global)` with the same
cluster bootstrap. Negative means the second model is better.

Calibration is descriptive: OLS slope of truth on prediction, intercept, R²,
and SD ratio, with a cluster bootstrap CI for the slope.

Failure cases list the worst A errors and the worst B-minus-A degradations,
de-duplicated by unordered molecule pair, separately for all cross pairs and
for cross pairs with a seen exact type.

## 9. Decision rules

- **Stable base effect**: LOMO type-mean MAE is lower than the Train-sub global
  MAE with a non-overlapping bootstrap CI, and pair-weighted sign consistency
  is above 0.5. This is evidence of a repeatable association, not causality.
- **Transferable context information**: B improves over A with the entire
  `MAE(B)-MAE(A)` CI below zero on **both** cross and dev–dev, and the gain is
  not confined to in-sample fitting. A positive or CI-including-zero result is
  reported as no evidence of transferable context information.
- **Independent repeat**: at least two distinct full contexts, with molecules,
  scaffolds, and components reported. The strict condition additionally
  requires at least two components and at least two scaffolds.
- **Data sufficient for a conditional model**: enough independent type-context
  repeats to fit and validate the conditional part on held-out molecules, with
  a transfer gain over A. Coverage alone is not sufficient.
- **Noise floor**: only with repeated structures or repeated
  `(exact type, full context)` cells. Neither exists here.

## 10. Engineering checks

The runner asserts:

- input hashes and split sizes;
- 4646 official-Train pairs with scopes 3561 / 474 / 611;
- transform-array consistency and zero environment-key mismatches;
- features finite, context-identity deterministic, vocabulary from Train-sub;
- labels finite on official Train and NaN outside it;
- response type counts and pair totals match the support thresholds;
- transfer subset counts match support coverage;
- bootstrap settings are the frozen ones.

It records elapsed time, peak RSS, output sizes and sha256 hashes in
`analysis/checks.json`.

## 11. Reproducibility

From the audit directory:

```bash
uv run --project /home/calendar/code/research python code/run_audit.py
```

Outputs:

- `analysis/support.json`
- `analysis/response_variation.json`
- `analysis/transfer_check.json`
- `analysis/checks.json`

The code is in `code/`:

- `audit_common.py`: input loading, pair table, units, bootstrap;
- `audit_features.py`: fixed context descriptors;
- `audit_support.py`: Task 1;
- `audit_response.py`: Task 2;
- `audit_transfer.py`: Task 3;
- `run_audit.py`: runner and checks.

## 12. Exclusions

- No chemical-causal interpretation.
- No claim that a local context descriptor is the true generative cause of a
  response.
- No claim that the residual after type+context is measurement noise.
- No Test/Valid performance claim.
- No model selection on Dev or Valid.
- No hyperparameter search; the post-hoc B variants are robustness checks only.