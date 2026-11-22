# Structural Response Learnability Audit v1 — Report

Protocol: `PROTOCOL.md`. Code: `code/`. Machine-readable results:
`analysis/support.json`, `analysis/response_variation.json`,
`analysis/transfer_check.json`, `analysis/checks.json`.

All intervals below are 95% percentile cluster bootstrap intervals
(2000 replicates, seed 0). Metrics are **delta-level** (`y(G_b)-y(G_a)`), not
whole-molecule RMSE. The audit is associational; no chemical-causal
interpretation is made.

## Results at a glance

- The analyzed directory has 4646 real matched pairs on official Train:
  3561 Train-sub–Train-sub, 474 Train-sub–Dev (cross), 611 Dev–Dev.
- Exact replacement types are overwhelmingly singleton-supported: 2273 of
  2651 supported types have one Train-sub pair. Only 378 types (1288 pairs) have
  at least two distinct full contexts; 50 types (505 pairs) have at least five.
  Under a strict component+scaffold independence condition the counts are
  344 types (1211 pairs) and 49 types (500 pairs).
- Exact replacement effects are stable on average but not homogeneous:
  pair-weighted sign consistency is 0.880 at `>=2` contexts, and leave-one-out
  exact-type prediction has MAE 0.350 against a global baseline of 0.570.
  However, 113 of 378 types show at least one sign reversal, and the median
  absolute difference between two contexts of the same type is about 0.27.
- The transformation-only reference A transfers to seen exact types
  (cross MAE 0.429, Dev–Dev 0.533, against global 0.647/0.740), but only 69/474
  cross pairs (14.6%) and 243/611 Dev–Dev pairs (39.8%) have a seen exact type.
  The overall gain over the global mean is therefore small: 0.032 (cross) and
  0.082 (Dev–Dev).
- The context-conditioned ridge reference B does **not** transfer. It is worse
  than A on cross (+0.089 MAE, CI [0.048, 0.137]) and Dev–Dev (+0.181,
  CI [0.117, 0.263]). Compact-only B, `lambda=100`, and a self-included-offset
  variant are also not better. In-sample context explains most within-type
  variance, but the fixed descriptors do not carry that information to Dev.
- The data are **not sufficient** for a conditional structural-response model
  at this split and support. The binding gaps are exact-type coverage (14.6% of
  cross pairs) and independent type-context supervision (31 cross pairs with a
  seen type and at least two contexts; B degrades even there). Context identity
  coverage is comparatively high for cross (79.7% of full contexts seen), so
  the primary gap is not simply “unseen contexts.”

## 1. Scope and inputs

Read-only audit of the real matched-pair directory
`task_response_composition_s0_v1`. Inputs and sha256:

| input | sha256 |
|---|---|
| `task_response_composition_s0_v1/data/trc_cache_lipo.npz` | `c53fee5d16869b7b7ff721d404a84289d0a315d392440ebfb436cbd307e3abc3` |
| `acra_relation_generalization_pilot_v1/data/acra_cache_lipo_v2.npz` | `1f6d7a3f870513a83b50f6aa6edbe0143f37c2a883308627f47145e2078fdc3b` |
| `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json` | `675289f6f997008d44fdabb4ab65350459fe5bccc5e1e5703e31e31fd794e845` |

The split is Train-sub 2856 / Dev 504, scaffold-disjoint, seed 0, both subsets
of the 3360 official Train molecules. Labels are masked to NaN outside official
Train immediately after loading. Valid and Test label values were never used;
Valid was not used for model selection. The run took 79.1 s and peaked at 649 MB RSS.

The pair directory has 4646 official-Train pair rows. Their scopes are:

| scope | pairs | dependence unit | delta mean | delta SD | mean abs delta |
|---|---:|---|---:|---:|---:|
| Train-sub–Train-sub | 3561 | 398 pair components | -0.032 | 0.912 | 0.670 |
| Train-sub–Dev (cross) | 474 | 129 Dev molecules | -0.141 | 1.016 | 0.775 |
| Dev–Dev | 611 | 72 full-Train pair components | +0.088 | 1.075 | 0.760 |

The two Dev scopes are not exchangeable with Train-sub even at the mean level:
cross pairs are more negative, Dev–Dev pairs more positive, than the Train-sub
mean of -0.032. A global-mean predictor therefore carries a positive bias of
about +0.126 on cross and a negative bias of about -0.084 on Dev–Dev. This is a
distribution shift, not a model-fitting error, and it limits all transfer
metrics.

## 2. Task 1 — independent support

### 2.1 Exact types

There are 3399 exact replacement types. 2651 have at least one Train-sub pair;
2273 of those are singletons. Per-type records for every exact and coarse type
(pairs, molecules, full/radius-1/radius-2 contexts, scaffolds, components,
scope counts) are in `analysis/support.json`; the tables here aggregate them.
The support levels are:

| exact type support | types | Train pairs = type-context cells | unique molecules | unique scaffolds | unique pair components |
|---|---:|---:|---:|---:|---:|
| `>=2` full contexts | 378 | 1288 | 999 | 305 | 275 |
| `>=3` full contexts | 153 | 838 | 778 | 230 | 217 |
| `>=5` full contexts | 50 | 505 | 583 | 154 | 174 |

For exact types, each `(type, full context)` cell contains exactly one real
pair (verified: maximum cell multiplicity 1 over all 4646 pairs), so the pair
count and the type-context-cell count are identical. The 1288 pairs at `>=2`
are still not 1288 independent observations: they span 999 molecules, 305
scaffolds, and 275 pair-graph components. The nominal pair count is 4.7 times
the component count at this level.

The strict condition (at least two contexts, at least two components, and at
least two scaffolds) selects:

| strict exact support | types | pairs |
|---|---:|---:|
| `>=2` contexts, `>=2` components, `>=2` scaffolds | 344 | 1211 |
| `>=3` contexts, `>=2` components, `>=2` scaffolds | 147 | 817 |
| `>=5` contexts, `>=2` components, `>=2` scaffolds | 49 | 500 |

Thus 34 of the 378 multi-context exact types fail at least one of the two
molecule-level independence conditions. Fourteen have only one pair component;
27 have only one scaffold (the categories overlap). At `>=5`, 49 of 50 types
are strictly independent.

Radius-1 and radius-2 context support is lower than full-context support
because the full context is the most specific matched environment:

| exact support | radius-1 contexts | radius-2 contexts |
|---|---:|---:|
| `>=2` | 267 types / 1037 pairs | 307 types / 1133 pairs |
| `>=3` | 94 / 583 | 113 / 710 |
| `>=5` | 27 / 265 | 42 / 460 |

Radius-2 is a refinement of radius-1, so adding radius-1 after radius-2 does
not increase the variance-decomposition rank (Section 3.4).

Scaffold and component thresholds are much more permissive than context
thresholds but do not by themselves establish repeatability:

| exact condition | `>=2` types / pairs | `>=3` types / pairs | `>=5` types / pairs |
|---|---:|---:|---:|
| scaffolds | 1392 / 2266 | 173 / 845 | 58 / 516 |
| pair components | 364 / 1258 | 135 / 782 | 49 / 500 |
| contexts and components | 364 / 1258 | 151 / 832 | 50 / 505 |
| contexts and scaffolds | 351 / 1225 | 147 / 817 | 49 / 500 |

The `>=2` row for “contexts and components” equals the component row because
every exact type with at least two pairs automatically has at least two
contexts; the 14 multi-context types with one component are excluded.

### 2.2 Coarse types

There are 1331 coarse types. 1099 have Train-sub support; 651 are singletons.
Coarse support is higher, but a coarse type can contain several pairs in the
same full context:

| coarse support | types | Train pairs | type-context cells | pairs/cell | unique molecules | unique scaffolds | unique components |
|---|---:|---:|---:|---:|---:|---:|---:|
| `>=2` contexts | 351 | 2664 | 1672 | 1.59 | 1401 | 493 | 364 |
| `>=3` contexts | 183 | 2221 | 1336 | 1.66 | 1255 | 439 | 327 |
| `>=5` contexts | 82 | 1764 | 1006 | 1.75 | 1110 | 392 | 285 |

At `>=2`, the raw pair count overstates the number of type-context cells by
about 1.6×; at `>=5`, by about 1.75×. Coarse support is therefore not a
substitute for exact-type support. Strict independence selects 335 coarse
types / 2623 pairs at `>=2`, 181 / 2214 at `>=3`, and 82 / 1764 at `>=5`.

### 2.3 Duplicates, multi-pairs, and dependence

Duplicate checks:

- 4646 unique full pair rows; **0 repeated full pair rows**.
- **0 exact reverse pair rows** (same pair with `a,b` and `b,a` and swapped
  words).
- 3219 unique unordered molecule pairs. 1427 pair rows (30.7%) are additional
  cuts on an already-used molecule pair; the multiplicity histogram is
  1:2135, 2:806, 3:221, 4:49, 5:8. Maximum 5 pair rows per molecule pair.
- 236 unordered molecule pairs appear with both molecule-ID orientations
  (`a<b` and `a>b`). These are not duplicates: the cuts/words differ. The
  canonical direction is the word order, not the molecule-ID order.
- 1865 molecules occur in at least one pair; mean pair incidence 4.98, median
  3, maximum 31 (two molecules occur in 31 pairs).

The matched-pair graph is therefore dense at the molecule level. Pair rows are
not independent observations.

Pair components:

- Train-sub pair graph: 398 components.
- Full Train pair graph (all 4646 pairs): 458 components; 415 components when
  restricted to the Train-sub nodes.
- 60 components contain both Train and Dev nodes.
- 474 cross pairs lie in those 60 components; 326 of 611 Dev–Dev pairs
  (53.4%) lie in them; 874 of 3561 Train-sub pairs (24.5%) lie in them; 160 of
  504 Dev molecules are in them.

Consequently, random pair-level cross-validation would leak heavily across the
Train/Dev boundary. All transfer uncertainty is clustered accordingly.

### 2.4 Evaluation coverage

The evaluated pairs cover only part of the type/context space:

| coverage | cross (474 pairs) | Dev–Dev (611 pairs) |
|---|---:|---:|
| exact type seen in Train-sub | 69 (14.6%) | 243 (39.8%) |
| exact type seen with `>=2` Train-sub contexts | 31 (6.5%) | 154 (25.2%) |
| full context seen | 378 (79.7%) | 107 (17.5%) |
| radius-1 context seen | 471 (99.4%) | 592 (96.9%) |
| radius-2 context seen | 463 (97.7%) | 500 (81.8%) |
| type and full context both seen | 36 | 18 |
| type seen, full context unseen | 33 | 225 |
| type unseen, full context seen | 342 | 89 |
| both unseen | 63 | 279 |

Two different coverage statements must not be conflated:

- **Context identity coverage is high for cross**: 79.7% of cross pairs have a
  full context that occurred somewhere in Train-sub, and over 97% have seen
  radius-1/radius-2 contexts.
- **Type coverage is low for cross**: only 14.6% of cross pairs have the exact
  type, and only 6.5% have the type with at least two Train-sub contexts.
  The dominant cross cell is “type unseen, context seen” (342/474, 72.2%).

For Dev–Dev, full-context coverage is also low (17.5%), so both type and
full-context coverage are limited there.

The transfer evaluation covers 298 of the 504 Dev molecules: 129 in cross
pairs, 259 in Dev–Dev pairs. Conclusions apply to this matched-pair subset, not
to all 504 Dev molecules.

## 3. Task 2 — real substitution responses

### 3.1 Response magnitudes

For the 378 exact types with at least two full contexts (1288 Train-sub pairs).
Per-type means, SDs, deltas, `eta2` values, and the full context lists are in
`analysis/response_variation.json`.

- mean of type means: -0.028;
- mean absolute type mean: 0.537;
- type-mean quantiles (5/25/50/75/95%): -1.265 / -0.437 / 0.000 / 0.416 / 0.990;
- median within-type SD: 0.182; mean within-type SD: 0.235;
- within-type SD quantiles (5/25/50/75/95%): 0.025 / 0.085 / 0.182 / 0.324 / 0.600.

Responses are not centered on a single global shift; individual replacement
types have material positive or negative means, and many have substantial
within-type spread.

### 3.2 Support-class consistency

| exact support class | types | pairs | mean abs type mean | median within-type SD | pair-weighted sign consistency | types all same sign | median `eta2` radius-1 | median `eta2` radius-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 contexts | 225 | 450 | 0.534 | 0.140 | 0.882 | 76.4% | 1.000 | 1.000 |
| 3–4 contexts | 103 | 333 | 0.576 | 0.184 | 0.904 | 70.9% | 1.000 | 1.000 |
| 5–9 contexts | 34 | 216 | 0.449 | 0.288 | 0.884 | 47.1% | 0.926 | 0.999 |
| `>=10` contexts | 16 | 289 | 0.502 | 0.295 | 0.848 | 25.0% | 0.582 | 0.781 |
| all `>=2` | 378 | 1288 | 0.537 | 0.182 | 0.880 | 70.1% | — | — |

The per-type `eta2` at two contexts is effectively binary and is not
interpretable as evidence; the 5–9 and `>=10` columns are the informative ones.
The all-same-sign fraction falls with support because a type with more contexts
has more opportunities to show a reversal, not because the average direction
becomes unstable: the pair-weighted sign consistency stays between 0.85 and
0.90 at every support level.

Across all 378 types, 265 (70.1%) have no sign reversal and 113 (29.9%) have at
least one. The majority direction is stable, but unanimity is not.

### 3.3 Cross-context differences

Pairwise absolute differences between contexts of the same exact type:

| exact support class | context pairs | median | mean | 90th pct | fraction >0.5 | fraction >1.0 |
|---|---:|---:|---:|---:|---:|---:|
| 2 contexts | 225 | 0.280 | 0.406 | 0.886 | 27.6% | 6.7% |
| 3–4 contexts | 381 | 0.260 | 0.395 | 0.970 | 26.0% | 9.4% |
| 5–9 contexts | 601 | 0.270 | 0.426 | 0.920 | 28.3% | 7.8% |
| `>=10` contexts | 3187 | 0.270 | 0.362 | 0.770 | 24.5% | 5.2% |

The difference distribution is remarkably stable across support levels. A
typical pair of contexts differs by about 0.27 in delta units; roughly a
quarter differ by more than 0.5 and 5–9% by more than 1.0. This is material
relative to the mean absolute type mean of 0.537. Context modification is real
even when the majority direction is stable.

Coarse types show the same phenomenon more strongly: median pairwise
differences are 0.57–0.61, with 54–57% above 0.5 and 26–29% above 1.0. This is
expected because coarse types merge distinct exact transformations.

### 3.4 Variance decomposition and context explanation

On the 1288 pairs whose exact type has at least two contexts:

| model | R² | adjusted R² | df | incremental R² over exact type |
|---|---:|---:|---:|---:|
| coarse type | 0.565 | 0.483 | 204 | -0.221 |
| exact type | 0.786 | 0.697 | 377 | 0.000 |
| radius-1 alone | 0.138 | 0.056 | 112 | -0.648 |
| radius-2 alone | 0.275 | 0.099 | 252 | -0.511 |
| exact type + radius-1 | 0.830 | 0.727 | 486 | +0.044 |
| exact type + radius-2 | 0.874 | 0.759 | 616 | +0.088 |
| exact type + radius-1 + radius-2 | 0.874 | 0.759 | 616 | +0.088 |
| exact type + additive full-context dummies | 0.938 | 0.770 | 938 | +0.152 |

The exact-type permutation test gives observed R² 0.786 against a null mean of
0.292 (SD 0.018), p = 0.001. Exact type is the dominant signal. Radius-2 adds
8.8 points of R² over exact type; radius-1 adds 4.4 points. Radius-1 adds
nothing after radius-2 because radius-2 refines it. Additive full-context
dummies add 15.2 points, but this is an in-sample additive model, not a
transferable conditional model.

Pooled within-type ANOVA, which avoids the small-sample per-type `eta2`
problem:

| level | within-type SD | radius-1 explains | radius-2 explains | residual after radius-2 |
|---|---:|---:|---:|---:|
| exact types (1288 pairs) | 0.342 | 71.7% | 86.5% | 13.5% |
| coarse types (2664 pairs) | 0.658 | 39.9% | 51.9% | 48.1% |

Within exact types, the local context identity explains most of the
within-type variance in-sample. Coarse types mix distinct exact transformations
and leave about half of their within-type variance unexplained by the same
local contexts.

The residual after radius-2 is **not** identified as measurement noise. It can
contain higher-order context, type-by-context interaction, or noise. Without
replicates, the audit cannot separate these.

### 3.5 Within-Train-sub predictive check

Leave-one-pair-out (LOMO) and leave-one-component-out type-mean prediction,
with the same shrinkage, evaluated on Train-sub:

| level | pairs / types | LOMO MAE [95% CI] | global MAE | MAE gain | component-excluded MAE [95% CI] |
|---|---:|---:|---:|---:|---:|
| exact `>=2` | 1288 / 378 | 0.350 [0.323, 0.380] | 0.570 | 0.220 | 0.359 [0.331, 0.390] |
| exact `>=3` | 838 / 153 | 0.314 [0.284, 0.346] | 0.567 | 0.253 | 0.320 [0.290, 0.353] |
| exact `>=5` | 505 / 50 | 0.287 [0.254, 0.325] | 0.535 | 0.248 | 0.288 [0.255, 0.326] |
| coarse `>=2` | 2664 / 351 | 0.549 [0.509, 0.592] | 0.618 | 0.069 | 0.570 [0.530, 0.613] |
| coarse `>=3` | 2221 / 183 | 0.530 [0.488, 0.577] | 0.603 | 0.073 | 0.544 [0.501, 0.592] |
| coarse `>=5` | 1764 / 82 | 0.517 [0.473, 0.566] | 0.581 | 0.064 | 0.523 [0.480, 0.573] |

Exact-type effects are clearly predictable within Train-sub: a held-out pair
is predicted much better by the mean of other pairs of the same exact type than
by the global mean, and the gain survives excluding the whole pair component.
Coarse type is a much weaker predictor: the gain is only 0.064–0.073, roughly a
quarter of the exact-type gain. Coarse type is not a reliable substitute for
exact type.

The component-excluded check has 30/1288 (2.3%) fallback-to-global predictions
at `>=2`, 6/838 at `>=3`, and 0/505 at `>=5`; the gain is not driven by those
fallbacks.

### 3.6 Noise floor

No noise floor was estimated:

- 0 duplicate canonical structures among the 3360 official Train molecules;
- 0 repeated `(exact type, full context)` cells; maximum cell multiplicity 1;
- no repeated measurements of the same structure or the same type-context cell.

Within-type across-context variation therefore mixes context modification and
measurement noise. The residual after radius-2 (13.5% within exact types) is
the in-sample context-unexplained fraction; it is not a noise estimate, and
because the context grouping is fitted in-sample it is likely optimistic.

A naive split-half check is reported only as a negative methodological check:
splitting contexts within a type and correlating half-means gives observed
weighted r = 0.852 for exact (74 types) against a permutation null mean of
0.889 (p = 0.858), and 0.655 for coarse (109 types) against a null mean of
0.703 (p = 0.864). The statistic is dominated by between-type differences and
is not diagnostic of within-type stability.

### 3.7 Failure cases

High-support exact types with material instability:

- Exact transform 301 (6 contexts): deltas `[-0.45, -0.02, 3.22, -0.55,
  -0.34, 2.45]`; mean +0.718, SD 1.522, sign consistency 0.667. Most contexts
  are slightly negative, but two contexts are strongly positive.
- Exact transform 40 (3 contexts): deltas `[-2.05, -0.30, +0.92]`; mean
  -0.477, SD 1.219.
- Exact transform 13 (43 contexts, the largest exact type): mean +0.372, SD
  0.284, sign consistency 0.930. The majority direction is stable, but 3 of 43
  contexts are opposite.
- Exact transform 176 (34 contexts): mean -0.469, SD 0.405, sign consistency
  0.853.
- Exact transform 1120 (5 contexts): mean -0.016, SD 0.082, sign consistency
  0.400. The effect is small and the sign is essentially noise at this
  magnitude.

Coarse failure example: coarse type `(85, 85, bond 2)` has 22 pairs over 9
contexts, mean -0.242, SD 1.601, with deltas including +2.94 and -3.01. The
coarse label merges exact transformations with opposite responses.

## 4. Task 3 — transfer to Dev

### 4.1 Overall transfer

Reference A is the shrunk Train-sub type mean (global mean for unseen types).
Reference B is A plus a ridge correction on fixed radius-1/radius-2 context
descriptors (634 raw features, 583 used; `lambda=10`; Train-sub only). Per-pair
predictions and the de-duplicated failure cases are in
`analysis/transfer_check.json`.

| scope | pairs (clusters) | global MAE | A MAE [CI] | B MAE [CI] | A corr | B corr |
|---|---:|---:|---:|---:|---:|---:|
| cross | 474 (129 Dev molecules) | 0.772 | **0.740** [0.643, 0.847] | 0.830 [0.728, 0.931] | 0.264 | 0.110 |
| Dev–Dev | 611 (72 components) | 0.763 | **0.680** [0.529, 0.843] | 0.861 [0.700, 1.029] | 0.394 | 0.110 |
| combined | 1085 (103 components) | 0.767 | **0.706** [0.613, 0.815] | 0.847 [0.738, 0.964] | 0.348 | 0.120 |

Paired comparisons:

| scope | A − global ΔMAE [CI] | B − A ΔMAE [CI] |
|---|---:|---:|
| cross | -0.032 [-0.049, -0.016] | +0.089 [+0.048, +0.137] |
| Dev–Dev | -0.082 [-0.111, -0.055] | +0.181 [+0.117, +0.263] |
| combined | -0.060 [-0.080, -0.043] | +0.141 [+0.095, +0.193] |

A is significantly better than the global mean overall, but the gain is small
because it comes only from seen types. B is significantly worse than A on both
scopes. The direction is consistent across all three scopes.

### 4.2 Seen versus unseen types

| scope / subset | pairs (clusters) | global MAE | A MAE | B MAE | A − global [CI] | B − A [CI] |
|---|---:|---:|---:|---:|---:|---:|
| cross, seen type | 69 (38) | 0.647 | 0.429 | 0.640 | -0.218 [-0.323, -0.122] | +0.211 [+0.115, +0.323] |
| cross, unseen type | 405 (117) | 0.793 | 0.793 | 0.862 | 0.000 | +0.069 [+0.026, +0.117] |
| cross, seen type `>=2` contexts | 31 (21) | 0.611 | 0.399 | 0.559 | -0.212 [-0.383, -0.051] | +0.160 [-0.005, +0.344] |
| cross, seen full context | 378 (81) | 0.773 | 0.753 | 0.798 | -0.020 [-0.035, -0.005] | +0.045 [+0.002, +0.090] |
| cross, unseen full context | 96 (67) | 0.769 | 0.690 | 0.954 | — | +0.265 [+0.171, +0.357] |
| Dev–Dev, seen type | 243 (54) | 0.740 | 0.533 | 0.759 | -0.207 [-0.265, -0.147] | +0.226 [+0.133, +0.340] |
| Dev–Dev, unseen type | 368 (45) | 0.777 | 0.777 | 0.929 | 0.000 | +0.151 [+0.087, +0.237] |
| Dev–Dev, seen type `>=2` contexts | 154 (48) | 0.680 | 0.465 | 0.722 | -0.214 [-0.299, -0.142] | +0.257 [+0.149, +0.384] |
| Dev–Dev, seen full context | 107 (17) | 0.499 | 0.475 | 0.585 | -0.024 [-0.058, +0.014] | +0.110 [+0.039, +0.305] |
| Dev–Dev, unseen full context | 504 (66) | 0.819 | 0.724 | 0.920 | — | +0.196 [+0.120, +0.297] |

The A gain is concentrated in seen types. On unseen types A is exactly the
global mean by construction, so the overall gain is the coverage-weighted
seen-type gain. The cross seen-type gain (-0.218) is large relative to the
overall gain (-0.032) because only 14.6% of cross pairs are seen.

The “seen full context / unseen full context” split is not a type split:
some pairs with an unseen full context still have a seen type, and vice versa.
A improves on unseen full-context pairs because some of their types are seen; B
degrades most on those same pairs, indicating that the context correction is
not aligned with the type signal.

### 4.3 Support bins

| scope | Train-sub pairs of the type | pairs (clusters) | global MAE | A MAE |
|---|---:|---:|---:|---:|
| cross | unseen | 405 (117) | 0.793 | 0.793 |
| cross | 1 | 38 (24) | 0.676 | 0.453 |
| cross | 2 | 14 (8) | 0.597 | 0.409 |
| cross | 3–4 | 12 (12) | 0.646 | 0.374 |
| cross | 5–9 | 5 (5) | 0.568 | 0.431 |
| Dev–Dev | unseen | 368 (45) | 0.777 | 0.777 |
| Dev–Dev | 1 | 89 (27) | 0.844 | 0.650 |
| Dev–Dev | 2 | 46 (18) | 0.703 | 0.472 |
| Dev–Dev | 3–4 | 28 (18) | 0.415 | 0.328 |
| Dev–Dev | 5–9 | 32 (20) | 0.757 | 0.564 |
| Dev–Dev | `>=10` | 48 (31) | 0.760 | 0.472 |

Two points are important:

- Singleton types already help A substantially. The transfer gain does not
  require independent repeats; it requires the type to be seen at all, and
  shrinkage makes even a one-pair type mean useful.
- The high-support cross bin is tiny (5 pairs, 5 clusters), so its MAE is
  fragile. The Dev–Dev `>=10` bin is more substantial (48 pairs, 31 clusters)
  and still shows a clear A gain.

This is why type coverage and repeatability must be separated: A benefits from
broad type coverage, while a conditional model needs repeated contexts.

### 4.4 The conditional reference B

B is worse than A everywhere. The failure is not a small regularization issue:

| variant | cross MAE | B − A [CI] | Dev–Dev MAE | B − A [CI] |
|---|---:|---:|---:|---:|
| A | 0.740 | 0 | 0.680 | 0 |
| B full (compact + WL, `lambda=10`) | 0.830 | +0.089 [+0.048, +0.137] | 0.861 | +0.181 [+0.117, +0.263] |
| B compact only (81 used features) | 0.766 | +0.026 [+0.007, +0.048] | 0.692 | +0.012 [-0.004, +0.033] |
| B full, `lambda=100` | 0.821 | +0.081 [+0.041, +0.126] | 0.803 | +0.122 [+0.080, +0.175] |
| B full, self-included Train offset | 0.768 | +0.028 [+0.005, +0.052] | 0.761 | +0.081 [+0.048, +0.127] |

The compact-only and `lambda=100` and self-included variants are post-hoc
robustness checks, not model-selection steps. None is better than A. Compact B
is slightly but significantly worse on cross and statistically indistinguishable
from A on Dev–Dev; the full fingerprint block is clearly harmful.

In-sample diagnostics show why:

- A leave-one-out Train MAE: 0.592; A self-included Train MAE: 0.330.
- B full Train MAE: 0.564; residual weighted R² 0.354.
- B compact Train MAE: 0.607; residual weighted R² 0.088.
- B full has 583 used features and 3561 training pairs from 398 components,
  with component-balanced pair weights; it can fit the training residuals but
  the fit does not transfer.

Calibration:

| scope | A slope [CI] | A R² | A seen-type slope [CI] | A seen R² | B slope [CI] | B R² |
|---|---:|---:|---:|---:|---:|---:|
| cross | 1.49 [1.08, 1.79] | 0.069 | 1.48 [1.08, 1.76] | 0.602 | 0.22 [0.03, 0.44] | 0.012 |
| Dev–Dev | 1.54 [1.13, 1.91] | 0.155 | 1.55 [1.09, 1.98] | 0.402 | 0.17 [0.05, 0.36] | 0.012 |

A is underdispersed (slope > 1, prediction/truth SD ratio 0.18–0.26 overall and
0.41–0.52 on seen types): the type means are directionally informative but
shrunk. B is compressed in slope (0.17–0.22) while adding prediction variance;
its corrections are not aligned with the true residual. A post-hoc calibration
of A would be a new model and is not part of this audit.

### 4.5 Coarse-type sensitivity

Repeating A and B with coarse types instead of exact types:

| scope | global MAE | coarse A MAE | coarse B MAE | coarse A seen | coarse A unseen |
|---|---:|---:|---:|---:|---:|
| cross | 0.772 | 0.778 | 0.876 | 0.729 (302 pairs) | 0.864 (172 pairs) |
| Dev–Dev | 0.763 | 0.705 | 0.877 | 0.713 (470 pairs) | 0.678 (141 pairs) |

Coarse A has much broader coverage (302/474 cross pairs seen, versus 69 for
exact), but it is slightly **worse** than the global mean on cross (0.778 vs
0.772) and only modestly better on Dev–Dev (0.705 vs 0.763). Coarse B is worse
than coarse A on both scopes. Increasing type coverage by coarsening destroys
the exact-type signal; it is not a route to a better conditional model.

### 4.6 Transfer failure cases

De-duplicated by unordered molecule pair:

- **Coverage failure (unseen type).** Pair row 27, true delta +3.67, A -0.032,
  error 3.70. The exact type is unseen; the full context is seen. A cannot
  produce an extreme response without the type.
- **Seen-type sign failure.** Pair row 5483, true delta -1.70, A +0.649, error
  2.35. The type is seen with two Train-sub contexts, but this full context is
  unseen; the type mean has the wrong sign for this context.
- **B overcorrection on a seen type.** Pair row 525, true delta +1.16, A +0.884
  (error 0.276), B +3.231 (error 2.071). The type is seen with three
  contexts; the ridge correction moves the prediction far in the wrong
  direction.
- **B overcorrection on an unseen type.** Pair row 4202, true delta -1.30,
  A -0.032, B +1.808. The type is unseen; the context correction invents a
  large positive response.

These are representative: the worst A errors are dominated by unseen types with
extreme deltas, while the worst B degradations are dominated by large,
poorly aligned context corrections.

## 5. Task 4 — conclusions

### 5.1 How many types have truly independent cross-environment repeats?

Using distinct full contexts as “cross-environment” repeats:

- **378** exact types have at least two distinct full contexts (1288 pairs).
- **344** of them (1211 pairs) also span at least two pair components and at
  least two scaffolds; these satisfy the strict independence condition.
- **153** types (838 pairs) have at least three contexts; **147** (817 pairs)
  are strict.
- **50** types (505 pairs) have at least five contexts; **49** (500 pairs) are
  strict.

So the strict counts are 344 types at `>=2` and 49 types at `>=5`. These are
small fractions of the 2651 supported exact types (13.0% and 1.8%). The raw
pair counts at these levels (1288 and 505) should not be read as independent
sample sizes: the `>=2` pairs span 999 molecules, 305 scaffolds, and 275
components; the `>=5` pairs span 583 molecules, 154 scaffolds, and 174
components.

Coarse types have broader nominal support (351 types / 2664 pairs at `>=2`),
but only 1672 type-context cells, and the coarse label merges distinct exact
transformations. They are not “truly independent” replacements in the exact
sense.

### 5.2 Do substitutions have stable base effects?

**Yes in the majority-direction sense, but not as constant effects.**

- Pair-weighted sign consistency is 0.880 at `>=2` contexts and remains
  0.848–0.904 across support classes.
- The mean absolute type mean is 0.537, so the average effect is material.
- Leave-one-out exact-type prediction has MAE 0.350 against a global baseline
  of 0.570 (gain 0.220); the leave-component-out check gives 0.359, so the
  gain is not a same-component artifact.
- The exact-type permutation test is far above its null (R² 0.786 vs null mean
  0.292, p = 0.001).

But stability is not unanimity:

- 113 of 378 types (29.9%) show at least one sign reversal.
- The all-same-sign fraction falls from 76.4% at two contexts to 25.0% at
  `>=10`; with more contexts, there are more opportunities for a reversal.
- Median within-type SD rises from 0.140 at two contexts to 0.295 at `>=10`,
  and the median absolute context-to-context difference is about 0.27.
- High-support counterexamples exist: exact transform 301 has six contexts
  with deltas from -0.55 to +3.22; coarse type `(85,85,2)` mixes +2.94 and
  -3.01.

The defensible statement is: **exact replacement types have stable average
directions and material average magnitudes, with real context-dependent
heterogeneity; they are not constant effects and a minority of types reverse.**

### 5.3 Does context add measurable transferable information?

**Measurable in-sample, not transferable out-of-split at this support.**

In-sample, context identity is strongly associated with within-type variation:

- Radius-2 explains 86.5% of within-exact-type variance; radius-1 explains
  71.7%; the residual after radius-2 is 13.5%.
- Adding radius-2 to exact type raises R² from 0.786 to 0.874 (in-sample).

Out-of-split, the fixed context descriptors do not improve on A:

- B full is worse than A on cross (+0.089 [0.048, 0.137]) and Dev–Dev
  (+0.181 [0.117, 0.263]).
- Compact-only B is +0.026 [0.007, 0.048] on cross and +0.012
  [-0.004, 0.033] on Dev–Dev; it does not improve.
- `lambda=100` and the self-included-offset variant are also worse.
- B's calibration slope is 0.17–0.22, versus 1.49–1.54 for A; its correction
  variance is not aligned with the true residual.

The correct conclusion is not “context is irrelevant.” Context identity
explains most within-type variance. The conclusion is that **the available
fixed descriptors and the available independent type-context supervision do
not learn a context effect that transfers to Dev.** The failure is in
estimation/transfer, not in the existence of context-associated variation.

### 5.4 Which structural support conditions are reliable?

- **Minimum for a repeatable type mean:** at least two distinct full contexts
  in Train-sub. One pair gives a type identity but no internal replication.
- **Better for a stable magnitude/direction:** at least five contexts, and
  ideally the strict condition (at least two components and two scaffolds).
  At `>=5`, 49 of 50 exact types are strict; 20 of 50 are all-same-sign,
  and the pair-weighted sign consistency is 0.863.
- **Singleton types:** still useful for the transformation-only reference A
  because shrinkage controls their noise; they are not sufficient for a
  conditional model.
- **Coarse types:** not a reliable substitute. Coarse leave-one-out MAE gain is
  only 0.064–0.073 (versus 0.220–0.253 for exact), and coarse A is worse than
  the global mean on cross despite much broader coverage.

### 5.5 Is the data sufficient for a conditional structural-response model?

**No, not at this split, descriptor set, and support.**

A conditional model would need to learn how the local context modifies the
exact-type response and transfer that function to Dev. The audit finds:

- exact-type coverage on cross is 14.6%; only 6.5% of cross pairs have a seen
  type with at least two contexts; and only 36 cross pairs (7.6%) have both the
  type and the full context seen, a different condition that does not require
  two contexts;
- the strict independent support is 344 types at `>=2` contexts and 49 types at
  `>=5`;
- B fails even on the seen-type `>=2` subset: cross +0.160
  [-0.005, 0.344], Dev–Dev +0.257 [0.149, 0.384];
- compact-only B, `lambda=100`, and the self-included variant do not rescue it;
- coarse-type expansion does not help.

The audit therefore does not support fitting a conditional structural-response
model on this directory and split.

### 5.6 If not, is the gap type coverage, context coverage, or supervision strength?

All three are limited, but they are not equally binding:

1. **Exact-type coverage is the first binding constraint.** Only 69/474 cross
   pairs (14.6%) have a seen exact type. A's gain is concentrated there
   (0.429 vs 0.647 global), but the overall gain is only 0.032 because the other
   85.4% of cross pairs fall back to the global mean. Coarsening raises coverage
   to 302/474 but destroys the signal (coarse A 0.778 vs global 0.772).
2. **Independent type-context supervision is the second binding constraint.**
   Even the 31 cross pairs with a seen type and `>=2` contexts are too few for B;
   B degrades there (+0.160, CI includes zero but the point is positive), and on
   the larger Dev–Dev subset (154 pairs) it degrades significantly (+0.257).
   The 1288 exact pairs at `>=2` contexts are only 275 components and 305
   scaffolds, and only 505 pairs / 174 components are at `>=5`.
3. **Context identity coverage is comparatively adequate for cross, but not
   for Dev–Dev.** Cross: 79.7% full-context seen, 99.4% radius-1, 97.7%
   radius-2. Dev–Dev: 17.5% full-context, 96.9% radius-1, 81.8% radius-2.
   The dominant cross cell is “type unseen, context seen” (342/474), so seeing
   the context without the exact type does not help A and is not enough for B.

In short: the gap is **primarily exact-type coverage and independent
type-context supervision**, not a lack of context identities. The data have
many contexts but too few exact types paired with enough independent repeats of
those contexts, and the fixed descriptors do not bridge that gap.

## 6. Engineering checks and reproducibility

The runner asserts input hashes, split sizes, 4646 analyzed pairs with scopes
3561/474/611, transform consistency, zero environment-key mismatches, finite
features, Train-sub-only vocabulary, label masking, response/support count
agreement, transfer subset coverage, and the frozen bootstrap settings. Outputs:

| file | bytes | sha256 |
|---|---:|---|
| `analysis/support.json` | 2,149,724 | `b154ac0ba392c47af08ed0d0456ce2bcc82cafa75c46e53e95665509d874dd6c` |
| `analysis/response_variation.json` | 1,178,117 | `738e755799bf478be93186249b05cbdc1910f8c88f7f83408d37bb7bc6e295e8` |
| `analysis/transfer_check.json` | 203,928 | `b31ee96792315862d3012b2571092cbe0c1eef262a4bfccb5d0052ec77a31cc0` |

The run used 79.1 s and 649 MB peak RSS. The hashes identify this archived
run; rerunning updates `generated_utc` and therefore the file hashes, while the
numeric content is deterministic. Reproduce with:

```bash
cd tracks/structural-sufficiency/structural_response_learnability_audit_v1
uv run --project /home/calendar/code/research python code/run_audit.py
```

## 7. Limitations

- Only matched pairs are evaluated; 298 of 504 Dev molecules occur in an
  evaluation pair. The conclusions do not cover the 206 Dev molecules without a
  matched pair.
- Metrics are delta-level, not whole-molecule RMSE, and are not comparable to
  the parent rounds' whole-molecule tables.
- The audit has no repeated structures or repeated type-context cells, so it
  cannot estimate a measurement-noise floor.
- B uses one fixed descriptor family and one fixed `lambda`; the compact,
  `lambda=100`, and self-included-offset variants are robustness checks, not a
  search. A different descriptor family could in principle transfer better, but
  that is outside this audit.
- The cross and Dev–Dev scopes have different mean deltas from Train-sub
  (-0.141, +0.088, -0.032), so global-mean baselines are biased; this is a
  property of the split, not a model choice.
- Some support bins have few clusters (e.g., 5 cross pairs in the 5–9 bin);
  their intervals are correspondingly fragile. The JSON records cluster counts
  and a `ci_reliable` flag.
- The audit is associational. The word/context identities are graph codes; no
  chemical mechanism, electron effect, or causal direction is claimed.

## 8. Bottom line

The directory contains a real, repeatable exact-type signal: exact replacement
types predict held-out Train-sub responses far better than a global mean, and
the majority direction is stable. But the independent support is thin (344
strict multi-context types; 49 at `>=5`), the responses have material
context-dependent heterogeneity, and the exact-type coverage of cross pairs is
only 14.6%. The transformation-only reference A transfers to seen types but
barely moves the overall cross MAE because most cross types are unseen. The
context-conditioned reference B does not transfer; its in-sample context
signal is not captured by the fixed descriptors in a way that generalizes.

The data are therefore **not sufficient for a conditional structural-response
model** at this split and support. The primary gaps are exact-type coverage and
independent type-context supervision; context identity coverage is comparatively
adequate for cross but is not paired with enough repeated exact types. No
chemical-causal conclusion is drawn.