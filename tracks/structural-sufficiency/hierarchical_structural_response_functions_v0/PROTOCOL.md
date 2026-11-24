# Hierarchical Structural Response Functions v0 — PROTOCOL

**Round:** `hierarchical_structural_response_functions_v0` (HSRF v0)
**Version:** 2026-11-24/v1 — frozen before any formal GPU run
**Task:** `ogbg-mollipo` whole-molecule lipophilicity regression, official scaffold split
**Primary metric:** RMSE in original logD units, model/checkpoint/hyper-parameter selected on official Valid only
**Secondary metric:** MAE
**Training seed:** 0 (single seed, no additional seeds)
**Test:** read from the beginning and reported as `DEVELOPMENT_TEST`; never used for selection
**Budget:** target ≤ 2 GPU·h, soft cap 4 GPU·h; CPU-first preparation

This round tests three pre-registered questions:

- **Q1 (representation):** can sufficiently covering, multi-granularity explicit structure
  counts support competitive graph prediction without message passing?
- **Q2 (nonlinear response):** does a fixed hinge expansion of those counts improve
  prediction over a purely linear structural response while keeping an exact additive ledger?
- **Q3 (real local relations):** do real adjacent-root co-occurrences add predictive value
  beyond the matched structural marginals?

No GNN, Graph Transformer, iterative message passing, LCRC/R2 hidden representation,
pretrained encoder, teacher, distillation, or unaccounted black-box residual is used in
any HSRF candidate. MLP and KRR are independent same-input prediction references and are
never mixed into the interpretable candidates.

## 1. Read-only reuse and data discipline

Pinned read-only sources (sha256 re-checked at runtime):

| source | sha256 | role |
|---|---|---|
| `tracks/structural-query-learning/code/sq_data.py` | `89ad5436be3ddb8825fe877535e7fbb13679f4cb012a3a3073d63deaf9a83647` | official OGB cache loader and split indices |
| `tracks/structural-query-learning/code/sq_common.py` | `fe7333978fd748cf2a61fcdb4d3fa73bf855ebbdc6b05c439477b2c921b9d8cb` | repo plumbing |
| `rooted_interaction_cumulant_encoder_pilot_v0/code/rice_structure.py` | `cb044e37e3c2bfd90881fca196b3e40fa7e05429bc389747da3957a56269629b` | rooted connected induced subgraph enumeration + exact canonical certificate |

Frozen data asset: `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`
sha256 `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c`.
42 000-row identity is checked item by item (row id, split membership, label). The label
vector is used for all three official splits because Test is authorised as
`DEVELOPMENT_TEST`; every label-derived statistic used for fitting (target mean/std,
all knots, all scales, every vocabulary decision) is computed on the official Train rows
only. No Valid/Test label is ever used for checkpoint, λ, bandwidth, vocabulary, knot or
model selection.

## 2. Stage 0 — multi-granularity structure library

For every molecule and every root atom `v`, enumerate connected induced subgraphs `S`
with `v ∈ S`, `|S| ∈ {1,2,3,4}`, `S ⊆` the radius-2 ego of `v`, and every real internal
bond retained. Each `(v,S)` is stored once. A physical subgraph reachable from several
roots is several rooted instances by definition. Supports are canonicalised as
root-marked attributed graphs with the root pinned to slot 0
(`rice_structure.canonical_signature`).

Four signatures per instance:

- **TOPO:** unlabelled topology + root mark (all atom/bond labels equal).
- **ELEMENT:** atomic number; bond type.
- **COARSE:** atomic number, formal charge, is-aromatic; bond type.
- **FULL:** all 9 OGB atom columns and all 3 OGB bond columns.

Only real OGB fields are used; no RDKit descriptor or external chemistry property. The
FULL→COARSE→ELEMENT→TOPO type mapping is saved.

**Vocabulary (Train only).** For each `(size, granularity)` group: keep types present in
at least 8 Train molecules; drop count columns that are constant on Train; sort by Train
document frequency descending and canonical signature ascending; cap at 1024 types.
TOPO keeps every qualified type. No label, residual, gradient, SHAP or prediction score
participates in this filter.

**X_BASE.** For each retained type `t`, `n_t(G)` = number of rooted occurrences. Add the
four named graph counts (atom count, physical bond count, connected components, cycle rank
= bonds − atoms + components). Add the bond-type slot counts `M_b` and endpoint marginals
`A_tb` required by the relation construction, so every arm shares them. Counts stay raw
and non-negative; no hashing or collision compression.

## 3. Stage 0 — real local relations and matched NULL

**Anchors.** From retained COARSE structure types: top 32 size-2 and top 32 size-3 types
by Train document frequency (same tie-breaking). `I_t(v)=1` iff root `v` has at least one
instance of type `t` (binary presence, never an occurrence product).

**Real relations.** For every physical bond `(u,v)` create directed slots `(u,v)` and
`(v,u)`, grouped by the full OGB bond type `b`. For canonical `t ≤ s`,

`R_tsb(G) = Σ_{(u,v)∈E_b^directed} I_t(u) I_s(v)`.

Because both directions exist, `R_tsb = R_stb`; only `t ≤ s` is stored. When `t=s` both
directed slots may hit; this is a directed-slot count, not a physical-bond count. Overlap
of the two anchor structures is allowed. A relation type is retained when it occurs in at
least 16 Train molecules, its Train count is non-constant, and it is within the top 1024
by (Train document frequency descending, canonical key ascending).

For every retained relation the actual molecule row, physical bond, directed slot,
anchor type ids and the hit anchor structure instances are saved.

**Matched marginals.** `M_b` = number of directed slots of bond type `b`;
`A_tb` = directed slots of type `b` whose source has anchor `t`; `B_sb` = directed slots of
type `b` whose target has anchor `s`. The two-direction construction gives `A_tb = B_tb`
per molecule; one column is kept and the identity is asserted.

**NULL.** Within each bond type `b` the target-end presence is uniformly permuted across
slots, ignoring real source–target correspondence. Then
`K_tsb ~ Hypergeometric(M_b, B_sb, A_tb)` and `E[K]=A_tb B_sb/M_b` with `M_b=0 → 0`.
For every REAL basis function `h` the NULL input is the exact hypergeometric expectation
`E[h(K)]` computed from a stable survival-function sum; `h(E[K])` is never substituted for
a nonlinear `h`. NULL changes no molecule-level structural count and is not a new graph.

**Pre-registered relation-support gate.** The relation axis is declared
`RELATION_SUPPORT_INSUFFICIENT` if any of: (a) fewer than 50% of Train molecules have at
least one retained relation hit, (b) fewer than 10% of Train molecules have at least one
REAL ≠ NULL relation column, (c) the median number of nonzero retained relation columns per
Train molecule is below 1, or (d) the relation identity checks fail. Otherwise the support
is sufficient and the relation axis is judged only by the predictive criteria below.

## 4. Stage 1 — response functions

For every X_BASE count column and every retained relation column use the fixed hinge basis

`h_0(n) = n`, `h_k(n) = max(n − k, 0)`,

with candidate knots `{1, round(median of Train positive counts)), round(90th percentile of
Train positive counts)}` after de-duplication. A hinge is kept only if at least 8 Train
molecules lie on each side of the knot (`n ≤ k` and `n > k`) and the hinge column is
non-constant on Train. Every function outputs 0 at input 0. Knots and basis are Train-only.
Relation knots are reused verbatim by NULL; NULL never discovers new types or knots.

Scaling: `scale_j = max(sqrt(mean_train(h_j²)), 0.1)` and the design uses `h_j/scale_j`.
NULL's main comparison uses the REAL scales; a separate low-cost F_NULL self-scale solve
uses NULL's own Train scales and does not participate in model selection.

Five convex arms (all with unpenalised intercept):

| arm | base columns | relation columns |
|---|---|---|
| `L_BASE` | linear `h_0` only | — |
| `F_BASE` | full hinge basis | — |
| `L_REAL` | linear `h_0` only | REAL linear `h_0` |
| `F_NULL` | full hinge basis | NULL expected basis of the REAL basis |
| `F_REAL` | full hinge basis | REAL full basis |

Solve `mean((ẑ−z)²) + λ‖β‖²` in FP64 by a stable dual/gram eigendecomposition with a
centred design and an unpenalised intercept. Candidate λ:
`[1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10]`.
Every λ is evaluated on all three splits. Each arm selects the minimum Valid RMSE; ties go
to the stronger λ. No L1 or top-K sparsity budget is used.

## 5. Stage 1 — independent prediction references

**MLP.** `D → 128 → 64 → 1`, SiLU, dropout 0.1, linear head. AdamW, lr 1e-3, min lr 1e-5,
cosine, weight decay 1e-3, gradient clip 5.0, batch 128, seed 0, 180 epochs run in full,
MSE on the Train-standardised label, full Train/Valid/Test evaluation every epoch,
checkpoint by lowest Valid RMSE (earliest epoch on ties). `MLP_BASE`, `MLP_NULL`, `MLP_REAL`
consume exactly the `F_BASE`, `F_NULL`, `F_REAL` matrices. `MLP_REAL` and `MLP_NULL` use
identical architecture, input width, column order, initialisation seed and batch order.
No width/depth/dropout/LR search.

**KRR.** RBF kernel on the normalised `F_BASE` and `F_REAL` matrices, σ = median Train
non-zero pairwise Euclidean distance × `{0.5, 1, 2}`, λ ∈ `{1e-4, 1e-3, 1e-2, 0.1, 1}`,
centred kernel with unpenalised intercept, all Train/Valid/Test metrics per cell, selection
by Valid RMSE (ties to stronger λ, then larger σ). One eigendecomposition per bandwidth.

## 6. Pre-registered decision rules

All gains are `RMSE(reference) − RMSE(candidate)`; positive means the candidate is better.

**Representation axis (10.1).** Black-box reference = better of the same-input MLP/KRR by
Valid RMSE for BASE and for REAL. If both black-box Valid RMSEs are more than 0.06 above
the frozen historical GINE Valid coordinate → `REPRESENTATION_OR_LEARNER_LIMITED`. If some
black-box reference is within 0.03 of the historical GINE coordinate →
`COMPETITIVE_INFORMATION_SIGNAL`. Otherwise `PARTIAL_INFORMATION_SIGNAL`.

**Nonlinear axis (10.2).** If `F_BASE−L_BASE` or `F_REAL−L_REAL` has Valid gain ≥ 0.02,
Test gain ≥ 0, and the Valid paired bootstrap 95% CI lower bound > 0 →
`NONLINEAR_RESPONSE_SIGNAL`. If the point estimates satisfy the two gains but the CI
crosses 0 → `NONLINEAR_RESPONSE_SCREEN_ONLY`. If no material gain →
`NO_MATERIAL_NONLINEAR_GAIN`. A clear opposite Valid/Test direction is separately marked
`SPLIT_SENSITIVE`.

**Relation axis (10.3).** If `F_REAL−F_BASE` and `F_REAL−F_NULL` both have Valid gain
≥ 0.02, Test gain ≥ 0, Valid CI lower bound > 0, relation support is sufficient, and the
exported relation contribution has non-zero variation → `REAL_RELATION_SIGNAL`. If only
`MLP_REAL` beats both `MLP_BASE` and `MLP_NULL` while `F_REAL` has no corresponding
increment → `RELATION_INFORMATION_NOT_EXPLOITED_BY_RESPONSE_FUNCTIONS`. If F_REAL beats
F_NULL but not F_BASE → `RELATION_CONTROL_ONLY`. Otherwise `NO_MATERIAL_RELATION_GAIN`.

**Decomposition axis (10.4).** For F_BASE/F_REAL against their best same-input black box:
both Valid and Test gap ≤ 0.03 and the contribution ledger passes → `LOW_DECOMPOSITION_COST`.
If the black box is better by a Valid gap ≥ 0.05 and the paired result supports the
difference → `DECOMPOSITION_COST_SIGNAL`. Otherwise `UNRESOLVED_DECOMPOSITION_COST`.

**Overall (10.5).** The round candidate is the better of F_BASE/F_REAL by Valid RMSE.
`PROMISING_INTERPRETABLE_PREDICTOR` requires Valid ≤ GINE Valid + 0.03, Test ≤ GINE Test
+ 0.03, `LOW_DECOMPOSITION_COST`, and an exact contribution reconstruction. Add
`ATTRIBUTION_NOT_IDENTIFIED` if near-optimal λ solutions differ materially in contributions.
Add `PARTIAL_PROGRESS` if the candidate beats the historical LCRC coordinate but is still
clearly behind strong GINE. Add `INTERPRETABLE_FUNCTION_CLASS_LIMITED` if the same-input
black box is competitive while the explicit decomposition clearly lags. Add
`REPRESENTATION_OR_LEARNER_LIMITED` if both the black box and the explicit function clearly
lag. Axes are reported independently.

**Bootstrap.** 2000 paired resamples, seed 1703. Bemis–Murcko scaffold clusters are
computed for Valid/Test with RDKit; if every molecule is its own cluster this equals a
molecule-row paired bootstrap, which is stated in the report. Fixed-state sample
uncertainty only; no cross-seed claim.

## 7. Engineering and positive-control gates

Before formal results: official split/label/row checks; Train-only scaler round trip and
NaN/Inf scan; vocabulary/scale/knot Train-only checks; artificial path/star/triangle/ring
enumeration vs brute force; one-record-per-`(root,support)`; internal edges complete;
relabelling invariance (≥16 Train molecules); four-granularity mapping; REAL/NULL bounds
and identities; hypergeometric expectation vs explicit enumeration; exact ridge solve vs
direct solve; exported-function replay; reload/single-vs-batch MLP replay; contribution
ledger max absolute error target ≤ 1e-7 logD.

Synthetic positive controls on real Lipo graphs, no real logD used to choose planted terms:
Synthetic-L (linear base), Synthetic-F (linear + hinge), Synthetic-R (linear + hinge +
real relation). A small full-rank design with planted terms plus independent distractors
must recover predictions and coefficients at λ=1e-6; a model without relations must not
recover Synthetic-R. The full-library solve then separates prediction recovery from
coefficient recovery under collinearity. Synthetic results are not chemical evidence.

## 8. Stage 2 — contribution accounting and identifiability

For F_BASE/F_REAL best-Valid solutions (with L_BASE/L_REAL as reference) export per
molecule: intercept, every rooted structure main effect, every relation effect, sums,
reconstruction and error. Rooted structures, whole-graph statistics and relation features
are reported as separate groups; overlapping rooted structures are never described as
mutually exclusive atom energies.

Response curves on Train-supported integer counts for the top 20 main and top 20 relation
mean-|contribution| features, with Train/Valid/Test support, contribution distribution and
out-of-Train-support flags. Nonlinearity diagnostics report hinge RMS, per-group nonlinear
mass, Valid/Test distribution, and the fixed-weight hinge-zeroing prediction/RMSE change
(clearly distinguished from refitting the linear model class).

Identifiability audit: Train design dimension, numerical rank, exact duplicate columns,
parent-child count identities, major collinear groups; among λ with Valid RMSE within 0.01
of the arm best, compare predictions, contribution rankings (Spearman), top-20 overlap,
sign agreement and function curves; mark `attribution_ambiguous` groups. If only one λ is
near-optimal, report that stability could not be assessed. A null-space projection of
coefficient differences is an optional diagnostic and never blocks the main results.

Matched-pair diagnostic reuses `task_response_composition_s0_v1/data/trc_cache_lipo.npz`
(existing directory, no regeneration). For aligned Valid/Test/cross-split pairs report
ΔMAE/RMSE, direction accuracy, main-vs-relation decomposition, seen/unseen transforms and
counts. If alignment fails → `PAIR_DIAGNOSTIC_UNAVAILABLE`; the directory is never rebuilt.

## 9. Execution

1. Stage 0 and all CPU checks locally; remote Stage 0 on `res` for the GPU jobs.
2. GPU0 `MLP_REAL`, GPU1 `MLP_NULL`, then the first free GPU runs `MLP_BASE`.
3. Five ridge arms and KRR on CPU/local; analyses as soon as predictions exist.
4. No new seed, dataset, architecture family, vocabulary size, knot rule or post-hoc
   branch after seeing intermediate results. Engineering fixes that do not change the
   scientific definitions are recorded with before/after evidence.