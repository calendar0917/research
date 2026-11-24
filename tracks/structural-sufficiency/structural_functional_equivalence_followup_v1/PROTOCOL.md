# SFEC v1 — Conditional Structural Functional Composition Confirmation

Round: `structural_functional_equivalence_followup_v1`  
Track: `tracks/structural-sufficiency/`  
Parent: `structural_functional_equivalence_v0/` (read-only)  
Status: **frozen before any label analysis in this round**. It records the
canonical pair orientation, metric definitions, estimators, masking rules,
transfer subsets, candidate rules and judgement rules. It is not changed after
seeing label results.

This round only implements corrections, numerical re-analysis, statistical
validation and reporting for the parent experiment. No new neural network, no
new representation, no literature survey. CPU only. All caches produced by the
parent round are reused read-only. The full PCQM4Mv2 Train set is never
re-scanned and the Train prefix is never enlarged. ZINC's `INSUFFICIENT_CROSSED_SUPPORT`
result is retained and is not re-run.

## 0. Fixed inputs and frozen objects

| item | choice |
|---|---|
| parent artifacts | `../structural_functional_equivalence_v0/data/cells_pcqm_train_1m.*`, `.valid.*`, `.zinc.*` (read-only) |
| labels | parent cache plus the official PCQM CSV `homolumogap` in eV; official Test labels are never read |
| primary S | exactly the 64 S types frozen by the parent (`universe.npz::primary/rank`); no re-selection |
| context split | exactly the parent masks `discovery_mask`, `primary_evaluation_cells`, derived from `h(C)=SHA256("0|"+C_key)[:16]%100`, discovery `<80`, salt 0 |
| identity isolation | exactly the parent rule: any molecule appearing in an evaluation context is removed from discovery; primary evaluation contexts are molecule-disjoint |
| q descriptor | exactly the parent `C.parquet::q_id` / `keys.json::q_keys`; no new descriptor, no second hop |
| label units | raw eV for PCQM; λ = 5 for the LOCAL ridge |
| ZINC | parent conclusion kept; no re-run |

The parent round's cell de-duplication (one row per `(C,S,b)`, label = mean over
distinct molecule identities) is reused. A cell keeps the number of distinct
molecules and the first molecule identity; duplicate rows of one molecule are
one identity.

## 1. Canonical pair orientation (correction A1)

All replacement pairs are canonical:

* `A = min(rank_i, rank_j)`, `B = max(rank_i, rank_j)`;
* `delta = y(C,B) - y(C,A)` in canonical orientation;
* if the original context order had `rank_i > rank_j`, the sign of the stored
  delta is flipped.

This is enforced in one function, `build_pairs_canonical`, used by every stage.
The parent orientation is reconstructed only for the correction comparison and
is never mixed into corrected results.

Required assertions (written to `checks.json`):

1. every pair row has `A < B`;
2. shuffling the cells of a context leaves the canonical pair set and every
   delta unchanged;
3. `(A,B)` and `(B,A)` are never two different structural pairs;
4. one exact context contributes at most one de-duplicated observation per
   canonical pair;
5. no `(ctx,A,B)` duplicate exists.

## 2. Metric definitions (correction A2)

Every metric is reported with three weightings:

* **observation-equal**: every evaluation pair observation has weight 1;
* **context-equal**: every evaluation exact context has weight 1 (observations
  inside a context first averaged);
* **pair-type-equal**: every canonical replacement type has weight 1; per-pair
  MAE is computed first, then averaged over pair types.

The parent `pair_equal` weighting was observation-equal; it was not a pair-type
macro and is renamed accordingly.

A complete 2×2 rectangle is counted as

`sum_{A<B} k_AB (k_AB - 1) / 2`,

where `k_AB` is the number of distinct contexts in which both A and B occur.
Pair occurrences inside one context are not rectangles.

Candidate-coverage numerator and denominator always use the same support set.
For every candidate set we report:

* number of discovery-screen candidates;
* number with sufficient evaluation support;
* candidate pair-type coverage;
* evaluation observation coverage;
* actual verification sample count per candidate.

## 3. Discovery / evaluation observation sets

Primary evaluation = parent `primary_evaluation_cells`. Discovery = parent
`discovery_mask`. All fits use discovery labels only.

For phase B the **main set** is the set of evaluation pair observations whose
canonical pair has:

* at least one discovery observation, and a frozen discovery median;
* `n_discovery >= 10` (the parent P1 support threshold);
* `n_evaluation_contexts >= 5` for that pair;
* A and B in the same supported SHARED component.

The **all-conditional-enabled set** is the subset of the main set on which
PAIR_Q, PAIR_Q_NULL, LOCAL and LOCAL_NULL all switch on their conditional
correction.

No MAE is compared across different observation sets. Every model comparison is
a paired difference on one fixed set.

## 4. Estimators (phase B)

All estimators reuse the parent formulas.

* ZERO: `delta_hat = 0`.
* PAIR: `delta_hat_AB = median_discovery(delta_AB)`.
* PAIR_MEAN: discovery mean instead of median (fit-target control only).
* SHARED: `delta_hat_AB = b_B - b_A`, context-centred least squares, each
  context equal weight, parent gauge per supported component.
* PAIR_Q: for every `(A,B,q)` with at least 10 distinct discovery exact
  contexts, `median_discovery(delta_AB | q)`; otherwise fall back to PAIR.
* PAIR_Q_MEAN: discovery conditional mean with the same support/fallback.
* PAIR_Q_NULL: q permuted in discovery within `(bond type, context row degree)`
  strata with fixed seed 0; identical conditional median and fallback rules.
* LOCAL: `b_B - b_A + d_(B,q) - d_(A,q)` with the parent context-centred
  ridge `lambda = 5`, parent minimum of 10 discovery contexts per q, and the
  parent same-component enable rule.
* LOCAL_NULL: parent matched permutation rule (seed 0), same estimator.

Every conditional model falls back to its unconditional base when its condition
is not supported. Enable/fallback is evaluated on the original true `q` of the
evaluation context; the NULL models were fitted on permuted q.

Reported for every model: delta MAE, RMSE, sign accuracy, enabled conditional
correction coverage. Key paired comparisons:

1. PAIR_Q − PAIR;
2. PAIR_Q − PAIR_Q_NULL;
3. LOCAL − SHARED;
4. LOCAL − LOCAL_NULL;
5. LOCAL − PAIR_Q.

## 5. Strict masking of direct supervision (phase C)

At most 20 candidate canonical pairs are fixed from structure only:

* the pair has direct discovery co-occurrence;
* the pair has at least 5 distinct evaluation exact contexts;
* selection order is discovery support descending, evaluation support
  descending, canonical keys ascending.

For each selected pair `(A,B)`:

1. every discovery context containing both A and B is found; the B cell is
   deleted from those contexts;
2. all original molecule identities in those deleted B cells are collected;
3. every discovery cell of every collected molecule identity is deleted from
   all discovery cuts (not only from the current cut);
4. the pair's usable discovery contexts are rebuilt;
5. it is verified that no discovery context contains both A and B, that no cell
   of a masked molecule remains in discovery, and that A and B remain in the
   same supported structural connected component.

If strict masking disconnects A and B, the pair is marked **not evaluable** and
is not replaced by a pair chosen after seeing labels.

On the identical strictly masked discovery data, refit:

* SHARED_MASKED;
* LOCAL_MASKED;
* LOCAL_NULL_MASKED (NULL q permuted inside the masked context set, within
  `(bond type, masked context degree)` strata, seed 0).

Evaluation stays on the original true q and original labels. Reference-only
models with extra direct supervision are:

* ZERO;
* PAIR_FULL (unmasked direct median);
* PAIR_Q_FULL (unmasked conditional median).

They are labelled as references and never used as fair unseen-replacement
contrasts.

The conditional-composition screen requires all of:

* LOCAL_MASKED better than SHARED_MASKED;
* LOCAL_MASKED better than LOCAL_NULL_MASKED;
* LOCAL_MASKED better than ZERO;
* context-equal and pair-type-equal directions agree;
* at least 10 evaluable masked pairs, each with at least 5 evaluation contexts;
* the gain is not entirely produced by one structural pair.

Report the actual relative gain, paired intervals, per-pair results, pair-type
macro, and leave-one-pair-out results. A 10% relative MAE improvement is kept
only as a practical screen for a possible next modelling round, not as a
scientific theorem.

Judgement labels for phase C:

* `CONDITIONAL_COMPOSITION_SUPPORTED` if every screen condition holds;
* `CONDITIONAL_COMPOSITION_SCREEN_ONLY` if a real but small positive gain
  exists without every condition;
* `STRICT_MASK_SUPPORT_INSUFFICIENT` if fewer than 10 evaluable masked pairs or
  another support condition fails;
* `NO_SUPPORTED_CONDITIONAL_RULE` if LOCAL_MASKED does not beat the controls.

## 6. Official Valid transfer (phase D)

No Valid label is used to fit or select anything. The frozen 1M discovery
models are applied unchanged. Complete molecules whose canonical graph already
appears in the 1M Train prefix are excluded. Valid Test labels are never read.

Report separately:

1. all evaluable Valid observations;
2. exact contexts unseen relative to the 1M prefix;
3. unseen exact contexts whose q is supported by discovery;
4. seen exact contexts;
5. the conservative clean subset.

"Unseen" means unseen relative to the **current 1M prefix**. The remaining
~2.38M official Train rows are not scanned and are not used to change this
definition. Official Valid is a development set, not Test.

## 7. Conditional functional-rule discovery (phase E)

For discovery observations with a sufficiently supported `(A,B,q)`:

* `global_residual = delta_AB - median_AB`;
* `conditional_residual = delta_AB - median_ABq`.

Tests:

* whether q conditioning reduces within-replacement response dispersion;
* whether any q group has a non-zero, relatively stable conditional effect;
* whether one structural pair changes sign across q groups and reproduces in
  evaluation;
* whether conditional functional-equivalence candidates increase;
* how many evaluation observations those candidates cover.

Conditional candidate rules (discovery only):

* at least 10 discovery exact contexts for the `(A,B,q)` triple;
* discovery `P80` absolute conditional residual `<= 0.10 eV` (sensitivity
  `0.20 eV`);
* at least 5 evaluation contexts to count as verifiable;
* near-zero (`|median_ABq| <= tau`) and clearly non-zero responses are reported
  separately.

Evaluation is used only for verification. No best-q search is performed. Up to
10 readable positive/negative examples are reported with A/B, q text, true
response distribution, predicted response, sample counts and unseen-context
performance.

## 8. Statistics and resources

Reuse the parent bootstrap style: 2000 context-level paired bootstrap
replicates, seed 0, resampling whole primary evaluation contexts that are
molecule-disjoint. Pair observations from different masked pairs may share
evaluation contexts and are never treated as independent samples. Pair-type
macro and per-pair results are always reported alongside pooled results.

CPU only. Local workers `<= 24`, one BLAS thread. Target wall time 20–40 min,
hard ceiling 60 min. Total RSS ceiling 64 GB. The run is chunked; full cell
tables are not materialised as Python objects.

Priority if time runs short: canonical orientation; PAIR_Q/LOCAL on one set;
strict masking with matched NULL; Valid unseen contexts; clean subset and
conditional candidate details. At the hard budget, completed artifacts are
saved and the run stops.

## 9. Required outputs

`PROTOCOL.md`, `configs/sfec_v1.json`, corrected code and tests,
`correction_table.csv`, `conditional_transfer.csv`, `strict_mask_results.csv`,
`valid_transfer.csv`, `conditional_function_candidates.csv`, `checks.json`,
`summary.json`, and a Chinese `REPORT.md` whose first page directly answers the
six questions in the task.

Final route labels are chosen from the task list without changing thresholds,
data or formulas. No next-round model experiment is started.
