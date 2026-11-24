# Task-conditioned structural vocabulary discovery v0 — PROTOCOL

**Round:** `task_conditioned_word_discovery_v0`
**Version:** 2026-11-24/v1 — **frozen before any formal GPU run**
**Task:** `ogbg-mollipo` whole-molecule lipophilicity regression
**Primary metric:** Dev RMSE in original label units at the best-Dev checkpoint
**Budget:** ≤ 2 h wall, ≤ 3 GPU-h; `res` dual-GPU; no requirement to exhaust it

This round asks one question only: from real R2 connected induced subgraphs, can a
compact **shared vocabulary selected by task supervision** be more predictive than a
same-size **frequency vocabulary**, and can it retain the prediction level of the
strong R2-SENTENCE trunk? It does not study RICE, Möbius, message scope, port
interaction, or EGO2 error strata.

All predictors are trained from scratch. No GNN, no pretraining, no teacher, no
distillation.

## 1. Read-only reuse and data

Read-only sources (sha256 pinned and re-checked at runtime by `code/tc_common.py`):

| source | role |
|---|---|
| `AGENT.md`, `docs/luyin/luyin23.txt` | repository rules / research intent |
| `r2_sentence_replication_v1/code/r2_common.py`, `r2_data.py` | frozen D0/D1 loader, split identities, R2 recipe |
| `rooted_interaction_cumulant_encoder_pilot_v0/code/rice_structure.py` | exact R2 support enumeration + canonical certificate |
| `graph_word_incidence_composition_pilot_v0/code/gwic_*.py` | label-safe raw layer, OGB feature dimensions |
| `structure_route_resolution_8h_v1/code/sr_struct.py`, `sr_fresponse.py` | verified canonicalisation / free-response conventions (read only) |
| `r2_internal_binding_gap_audit_v0/REPORT.md` | R2/EGO2 descriptive coordinates (read only) |
| `word_sentence_port_route_v1/PROTOCOL.md` | local precedent for pinned structure identity (read only) |
| `tracks/ksvd/code/run_molhiv_stable_exact_motif_probe.py` | precedent for nested scaffold stability selection (read only) |

Data:

* `ogbg-mollipo` official **Train only** (3360 molecules).
* Frozen D0 and D1 splits of official Train; both are 2856 Fit / 504 held-out.
* Official **Valid** labels are never loaded (they stay NaN in the training object).
* Official **Test** is never touched: no structure, no label, no statistic.
* Every statistic (target scaler, feature standardisation, category support,
  graph-scalar normaliser, vocabulary) is computed from the current Fit rows only.

Split identities:

| split | identity sha256 | role |
|---|---|---|
| D0 | `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` | primary |
| D1 | `c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee` | split check; re-discovers its own vocabulary |

## 2. Candidate vocabulary (frozen before supervision)

For every root `v`, `R2(v)` is the full radius-2 induced subgraph with every real
induced bond. A candidate is a rooted connected induced subgraph `S` with
`1 <= |S| <= 4`, `v in S`, every atom within distance 2 of `v`, and the **root in
slot 0**. Identity is `rice_structure.canonical_signature`: the lexicographically
minimal root-marked attributed-graph certificate over all non-root permutations.
Atom/bond ids, molecule ids, labels and model predictions are never part of the
type.

Granularity:

| size | granularities | fields |
|---|---|---|
| 3, 4 | `FULL`, `COARSE` | FULL: 9 atom / 3 bond OGB columns. COARSE: atomic number, formal charge, aromaticity / bond type |
| 1, 2 | `FULL` only | low-order X0 control, **not vocabulary words** |

The vocabulary pool is size 3/4 only. This follows directly from the brief's X0
definition (size-1/2 counts are already in X0) and from the requirement that
D_FREQ match a size-3/4 distribution. COARSE is an explicit attribute abstraction,
not a claim of full chemical identity.

A size-3/4 type enters the pool only if it occurs in **≥ 20 distinct Fit
molecules and ≥ 5 distinct Fit Bemis–Murcko scaffolds**. Candidates whose Fit
root-level count columns are exactly identical are deduplicated; the
representative is chosen by (smaller Fit document frequency, then COARSE, then
canonical key ascending). Every equivalent member is recorded. This is
observational equivalence on Fit, not a chemical-equivalence claim.

The whole candidate registry and the per-atom count matrix are frozen and
hashed (`analysis/candidates_{D0,D1}.json`,
`analysis/candidate_columns_{D0,D1}.npz`) **before** any supervised selection.

## 3. X0 low-order reference

`X0` contains:

* all size-1 FULL rooted type counts observed in Fit;
* all size-2 FULL rooted type counts observed in Fit;
* `n_atoms`, `n_bonds`;
* the 8 frozen graph scalars.

No size-3/4 type is in X0. All standardisation uses only the relevant training
part of the current fold or the Fit rows for the final ridge arms.

## 4. CPU: conditional task-vocabulary discovery

For a candidate `t`, let `c(v,t)` be the number of rooted supports of type `t`
with root `v`, and

`x_t(G) = sum_v log1p(c(v,t))`.

The current Fit is split into **three fixed scaffold folds**: Bemis–Murcko
scaffolds, sorted keys, `default_rng(20261124).permutation`, greedy whole-group
assignment to the currently smallest fold (ties by fold index). Folds are
scaffold-disjoint.

For each fold `j`:

1. Fit ridge `y ~ X0` on the fold's training part (features standardised with
   that training part; `alpha = 10`).
2. Fit multi-output ridge `X_words ~ X0` on the same training part, with the same
   standardisation and `alpha = 10`.
3. On the held-out fold compute the label residual `r` and every candidate
   residual `x_res`.
4. Record the Pearson correlation `corr_j(x_res, r)`.

Constant training columns, zero residual variance and non-finite correlations
are recorded separately; they are never filled with an arbitrary value. A
candidate is direction-consistent only when all three correlations are defined
and have the **same strict sign**.

Score:

`score(t) = min_j abs(corr_j(t))`.

Order: score descending, canonical key ascending. `K = 128`; if fewer than 128
candidates are direction-consistent, use the actual number; if fewer than 32,
stop neural training and report insufficient support. The per-candidate
original (raw `x_t` vs `y`) and conditional correlations are stored.

These correlations are a supervised **selection** statistic. They are not an
independent significance test and not a causal effect.

## 5. Frequency control and ridge arms

`D_TASK` is the ordered score vocabulary. `D_FREQ` is built from the **same
deduplicated candidate pool** and matches D_TASK exactly on the joint
distribution of (word size, granularity, Fit document-frequency quartile
bucket). The four empirical quartile buckets are formed by sorting eligible
candidates by (Fit document frequency ascending, canonical key ascending) and
splitting into four contiguous equal-count groups. Inside every cell, words are
chosen by document frequency descending, canonical key ascending. Mutual
exclusion is not enforced; the actual intersection is reported. If the two
vocabularies are identical, no duplicate neural arm is run.

Four CPU ridge models use the same Fit-only standardisation and `alpha = 10`:

| arm | features | role |
|---|---|---|
| `LOW_RIDGE` | X0 | low-order reference |
| `ALL_RIDGE` | X0 + all qualified candidates | vocabulary-compression reference |
| `TASK_RIDGE` | X0 + D_TASK | task selection |
| `FREQ_RIDGE` | X0 + D_FREQ | frequency control |

They predict the real Dev labels. `ALL_RIDGE` is a compression reference, not an
upper bound over all possible readouts.

## 6. Known-structure positive control

A separate synthetic regression uses the same real molecule graphs. From the
qualified size-3/4 pool, choose the candidate with strictly positive conditional
residual variance in **all three** inner folds, maximising the minimum fold
variance. This choice reads no real label. Labels are

`y_syn = Fit-standardised x_planted`

and the same Fit scaler is applied to Dev. The identical three-fold selection
flow is run; it must place the planted type (or a Fit-count-equivalent type) in
the synthetic task vocabulary. Synthetic Dev predictions are reported for
LOW/TASK/FREQ ridge. Synthetic labels never enter real Lipo training and are
stored in a separate artifact.

## 7. Neural shared root readout

All arms use the same non-iterative root representation:

* atom embedding 48 (per-field OGB embeddings summed);
* bond embedding 24 (per-field OGB embeddings summed);
* shared first-order pair encoder: `[neighbour atom 48, incident bond 24] = 72
  -> 128 -> 64`, sum over real incident neighbours;
* word projection: `K -> 128 -> 64`, SiLU;
* sentence MLP: `concat(root 48, neighbour 64, word 64) = 176 -> 192 -> 96`
  (SiLU, dropout 0.1);
* molecule readout: per-molecule sum/mean/std of the 96-d root representation,
  `log1p(n_atoms)`, Fit-standardised 8 graph scalars → `297 -> 192 -> 96 -> 1`.

The brief's `192->96` sentence line is implemented as **hidden width 192,
output width 96**; the specified components sum to 176, so the first projection
is 176→192. No unspecified feature was added. Readout `192->96->1` is exact.
This is the only dimension adaptation and is frozen in `configs/tc.json` before
the first formal run.

`z_v[t] = log1p(c(v,t))` for the frozen vocabulary. `BASE` uses an all-zero `z`
of the same shape and the same modules/parameter count; it is a low-order
reference and is not claimed to have the same effective capacity. D_TASK and
D_FREQ have identical parameter count, initialisation, batch order and
optimiser recipe.

Prohibited and absent: full R2 trunk bypass, S2 independent branch, extra R2
statistics, GNN, and any neural path reading an unselected word type.

Recipe: the frozen R2 recipe — AdamW, peak LR `3e-4`, min LR `1e-5`, weight
decay `1e-4`, gradient clip 5, batch 64 molecules, 5 warmup epochs, cosine,
max 240 epochs, min 80 epochs, patience 45, eval batch 256, best-Dev
checkpoint, whole-molecule standardised MSE only, no search.

## 8. Run matrix and progression

Stage 1, D0/s0, one process per GPU on `res`:

| run | split | seed | arm |
|---|---|---|---|
| `D0_S0_BASE` | D0 | 0 | BASE |
| `D0_S0_TASK` | D0 | 0 | D_TASK |
| `D0_S0_FREQ` | D0 | 0 | D_FREQ |

Historical same-batch `R2_REF` D0/s0 = 0.547679 is a descriptive coordinate, not
a new architecture attribution.

Stage 2 is executed only if **either**:

* D_TASK beats D_FREQ by ≥ 0.015 Dev RMSE; **or**
* D_TASK beats BASE by ≥ 0.03 Dev RMSE and is within 0.025 of `R2_REF` 0.547679.

Then run `D0_S1_TASK`, `D0_S1_FREQ` (seed 1).

Stage 3 is executed only if both D0 seeds show a positive TASK-vs-FREQ
difference and their mean difference is ≥ 0.01 RMSE. Then run D1/s0
`BASE`, `D_TASK`, `D_FREQ`. D1 re-discovers its own vocabulary, scaler and
weights from D1 Fit; no D0 selection result is transferred. All selection rules
were frozen before D1 Dev was read.

Otherwise stop. No third seed. No searching over K, word size, granularity,
alpha, MLP width or scoring function.

## 9. Evaluation

1. **Prediction:** Fit/Dev RMSE, Dev MAE, full curves, best epoch, epochs run.
2. **Task-selection gain:** same-molecule paired D_TASK vs D_FREQ scaffold
   bootstrap, 5000 whole-scaffold draws, seed 1703, 95% percentile interval.
3. **Compression:** candidate type count, selected words, root coverage,
   molecule coverage, effective words per root.
4. **Stability:** D0/D1 selected-word intersection (both Fit sets overlap, so
   this is not independent replication), size/granularity distribution,
   cross-scaffold support.
5. **Structure content:** top 20 real words by frozen score, with reproducible
   rooted graph serialisation and actual molecular occurrence positions.
6. **Information repetition:** selected-word vs low-order feature correlations,
   equivalent columns, highly correlated candidate pairs.

Selection scores, neural weights, word occurrences and local correlations are
never called true chemical contributions.

## 10. Verdict and deliverables

Verdict labels: `VOCABULARY_PROMISING`, `DISCOVERY_ONLY`, `COMPRESSION_ONLY`,
`NO_SUPPORTED_DISCOVERY`, `INSUFFICIENT_OR_INVALID`, with the exact rules in
`configs/tc.json`. They are resource judgments for this round only; a failure of
one selection method is not a claim that all task-conditioned vocabularies are
unlearnable.

Artifacts: `PROTOCOL.md`, `configs/tc.json`, `code/`, `tests/`,
`analysis/selected_words.*`, `analysis/selection_scores.*`, candidate/fold/data
identity hashes, per-run manifest/checkpoint/predictions/curves, `REPORT.md`.
All GPU runs go through `rr` on host `res`; local Git is the source of truth.