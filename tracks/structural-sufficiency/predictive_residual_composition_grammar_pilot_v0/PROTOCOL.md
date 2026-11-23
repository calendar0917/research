# PRCG-v0 — Predictive Residual Composition Grammar (pilot protocol)

**Protocol id** `prcg-v0-predictive-residual-composition-grammar-pilot`
**Status** frozen before any formal training run; grammar, model and recipe are fixed in
`configs/prcg.json`.
**Scope** one algorithmic claim, on one split, with one seed. This is a pilot, not a
benchmark submission.

---

## 1. The single claim under test

> Connected graph words whose *merge rules* were selected by **cross-molecule predictive
> residual gain** improve whole-molecule regression in a **fixed, non-iterative** composition
> model, relative to a **pure-frequency** word lexicon of the same shape.

Everything else is held constant so that only the *source of the lexicon* (and one port
ablation) can move the number. This is explicitly **not**:

* GWIC-v1 (no incidence-binding re-derivation, no new binding objective),
* BBPA-v1 (no bilinear pair algebra, no attachment-pattern table),
* an R2-SENTENCE patch (no new sentence/context design, no re-tuning of the R2 recipe),
* a pretraining / GNN / Transformer / K-SVD / attention-selector / new-dataset study.

No iterative message passing, no per-word-type lookup tables, no auxiliary objectives, no
hyper-parameter search, no second seed, no D1, no Valid/Test evaluation.

### The two historical failure modes this protocol must exclude

1. **Pooled enumerated candidates without a reliable gain** (structural-query learning):
   candidates were enumerated, never screened on held-out *molecules*, and the pool grew
   without a per-candidate acceptance test. → Here every rule must pass a *two-direction,
   scaffold-disjoint, cross-fitted* held-out-MSE screen on D0 Fit only, before the model
   exists.
2. **An explicit composer collapsing to a constant** (K-SVD binding note: shared `Compose`
   became an exactly constant function, `composition_disagreement` 6.7e-9, 53.7 % of
   parameters weight-decayed away, effective rank ≈ 1.05). → Here non-degeneracy is a
   *pre-training gate* (dispersion, effective rank, gradient norms, port sensitivity) and a
   *post-training measurement* (Dev word-state participation ratio, variance), not a
   post-hoc excuse.

## 2. Data

| item | value |
| --- | --- |
| raw cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz` |
| raw sha256 | `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| molecules | 4200 (3360 official Train / 420 official Valid / 420 official Test) |
| split | `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json` |
| split sha256 (canonical json) | `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` |
| labels | `structural_path_composition_route_v1/data/labels_official_train.npz` (3360 rows) |
| D0 Fit | 2856 molecules, 1354 Train-sub scaffolds |
| D0 Dev | 504 molecules, 231 scaffolds (scaffold-disjoint from Fit) |
| official Valid/Test | **never read, never featurised, never evaluated** |

Label safety is enforced in code: the raw cache is loaded through a key whitelist that
contains `train_idx`/`valid_idx` but **no** labels and **no** `test_idx`; labels exist only
on official Train rows; every finite label outside official Train is a hard error.
Standardisation (target and the eight graph scalars) is computed from D0 Fit rows only and
stored with the bundle.

## 3. Words and word types

A **word** is a *real connected induced* atom support of a molecule graph, `|support| ≤ 5`.
Words are built by merging two adjacent words along the **real bonds** between them,
starting from single atoms. Every atom is covered exactly once; every real bond is either
internal to exactly one word or connects two different words.

**Word type** uses only the frozen core fields:

* atom: atomic number, formal charge, aromaticity,
* bond: bond type, conjugation.

Everything else (degree, ring membership, hybridisation, chirality, numH, radical, bond
stereo) stays available to the model as *context* but can never enter a type. The type
certificate is exact, not hashed: `word_signature` enumerates all `k! ≤ 120` orderings of the
support, takes the lexicographically smallest adjacency string as the certificate and
derives automorphism orbits from the set of minimising orderings.

**Rule** = `(child type A, child type B, actual attachment pattern) -> parent type`, where the
attachment pattern records, per crossing bond, `(bond type, conjugation, left port orbit,
right port orbit)` and the orbit sizes. The child orientation is canonical (lexicographically
smaller certificate on the left); for equal certificates the order is the (structure-derived)
word-history order, so a rule is orientation-free.

**Matching** is a deterministic greedy maximal matching of a rule's occurrences ordered by
exact canonical atom ranks. The canonical order is an individualisation–refinement labelling
that is deterministic and label-independent: the target cell is always the first
non-singleton colour class (an invariant choice), and the resulting order is a graph
invariant *up to automorphism* (`rank_2[π(i)] = rank_1[σ(i)]` for some automorphism `σ`).
No raw atom index ever breaks a tie. If the search exceeded its node budget the code raises
instead of returning an unverifiable order.

## 4. Stage 1 — predictive grammar discovery (Fit only, no model, no GPU)

For each round, starting from the atom partition:

1. Enumerate all legal merge candidates (adjacent word pairs, `|A|+|B| ≤ 5`), deduplicated
   by `(left cert, right cert, pattern)`. At most `candidate_cap = 400` candidates per round
   (deterministic order by occurrence count, then key).
2. Split D0 Fit into two **scaffold-disjoint** folds (seed `20261201`; each fold has
   ≥ 8 molecules and ≥ 4 scaffolds, enforced in code).
3. Score each candidate by its **two-direction cross-fitted held-out MSE gain**:
   for direction `p → q` fit `Ridge(y ~ X)` on fold `p`, then compare
   `Ridge(y ~ [X, x_m])` against `Ridge(y ~ X)` on fold `q`, where `X` is the current
   language's word-count matrix plus the eight graph scalars and `x_m` is the candidate's
   count column (all standardised, one fixed `λ = 5`).
4. Accept the best candidate only if **both directions are positive** (held-out MSE
   improvement in each fold), its Fit support is ≥ 2 molecules, and it passes the fold size
   constraints. At most one rule per parent type; a rule whose parent type already exists is
   skipped.
5. Apply the accepted rule to the Fit molecules and repeat. Stop at
   `max_rules = 32` rules or when no candidate has a positive two-direction gain.

If **no** rule is accepted, the run prints `NO_TRANSFERABLE_PREDICTIVE_MERGE` and stops
before any GPU/CPU training (the pilot is negative by construction).

**Frequency control.** The same procedure with the score replaced by *Fit occurrence
frequency* (number of molecules in which the merge is realised), same candidate set, same
matching, same constraints, stopping at **exactly** the number of rules the predictive
grammar accepted. The two lexicons therefore have identical size, identical word bound and
identical matching machinery; they differ only in *which* merges are in the lexicon
(predictive-residual vs frequent).

Deterministic replay of the whole discovery trace is a required check (C3).

## 5. Stage 2 — the composition model (identical for all arms)

`~105 k` parameters, three arms differ only in `(grammar, port mode)`.

* **Atom words** come from frozen core-field embeddings (`atomic number`, `formal charge`,
  `aromaticity`) plus a small MLP over those embeddings and the four intrinsic atom fields.
* **Merged words** are computed by one **shared symmetric** `Compose(z_left, z_right,
  crossing-bond core features, attachment-port features)` applied recursively by word size
  (2 → 5), with a LayerNorm. No per-type lookup table, no bypass, no residual shortcut from
  children to the parent: a merged word’s state is a pure function of its frozen type
  (verified: maximum within-type dispersion ≈ 1e-6).
* **Inter-word term** `Q(z_A, z_B, port_A, port_B, bond, local_context)` is computed from
  *symmetric* pair features only (`z_A+z_B`, `|z_A−z_B|`, `z_A⊙z_B`, and likewise for the
  port roles and the local context), so the edge term depends on the unordered pair.
* **Readout** = `Σ f(word) + Σ q(edge) + small nonlinear correction(readout features)`, where
  `f` is a per-word scalar, `q` a per-edge scalar, and the correction sees per-molecule
  aggregates of the final words and edges (sum / mean / std of `z` and of the `Q` hidden
  state), two counts, and the eight graph scalars. There is **no iterative message passing**.

Arms:

| arm | lexicon | port roles |
| --- | --- | --- |
| `FREQ_GRAMMAR` | frequency grammar (32 rules) | real |
| `PRED_GRAMMAR` | predictive grammar (32 rules) | real |
| `PRED_PORT_MEAN` | predictive grammar (32 rules) | real port roles replaced by the mean over the word's ports |

Single-port words have one port, so their mean role equals the real role; the ablation is
therefore confined to multi-port words (a pre-training check verifies both the equality and
the confinement).

## 6. Stage 3 — pre-training gates (all must pass before training)

1. Word supports are real, connected, induced; exact atom/bond coverage; no invented bond.
2. Rule replay is deterministic and reproduces the discovery trace round by round.
3. Same type ⇒ same internal state (max within-type `z` dispersion < 1e-5) for both grammars.
4. Atom renumbering invariance: the canonical order changes only by an automorphism, the
   partition reproduces in rank space, the word-type multiset is identical and predictions
   agree to < 1e-5. Bond-order and batch-order invariance likewise.
5. `REAL` vs `MEAN` differ only in the port-role fields (`x_l, x_r, e_i, e_j`), plus a
   checkpoint toggle that switches the port mode at load time.
6. `Compose` and `Q` gradients are finite and non-zero; 30 optimisation steps reduce the loss.
7. Fit-only statistics: target scaler and graph standardiser recomputed from D0 Fit match the
   stored values; the discovered rules are unchanged when a Dev-label shift is simulated.
8. Non-degeneracy at initialisation: per-size `Compose` dispersion > 0, effective rank of `z`
   > 1, variance > 0 (K-SVD collapse signature absent).

Expected result: 22/22 checks pass, recorded in `analysis/checks.json`.

## 7. Stage 4 — training (frozen R2 recipe, D0 seed 0 only)

AdamW, peak LR `3e-4`, weight decay `1e-4` (biases/norms excluded), gradient clip `5`,
batch `64` molecules, 5-epoch warmup, cosine decay to `1e-5`, max 240 epochs, min 80 epochs,
early stopping with patience 45 on **Dev RMSE**, best-Dev checkpoint. Objective is the
whole-molecule standardised MSE only; no auxiliary targets, no search.

Budget: ≤ 4 GPU-hours and ≤ 2 h wall in total. (The pilot is small enough that the frozen
recipe completes on CPU in minutes; the wall clock actually used is reported.)

## 8. Stage 5 — results and decision rule

Primary metric **Dev RMSE** (label units), secondary **Dev MAE**.

* `Δ_discovery = RMSE(FREQ_GRAMMAR) − RMSE(PRED_GRAMMAR)` (positive ⇒ predictive lexicon better)
* `Δ_port = RMSE(PRED_PORT_MEAN) − RMSE(PRED_GRAMMAR)` (positive ⇒ real ports help)

Uncertainty: paired bootstrap over the 504 Dev molecules (10 000 resamples, same resample
indices for all three arms, 95 % percentile interval), reported for RMSE and for both Δs.

Route label, first match wins:

| label | condition |
| --- | --- |
| `PROMISING_PREDICTIVE_GRAMMAR` | `RMSE(PRED) ≤ 0.56` **and** `Δ_discovery ≥ 0.015` **and** `Δ_port ≥ 0.01` **and** word states non-degenerate |
| `LEXICON_SIGNAL_WITHOUT_PORT_GAIN` | `Δ_discovery ≥ 0.015` and not the above |
| `PORT_SIGNAL_WITHOUT_PREDICTIVE_LEXICON` | `Δ_port ≥ 0.01` and not the above |
| `INCONCLUSIVE_SINGLE_SEED` | neither Δ reaches its threshold, single seed |
| `NO_TRANSFERABLE_PREDICTIVE_MERGE` | Stage 1 accepted no rule |

Reference points on the same D0 Dev split: R2-SENTENCE `0.5506` (seed 0) / `0.5464`
(seed 1), GINE `0.5266`. The pilot is *competitive* only if `RMSE(PRED_GRAMMAR)` is within
~0.01 of the R2-SENTENCE line without any iterative GNN or sentence mechanism.

Also reported (descriptive, not gating): vocabulary size, cross-scaffold type reuse,
multi-port fraction, mean words per molecule, the most frequent words and the
highest-residual-gain rules, Train–Dev gap, and the post-training word-state effective rank.

## 9. Stopping rules

Stop after the three seed-0 arms on D0. No second seed, no D1, no Valid/Test, no extra arm,
no hyper-parameter change after seeing Dev, no PRCG-v1 proposal in this round. If a
pre-training gate fails and cannot be fixed without changing the frozen protocol, report the
blocker instead of training.

## 10. Deliverables

`PROTOCOL.md`, `configs/prcg.json`, `code/` (grammar, discovery, model, tensorisation,
training, checks), `tests/`, `grammar/` (both lexicons + trace + stats), `results/` (three
runs + checkpoints), `analysis/` (bootstrap, vocabulary, checks) and `REPORT.md`.

## 11. Falsification

The claim is falsified if `Δ_discovery ≤ 0` (predictive lexicon no better than frequency) or
if the pre-training non-degeneracy gates fail, or if `PRED_GRAMMAR` needs an iterative
mechanism to become competitive. A positive `Δ_discovery` with `Δ_port ≈ 0` means the gain
is real but not attributable to attachment-port identity, and must be reported as such.