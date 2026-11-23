# RICE-v0 — rooted interaction cumulant encoder pilot — protocol

Round: `rooted_interaction_cumulant_encoder_pilot_v0`
Version: 2026-11-24/v1, **frozen before any formal training run**
Task: `ogbg-mollipo` whole-molecule lipophilicity regression
Primary metric: whole-molecule Dev RMSE in original label units (D0, seed 0)
Runs: exactly two (`RICE_CUMULANT_S0`, `RICE_RAW_S0`). No third run, no retry, no
search, no second seed, no D1, no official Valid/Test.

This round is an **independent new computational method**. It is not LCRC-v1,
GWIC-v1, PRCG-v1 and not a W2S distillation variant.

## 1. Question

Can "real graph factor contraction + root-conditioned Möbius inversion" learn
*irreducible* connected structural interactions end-to-end from the raw molecular
graph, and do those interactions form a second-order local sentence that predicts
better than the same responses **without** the inversion?

Two formal arms answer the second half. The first half is answered by the
pre-training mathematics checks (section 7, A–C) and by the frozen-checkpoint
interventions (section 9).

Forbidden in this round, and verified absent: R2-SENTENCE / LCRC hidden vectors
or checkpoints as teachers, representation distillation, pretraining or
auxiliary reconstruction losses, the original R2 S1/S2 statistical embeddings as
inputs, GNN/MPNN/Transformer/attention selectors, a `ComposeMLP([child_A,
child_B, …])` standing in for structural composition, PRCG's 32 rules or model
weights, a second seed, D1, official Valid/Test, automatic hyper-parameter
search. R2-SENTENCE appears only as a reference coordinate in the report.

## 2. Data

D0 only, as frozen by `acra_relation_generalization_pilot_v1`:

| item | value |
|---|---|
| raw structure cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`, sha256 `fe002287…23c7c` |
| reused loader | `graph_word_incidence_composition_pilot_v0/code/gwic_data.py`, sha256 `f02863be…0cef` (pinned again at runtime; label-safe helpers only, no parent model/feature/checkpoint) |
| Train labels | `tracks/structural-sufficiency/structural_path_composition_route_v1/data/labels_official_train.npz` (3360 rows) |
| D0 split | `acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json`, split sha256 `60ac13c0…559e` |
| Fit / Dev | **2856 / 504**, scaffold-disjoint, exactly covering official Train |
| target scaler | Fit rows only, z-score (`mean 2.185815826311436`, `std 1.2139664740467941`) |
| graph scalars | 8 deterministic whole-graph scalars, standardised on Fit rows only |
| atom / bond fields | all 9 official OGB atom categorical fields and all 3 official bond categorical fields |
| official Valid | structure only; labels are NaN in the training object |
| official Test | never touched: no structure, no label, no metric |

## 3. Root-conditioned connected structure words

For every atom `v` of a molecule, the root context is the induced subgraph on
`{t : dist(v,t) <= 2}` with every real bond.

A **root-conditioned structure word** is a distinct atom support `S` with

* `v ∈ S`, `1 ≤ |S| ≤ 4`,
* every atom of `S` is within distance 2 of `v`,
* `G[S]` is connected,
* the internal bond set of `S` is **exactly** "every real bond whose two
  endpoints are both in `S`" (so a triangle's closing bond is always read, and a
  4-clique contributes all six bonds).

The same atom set under a different root is a different root-conditioned word.
Supports overlap freely; no non-overlapping cover of `R2(v)` is required. The
word stores the root in slot 0 and the remaining atoms in ascending
molecule-local order; atom ids are enumeration keys, never features.

## 4. Learnable finite-state graph energy

Fixed and shared by every molecule, root and word type (no per-word-type
parameter anywhere):

| item | value |
|---|---|
| independent energy channels | `R = 16` |
| internal states per atom, per channel | `d = 2` |
| unary potential | `u_i ∈ R^[16,2]` from the 9 OGB atom categorical fields (9 embeddings of width 16 → MLP 144-64-32 → `2·tanh`) |
| pairwise potential | `J_e ∈ R^[16,2,2]` from the 3 OGB bond categorical fields (3 embeddings of width 12 → MLP 36-64-48 → `2·tanh`) |
| binary values per channel | three independent values `00`, `01/10`, `11`; `J_e[a,b] = J_e[b,a]` by construction |
| potential bound | `2.0` (tanh) so `exp` can never overflow |

```
E_r(S, σ)     = Σ_{i∈S} u_i[r, σ_i] + Σ_{(i,j)∈E(S)} J_ij[r, σ_i, σ_j]
logZ_r(S)     = logsumexp_σ E_r(S, σ)                      (exact enumeration, ≤16 states)
logZ_r(S; t)  = the same sum restricted to σ_v = t
logZ(empty)   = 0
Φ_r,t(S, v)   = logZ_r(S; σ_v=t) − logZ_r(S \ {v})
κ(S, v)       = Φ(S, v) − Σ_{T ⊊ S, v∈T} κ(T, v)           (size 1 → 4, increasing)
```

`S \ {v}` may be disconnected; its energy is still well defined and is evaluated
by the same exact enumeration with the root's unary and root-incident pairwise
terms switched off, restricted to root-state 0 so every remaining state is
counted exactly once. No sampling, no message passing, no approximation.

`κ` is the only high-order structural composition operation in this round. It is
never replaced by an MLP over concatenated child vectors.

## 5. From words to a sentence

For every root `v`, and for each order `k = 1,2,3,4`, the model pools the `k`-th
order values over all `v`-rooted words of that order:

```
order block_k = [ Σ, mean, std ] over the 32 (channel, root-state) components
                 -> independent LayerNorm_k (96 dims)
                 + log1p(count_k)  (1 dim)
h_v           = shared MLP( concat_k block_k )  : 4·97 = 388 -> 128 -> 96
```

The whole-molecule prediction uses **only**

```
[h_sum, h_mean, h_std, log1p(atom_count), 8 fit-standardised graph scalars]  (297 -> 128 -> 64 -> 1)
```

No R2 S1/S2 statistics, no closure/path embeddings, no parent hidden state.
All parameters are trained end-to-end by whole-molecule standardised MSE on D0
Fit rows. There is no other supervision term.

Frozen widths: `R=16`, `d=2`, `h_v=96`, sentence MLP hidden 128, readout hidden
128/64, dropout 0.1, potential bound 2.0. Parameter total 129,198.

## 6. The two arms

| arm | sentence input |
|---|---|
| `RICE_CUMULANT` | `κ(S,v)` (Möbius inversion, sizes 1→4) |
| `RICE_RAW` | `Φ(S,v)` (same supports, same `Φ` computation, same parameters, same readout) |

The arms share the identical module set, identical initialisation and identical
training recipe; the only difference is which tensor enters the sentence
pooling. `RICE_RAW` is the test of whether removing low-order redundancy and
forming irreducible words has value.

## 7. Pre-training checks (must all pass before any formal run)

`code/rice_checks.py` writes `analysis/checks.json`.

**A — inversion identity.** For *all* 75 graphs with ≤ 4 atoms, every root and
every connected root-conditioned support, `Φ(S,v) == Σ_{T⊆S, v∈T} κ(T,v)` with an
**independent brute-force** subset sum as the reference; error `< 1e-5`. The
vectorised production path is additionally compared against the brute-force
`Φ`/`κ` in float64, and the float32 path's deviation is reported.

**B — connectivity.** `κ(S,v) ≈ 0` for every disconnected root-anchored `S` with
`|S| > 1`; and with all bond potentials set to zero, every `κ` of size ≥ 2 is
zero.

**C — real structure sensitivity.** Two graphs with the same atom and bond
attribute multisets but different real connectivity (4-cycle vs triangle+tail)
and two graphs with the same multisets but a different attribute↔topology
binding (`C–O–O` rooted at C with a chain tail vs `O–C–O` rooted at C) must be
separated by `Φ` and by `κ`, using potentials derived deterministically from the
fields. The 3-cycle must store all three internal bonds and its `κ` must differ
from the open 3-chain; a 4-clique must use all six bond slots. A separate check
records that the order-4 `κ` of a middle-rooted 4-path and of a 4-star is
identically zero (conditional independence of the root branches given the root).

**D — invariance.** Atom renumbering, bond direction swap, bond storage order,
batch composition and molecule order leave the prediction and the
per-certificate `κ` unchanged.

**E — gradients.** Unary, pairwise, embedding, sentence and readout gradients are
finite and non-zero on the first real batch; order-3 and order-4 `κ` are non-zero
on non-degenerate test graphs; 30 optimiser steps reduce the standardised MSE;
no NaN/Inf.

**F — cost.** Word census and per-order distribution (including the fraction of
roots that own size-3 and size-4 candidates); evidence that the implementation is
vectorised (module forward-call counts per 64-molecule batch do not scale with
the word count; supports are bucketed by size, `logsumexp` is batched and subset
containment is precomputed); measured step cost.

## 8. Formal training

D0, seed 0 only, two arms, frozen recipe (identical to the frozen R2 recipe):

| item | value |
|---|---|
| optimiser | AdamW, peak LR 3e-4, min LR 1e-5, weight decay 1e-4 |
| schedule | 5 warmup epochs, cosine to 1e-5, max 240 epochs |
| batch | 64 molecules |
| gradient clip | 5.0 |
| minimum epochs / patience | 80 / 45 |
| checkpoint | best Dev RMSE |
| objective | whole-molecule standardised MSE only |
| widths | frozen in section 5; **no** Dev-based width or recipe change |

Execution: one process per GPU via `rr` on `res`, both arms in parallel, at most
2 GPU-hours in total. Preprocessing wall time is reported separately. If the
budget cannot be met, the bottleneck is recorded and the round stops — the
algorithm is not swapped for a cheaper one.

## 9. Post-training mechanism analysis (frozen checkpoint, no retraining)

1. per-order `κ` mean, std, norm and effective rank (participation ratio) on Dev;
2. zero every `κ` of size ≥ 3 before sentence pooling, re-evaluate Dev RMSE;
3. zero size ≥ 4, re-evaluate Dev RMSE;
4. zero size ≥ 2, re-evaluate Dev RMSE;
5. secondary variant of (2) that also zeroes the order counts.

Each is a frozen-checkpoint intervention, not a capacity-matched control. The
recompute path is validated twice before it is used: a no-op intervention must
reproduce the checkpoint's own predictions, and one real type ablation must match
a direct masked model forward.

Word types are then defined on **Fit only** by a root-marked attributed-graph
canonical certificate (lexicographically minimal relabelling of the non-root
slots over the full OGB categorical node and bond labels). Only types that appear
in at least **10 Fit molecules and 5 Fit scaffolds** are studied. For each such
type, all of its `κ` contributions are switched off in the frozen model and the
Fit prediction change is measured; the largest-impact types are reported with
their shape, the root's position in the word, their cross-molecule and
cross-scaffold reuse, and their Fit prediction change. The same ablation is then
run once on Dev as a **descriptive** check for the types already selected on Fit.
Dev is never used to select a type. These ablations are not chemical causal
effects; the latent states and channels are permutation-non-identifiable, so no
single `κ` component may be given a fixed chemical reading.

## 10. Verdict rule (engineering, single seed)

Main contrast `Δ = RMSE(RICE_RAW) − RMSE(RICE_CUMULANT)` (positive = CUMULANT
better), with a paired molecule-level bootstrap on the same Dev rows as a
descriptive interval.

* `RICE_PROMISING` iff `CUMULANT Dev ≤ 0.60` **and** `Δ ≥ 0.015` **and** zeroing
  size ≥ 3 on the frozen checkpoint worsens Dev RMSE by `≥ 0.015` **and** the
  high-order response, the real bond potentials and the training gradients are
  non-degenerate **and** at least one reusable cross-scaffold word type has a real
  Fit effect.
* `RICE_MECHANISM_ONLY` iff the CUMULANT arm is not close to R2 but `Δ > 0` with a
  clear direction and the high-order structure is genuinely active.
* `RICE_NO_SUPPORTED_GAIN` otherwise.

The thresholds are an engineering judgement for one seed-0 run, not a
significance test. No threshold may be repaired by a new run.

## 11. Deliverables and stop condition

`PROTOCOL.md`, `code/`, `configs/rice.json`, `tests/`, `analysis/` (checks,
summary, kappa statistics, interventions, word types, verdict), `results/` with
the two checkpoints and their Fit/Dev predictions, and a short `REPORT.md`
answering exactly the five questions in the brief.

The round stops after these two runs and the analysis. No second seed, no D1, no
official Valid/Test, no order extension, no latent-state search, no RICE-v1.