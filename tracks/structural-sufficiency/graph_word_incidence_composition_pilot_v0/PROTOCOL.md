# GWIC-v0 — graph word–incidence composition pilot — protocol

Round: `graph_word_incidence_composition_pilot_v0`  
Version: 2026-11-23/v1, **frozen before formal training**  
Task: `ogbg-mollipo` whole-molecule lipophilicity regression  
Primary metric: whole-molecule **Dev RMSE in original label units**  
Seed: `0` only. No D1, no official Valid/Test labels, no second seed.

This protocol was written before any formal GPU run. The only pre-training
adjustment permitted here is the candidate-size audit in §4: it was run before
training and is recorded below; no threshold was changed after seeing Dev
results. Any later change is append-only and dated in §12.

## 1. Question

A complete second-order rooted ego graph `R2(v)` is treated as a sentence. A
candidate word is a real connected induced atom support inside that sentence.
The round asks:

1. does the complete R2 sentence trunk itself predict whole-molecule
   lipophilicity competitively?
2. does learning to select connected candidate words add absolute predictive
   value over the sentence trunk?
3. does binding selected words back to **where their atoms sit in the real
   molecular graph** add value beyond word content and unbound word statistics?

The mechanism question is explicit: **is any gain due to word-to-original-atom
incidence, rather than to a larger set of learnable statistics?**

## 2. Data, split and label isolation

| item | value |
|---|---|
| raw structure cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`, sha256 `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| raw members read | `node_ptr, bond_ptr, atom_feat, bond_feat, bond_u, bond_v, degree, min_ring, in_cycle, graph_n_*, smiles, row_ids, train_idx, valid_idx` — **no `labels`, no `test_idx`** |
| Train labels | `tracks/structural-sufficiency/structural_path_composition_route_v1/data/labels_official_train.npz`, sha256 prefix `a44d29deba1fd36e` |
| D0 split | `tracks/structural-sufficiency/acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json`, identity `sha256_json({train_sub_rows, dev_rows, seed}) = 60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` |
| Fit | D0 `train_sub_rows`: **2856** molecules |
| Dev | D0 `dev_rows`: **504** molecules |
| official Valid | structure not used for training; **labels never loaded** |
| official Test | **never touched** |

The training object carries finite labels only on official Train rows. Fit and
Dev are disjoint and cover exactly the 3360 official Train rows. Target
normalisation is fit-only z-score:
`mean=2.185815826311436`, `std=1.2139664740467941` (n=2856). All metrics are
inverted back to original label units. Dev labels are used only for per-epoch
evaluation and best-Dev checkpoint selection; they never enter a gradient,
scaler, category support, vocabulary or any learned statistic.

All OGB atom categorical fields (9) and bond categorical fields (3) are used.
Fit-only category support zeroes unsupported embedding rows (including composite
path/neighbour codes). No official Valid/Test label is loaded at all.

## 3. Sentence object: complete rooted radius-2 ego graph

For every atom `v` of every molecule:

```
R2(v) = {t : dist(v,t) <= 2}
```

with **every real induced bond whose two endpoints are both in R2(v)**. No
truncation, no sampling, no path-bag substitute. The sentence keeps:

* root atom `v` and each atom's original 9 categorical fields;
* all original induced bond records and 3 categorical fields;
* actual graph distance from the root (0/1/2);
* rooted non-backtracking 2-paths `v -> w -> u` (`u != v`) with their real bond
  types and closure indicator;
* complete R2 atom summaries at distance 1/2 and complete R2 bond summaries,
  including bonds not incident to the root.

Every atom is a root, so a molecule has `n_atoms` sentence objects.

The non-iterative sentence trunk is LCRC-style: atom embedding (48), bond
embedding (24), S1 incident-bond sum/mean, S1 typed-neighbour sum, S2 rooted
typed 2-path sum + gated end-atom content, complete R2 atom/bond summaries,
and fit-only-standardised atom/R2 scalars. A shared MLP maps this to
`h_v in R^96`. There is no message passing, GINE, Graph Transformer or graph
pretraining. `h_v` uses real atom–bond connectivity and the root position.

## 4. Candidate words: connected induced supports

Inside `R2(v)` enumerate every distinct atom support `S` satisfying

* `v in S`;
* `2 <= |S| <= 4`;
* the original graph induced on `S` is connected.

Each support is enumerated once. It keeps all real induced internal bonds and
the real **boundary bonds from `S` to `R2(v) \ S`**. Boundary bonds are never
counted as internal. Atom IDs are molecule-local indices only and are never
learnable features. Candidates are not truncated by atom ID and no per-molecule
top-K is applied. Supports for one root are sorted lexicographically; candidate
order is canonical.

Formal record:

```
W(v,S) = (root=v, atom_support=S, induced_bonds, boundary_bonds)
```

**Pre-training candidate audit (run before formal training).** On all 3360 D0
sentences the complete enumeration gives **1,124,230 candidates** (mean 334.6
per molecule, median 342; per-root mean 12.42, median 10, p99 29, max 53). This
is below the frozen 20,000,000 cap, so **support_max stays 4 for all arms** and
the fallback to 3 is not used. No per-molecule candidate was dropped. The
structure arrays occupy about 252 MB and the full audit peaked at about 742 MB
RSS; full D0 structure preparation took about 32 s. These facts are recorded in
`analysis/checks.json` check 15.

## 5. Word encoder and learned selection

Atom embeddings are 48-d, bond embeddings 24-d, using all original OGB
categories. One shared MLP encodes each candidate with:

* root atom representation;
* support atom sum / mean / std;
* all internal bond sum / mean / std;
* for every internal bond `(a,b)`, a shared joint representation from
  `a+b`, `|a-b|`, `a*b` and the bond attributes, aggregated by sum/mean;
* support atom count, internal bond count, cycle rank;
* root degree in the word and induced internal distance distribution;
* boundary bond count and boundary categorical type fractions.

Output `w_(v,S) in R^64`.

Same-root candidates are scored by a shared
`ScoreMLP([w_(v,S), h_v])`, normalised with softmax over **all** candidates of
that root:

```
alpha_(v,S) = softmax_S s_(v,S)
z_v = sum_S alpha_(v,S) w_(v,S)
```

No hard top-K. `alpha` is a selection weight, not a causal contribution and not
a claim of discovered chemical functional groups.

## 6. Word-to-original-graph binding

```
c_(v,t) = sum_{S: t in S} alpha_(v,S)          (0 if t not in R2(v))
o_uv    = sum_t c_(u,t) c_(v,t)                 (real shared atoms)
r_uv    = c_(u,v)
r_vu    = c_(v,u)
```

For each real undirected bond once, build

```
q_uv = Q([h_u+h_v, |h_u-h_v|, h_u*h_v,
          z_u+z_v, |z_u-z_v|, z_u*z_v,
          bond_emb, o_uv, r_uv+r_vu, |r_uv-r_vu|]) in R^64
```

`Q` is strictly endpoint-swap invariant. This is a one-pass word–incidence
fusion: no iteration, no hidden-state GNN, no update of `h`.

Whole-molecule readout: `h_v` sum/mean/std; `z_v` mean/std; `q_uv` mean/std;
and 8 deterministic graph scalars (log atom/bond counts, cycle rank, component
count, mean/max degree, ring-atom fraction, aromatic-atom fraction), all
fit-standardised. A shared nonlinear MLP outputs the standardised regression
value.

## 7. Fixed arms (seed 0)

| arm | words | Q geometry | role |
|---|---|---|---|
| `SENTENCE_BASE` | no | no Q | complete R2 trunk + raw atom/bond summaries + nonlinear graph head |
| `WORD_GEOM_NULL` | yes | three coverage inputs fixed to 0 | word content and learned selection, no word-to-position incidence |
| `WORD_REAL` | yes | real `o_uv, r_uv+r_vu, |r_uv-r_vu|` | full mechanism |

`WORD_GEOM_NULL` and `WORD_REAL` instantiate exactly the same modules and have
the same parameter count (619,586) and the same explicit initialisation. They
differ only in whether the three geometric coverage inputs are non-zero.
`SENTENCE_BASE` has 416,177 parameters and does not instantiate word modules.
Both null and real compute `alpha`, `z_v` and `Q` on real molecular bonds; the
null arm still computes the coverage matrix internally so the same checkpoint
can be switched between real and zero geometry without changing parameters.

## 8. Training recipe

All arms, no hyperparameter search:

| item | value |
|---|---|
| seed | 0 |
| optimiser | AdamW |
| peak LR | 3e-4 |
| min LR | 1e-5 |
| weight decay | 1e-4 |
| gradient clip | 5.0 |
| batch | 64 molecules |
| schedule | 5 warmup epochs, cosine to 1e-5 |
| max epochs | 240 |
| objective | whole-molecule standardised MSE only |
| evaluation | D0 Dev RMSE / MAE every epoch |
| checkpoint | best Dev RMSE |
| minimum epochs | 80 |
| early-stop patience | 45 |
| auxiliary / distillation / search | none |

Train/Dev predictions, curves, manifests, best and final checkpoints are saved
per arm. The same rule is used for all three arms.

## 9. Required pre-training checks

`analysis/checks.json` must contain all 15 PASS before formal training:

1. all atom/bond attributes match the frozen raw cache;
2. every candidate `S` is a real connected induced support;
3. boundary bonds are exact `S -> R2(v)\S` induced bonds;
4. atom renumbering, edge reordering and candidate reordering do not change the
   prediction;
5. swapping the two endpoints of a bond leaves `q` unchanged;
6. `alpha` sums to 1 for every root with candidates;
7. `c_(v,t)` matches an explicit independent sum over candidates;
8. `o_uv` is a sum over real shared atoms, not an embedding cosine;
9. a synthetic path and star with identical atom/bond counts but different
   connection give different candidates and coverage;
10. `WORD_GEOM_NULL` and `WORD_REAL` have identical initial state and parameter
    count;
11. zeroing the three geometry inputs on one checkpoint matches the real
    forward, and changes the output relative to real geometry;
12. all key modules have finite non-zero gradients;
13. single-molecule and batched predictions agree;
14. official Valid/Test labels are absent from every data tensor;
15. candidate scale, memory and speed are bounded.

## 10. Frozen classification rule

```
delta_word    = RMSE(SENTENCE_BASE) - RMSE(WORD_REAL)
delta_binding = RMSE(WORD_GEOM_NULL) - RMSE(WORD_REAL)
```

Positive means `WORD_REAL` is better. `WORD_REAL` is degenerate if any of:
mean word-representation effective rank < 2; mean q effective rank < 2;
per-root alpha is constant (max-min < 1e-6) for more than 90% of roots; or
mean per-dimension `z` standard deviation < 1e-6.

Apply in this order:

1. **`PROMISING_WORD_ROUTE`** iff
   `delta_word >= 0.02` and `delta_binding >= 0.015` and
   `WORD_REAL Dev RMSE <= 0.62` and not degenerate.
2. **`WORD_CONTENT_ONLY`** iff `delta_word >= 0.02` but
   `delta_binding < 0.015` (prediction improves, but no clear incidence gain).
3. **`NO_COMPETITIVE_WORD_SIGNAL`** iff `delta_word <= 0` or
   (`WORD_REAL Dev RMSE > 0.6167` and `delta_binding < 0.015`).
4. Otherwise **`INCONCLUSIVE_SINGLE_SEED`**.

These are single-seed engineering gates, not significance tests. Paired
molecule bootstrap intervals are descriptive only.

## 11. Budget, outputs and stop rule

Target: at most 2 h wall and 4 GPU-hours, using `rr` and two `res` GPUs, one
process per GPU. The three arms follow the same recipe. Stop after this single
seed. No second seed, D1, official Valid/Test evaluation, K-SVD, additional word
sizes or architecture search.

Required artefacts: `PROTOCOL.md`, `code/`, `configs/`, `tests/`,
`analysis/checks.json`, three best/final checkpoints with curves/manifests/Dev
predictions, `results/summary.json`, and `REPORT.md`.

## 12. Amendments

None before formal training. Pre-training audit decision recorded in §4:
support_max remains 4 because the full candidate count is 1,124,230 (<20M).
