# PROTOCOL — attachment_conditioned_response_algebra_pilot_v1 (ACRA-v1)

Status: **frozen before the first formal training run** (commit is recorded in
every run manifest).  Later commits may append timing facts, actual conditions
and explained deviations; the plan itself is not rewritten.

Round: `tracks/structural-sufficiency/attachment_conditioned_response_algebra_pilot_v1/`.
Main task: `ogbg-mollipo`, official scaffold split (Train 3360 / Valid 420 /
Test 420).  **Test labels are never read** by preprocessing, training, model
selection or analysis.

## 1. Question

Can a whole-molecule property be assembled from many reusable *local* structural
responses and their explicit relations, instead of predicting the whole
molecule from every cut separately?

Three fixed arms (seed 0):

| arm | E_atom | E_bond independent | context interaction | E_rel |
|---|---|---|---|---|
| `ACRA_ADD` | yes | yes | no | no |
| `ACRA_COND` | yes | yes | yes | no |
| `ACRA_REL` | yes | yes | yes | yes |

## 2. Local objects (computation path, not names)

For every real undirected bond `{u,v}` exactly two **directed connection
events** are created, `u→v` and `v→u`.  For an event `u→v`:

1. the bond `{u,v}` is removed from the adjacency only; the rest of the
   molecule is untouched;
2. `S` = the radius-1 ball around `u` in `G − {u,v}`, induced subgraph, with
   all original atom (9) and bond (3) attribute columns;
3. `C` = the radius-2 ball around `v` in `G − {u,v}`, induced subgraph, same
   attributes;
4. the removed bond keeps its raw 3-column attributes and its `is_bridge` and
   `in_ring` flags;
5. the root of each ball is kept (`S` root = source atom, `C` root = target
   atom) by pinning it to frame slot 0.

Consequences, all checked in `analysis/checks.json`:

* every atom is used (atom table = raw attributes, `E_atom` sums over all of
  them) and every bond produces exactly two events (`E_bond` averages them);
* ring bonds are **not** cut away: removing a non-bridge bond never
  disconnects the molecule, and `S`/`C` may overlap (verified on short rings);
* no bridge is required for an event to exist; molecules without any bridge
  still have one event per bond direction;
* multiple components: balls are taken on the whole graph, unreachable atoms
  simply stay outside the ball (they remain in the molecule and are covered by
  other events / `E_atom`).  `C` is a *local* ball, not the whole complement;
* single-atom / bond-free graphs: no events; the prediction is
  `beta + E_atom` (and `E_rel = 0`); no crash path.

`S`/`C` are local observations of the same graph, not a partition of it.

## 3. Structural representer

For each local ball (`S` or `C`):

1. enumerate every connected induced subset of 1..4 atoms containing the root
   (all subsets, no truncation, no sampling);
2. canonicalise each object with the parent round's root-fixed frame LUT: the
   root is pinned to slot 0, all permutations of the other slots that attain
   the minimal canonical adjacency bits are kept, so the resulting frame
   multiset is invariant to atom numbering;
3. compress equal frames to **unique frame + multiplicity**.  The key contains
   exactly the encoder's inputs (size `k`, canonical adjacency bits, the 4 raw
   atom rows in canonical slots, the 6 raw bond rows for present pairs), so
   this compression is lossless — no silent truncation, no approximation;
4. a **shared** attribute-aware `FrameEncoder` maps each frame to **64d**
   (`9×emb6 → 32 → 16` per atom slot, `3×emb6 → 16 → 8` per bond slot, frame
   MLP `126 → 128 → 64`, SiLU + LayerNorm).  No learnable fragment-id
   embedding is used; the exact structural identity (coloured canonical
   SMILES) is only used for *support statistics*;
5. for each `k = 1..4`: multiplicity-aware mean pooling of the 64d frame
   vectors, plus `log(1 + count_k)` (counts include multiplicity; 0 when the
   ball has no frame of that size);
6. concatenate the four pooled blocks (256), the four log counts, local
   `log1p(n_atoms)`, `log1p(n_bonds)` and ring attributes
   (`cycle_rank`, `n_ring_bonds`, `ring_bond_frac`, `root_degree`,
   `root_in_ring`, `has_ring`) → 268d;
7. a per-side MLP (`268 → 128 → 64`, SiLU + LayerNorm) gives the local state
   `z_S` (`64d`) or `z_C` (`64d`).  `S` and `C` share the frame encoder and the
   MLP architecture; the two projection heads have separate parameters.

No neural message passing, no GNN backbone, no Graph Transformer, no extra
attention, no pretrained weights.

## 4. Shared context-conditioned response operator

`e_bond` = concat(3 bond field embeddings (8 each), bridge embedding (4), ring
embedding (4)) → 32d.

```
s = MLP_S([z_S, e_bond])          # 2 hidden layers, width 64, SiLU, dropout 0.10
c = MLP_C([z_C, e_bond])          # same architecture, separate parameters
q = Linear_Q(s), k = Linear_K(c)  # rank 32
interaction = dot(q, k) / sqrt(32)
r = scalar_S(s) + scalar_C(c) + interaction
h = SiLU(Linear([s, c]))          # 64d site representation (ACRA_REL only)
```

`ACRA_ADD` uses the same `MLP_S/MLP_C` towers but **without** the interaction
term, i.e. its response is exactly additive in the two sides and cannot
represent a multiplicative context effect.  `ACRA_COND` adds the interaction.
Branch modules that an arm does not use are not instantiated; parameter counts
are reported as-is and are not claimed to be capacity-matched.

The scalar `r` is a **local response**, never a whole-molecule label.

## 5. Molecule-level composition

```
E_atom  = sum_v atom_contribution(v)            # 9 field embeddings -> MLP -> scalar
E_bond  = sum_{(u,v)} 0.5 * (r(u→v) + r(v→u))
yhat_std = beta + E_atom + E_bond + E_rel
yhat     = yhat_std * train_std + train_mean
```

`E_rel` (only `ACRA_REL`): for every unordered bond pair `e < f`

```
pair(e,f) = 0.5*(dot(Q h_e, K h_f) + dot(Q h_f, K h_e)) / sqrt(32)
            + relation_bias[bucket(e,f), shared_ring(e,f)]
h_e       = 0.5*(h(u→v) + h(v→u))
E_rel     = sigmoid(gate_raw) * sum_{e<f} pair(e,f) / max(1, m),
            gate_raw initialised so sigmoid(gate) = 0.12
```

* `bucket(e,f)`: the shortest path length in the **line graph** (adjacent
  bonds = 1), which equals `1 + min endpoint graph distance`; mapped to
  `1,2,3,4+`.  Unreachable (different components) → `4+`.  Two distinct bonds
  can never be at distance 0, so the four buckets match the brief.
* `shared_ring(e,f)`: the two bonds lie on a **common cycle**, decided exactly
  as "same vertex-biconnected block and the block is not a bridge" (a cycle is
  2-connected; a block with ≥2 edges is 2-connected, and any two edges of a
  2-connected graph lie on a common cycle).  This is checked on a path, a
  4-cycle and a figure-eight graph.
* `E_rel = 0` when a molecule has fewer than two bonds; no global Transformer
  and no iterative message passing are introduced.

`h` is only present in `ACRA_REL`; `ACRA_ADD`/`ACRA_COND` have `E_rel = 0` by
construction (and their `L_rel` is 0).

The final prediction is exactly the sum of the three composed terms (checked
to bitwise equality in the test suite).

## 6. Losses

Only whole-molecule labels supervise the composition:

```
L_task = mean_G (yhat_std - y_std)^2
L_site = mean_G mean_{sites of G} scalar_response^2,
         sites = atoms (atom contribution) and directed events (r)
L_rel  = mean_G E_rel^2
L      = L_task + 0.001 * L_site + 0.01 * L_rel
```

No cut-wise loss, no CUTRESP/auxiliary supervision, no requirement that a local
response equal the molecule label.  The same rule is used in all three arms;
`L_rel` is identically 0 without the relation term.  Cancellation between
positive and negative responses is monitored (`analysis/contribution_diagnostics.json`).

## 7. Data and recipe

* official raw cache `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`
  (sha256 `fe002287…123c7c`), official Train/Valid/Test split;
* label standardisation with **Train-only** mean/std;
* atom/bond category support: fixed official feature dimensions, categories
  absent from Train are zeroed (embedding rows masked to 0); Valid/Test labels
  and features never fit anything;
* structure cache is label-free and built on CPU for all molecules (the test
  suite verifies that replacing every label with NaN leaves the cache
  bit-identical);
* seed 0 everywhere; AdamW; peak LR 1e-3; weight decay 1e-4; gradient clip 5.0;
  5 warmup epochs then cosine decay to 1e-5; batch 64 molecules; at most 160
  epochs; best-by-Valid whole-molecule RMSE (Valid labels only);
* no hyperparameter search.  If a GPU timing smoke shows the 4 A100-hour budget
  is insufficient, one **shared** smaller horizon is fixed for all arms before
  the formal runs (recorded in `analysis/horizon.json`) — never a per-model
  horizon.

## 8. Reference points (read-only)

* `TRC_FACTOR_CUTRESP` and `GINE_RAW_REF` prediction records from the parent
  round: Valid RMSE read from the files, never retrained, never used as a model
  component or residual.
* Light non-GNN reference implemented here: Morgan radius-2 **count**
  fingerprint (2048 bits) + Ridge with `alpha = 10`, fit on Train only,
  evaluated on Valid.  This is an external prediction reference, not an ACRA
  ablation.

## 9. Analyses (fixed rules)

1. **Structural support** (`analysis/structural_support.json`): for every
   directed event, exact colour-canonical identities of `S`, of `C`, and of the
   `(S, C, cut bond attributes)` combination; "seen" is defined by Train events
   only.  Reported overall, split into bridge / non-bridge and ring /
   non-ring bonds, with occurrence-count histograms (Train) and Train/Valid
   reuse rates.  No Test structural statistics are reported.
2. **Mechanism** (`analysis/contribution_diagnostics.json`): scale of
   `E_atom`, `E_bond`, `E_rel`, their share of the composed prediction,
   cancellation ratio, per-branch gradient norms (after training), signed
   response distribution, whether the interaction term is actually used (its
   magnitude and the prediction change when it is removed), and response size
   vs. Train support frequency of the event's local identity.
3. **Relational capacity** (`analysis/relational_capacity.json`): real
   molecules grouped by the exact multiset of their per-event local identities,
   plus an exhaustive search over small synthetic graphs (all connected graphs
   on 6 and 7 vertices with one common atom/bond attribute pattern) for pairs
   with identical local multiset but different bond-pair relation profiles;
   the `ACRA_REL` relation term is compared on such pairs.  Structural
   capability only, not task performance.
4. **Real matched pairs** (`analysis/pair_delta.json`): the parent round's real
   replacement directory (`pair_mol_a/b`, same exact context).  Only
   `predicted_delta = yhat(G_b) - yhat(G_a)` of the full model is evaluated
   (delta MAE/RMSE/direction accuracy), scopes Train×Train, Train×Valid (cluster
   bootstrap by the Valid molecule), Valid×Valid (15 pairs, descriptive only).
   Test-touching pairs are dropped before reading anything.  For representative
   pairs a fixed rule (largest, median and smallest `|Δy|` within the most
   populated context groups) reports the exact algebraic split
   `Δyhat = ΔE_atom + ΔE_bond + ΔE_rel`; these are model algebra, not chemical
   causal contributions.
5. **Failure cases** (`analysis/cases.json`): deterministic top-3 lists (ties by
   ascending row id) for "COND better than ADD", "COND worst than ADD", "REL
   better than COND", "REL worst than COND" on Valid molecules.  No manual
   cherry-picking.
6. Valid molecule-level paired bootstrap (2000 reps) for the current
   checkpoints; it quantifies evaluation-molecule uncertainty only, not
   training-seed uncertainty (single seed 0).  A bootstrap crossing zero is
   never reported as "the models are equivalent".

## 10. Engineering checks (before formal training)

All 13 items of the brief are exercised by `tests/test_acra.py` through the
real computation path (independent BFS, brute-force frame enumeration, pack
rebuilds under permutations, batch/single comparison, exact decomposition,
pair symmetry, gradient coverage, label poisoning, finiteness) and summarised
in `analysis/checks.json`.  The check run and the test run are committed before
the first formal training.

## 11. Execution and budget

* local: implementation, tests, cache, checks, support statistics, Ridge
  reference, all analysis;
* remote: three ACRA runs via `rr` on `res` (`res-gpu0` / `res-gpu1`), one seed
  each, at most 4 A100 GPU-hours total, at most two GPUs in parallel;
* provenance: commit, code/config hashes, cache/raw sha256, environment
  (hostname, driver, torch/CUDA, GPU), curves, checkpoints, `rr` run metadata.

## 12. Ambiguities and resolutions (fixed here)

* **"最短端点距离" with buckets 1,2,3,4+**: adjacent bonds share an endpoint
  (endpoint distance 0), so the bucket is the line-graph distance
  `1 + min endpoint distance`, which is exactly `1,2,3,4+`.
* **"MLP 深度 2"**: two hidden linear layers of width 64 (SiLU, dropout 0.10
  after each), producing 64d.
* **"context interaction: no"**: the bilinear interaction term is removed; the
  two tower MLPs remain, so the ablation isolates the multiplicative term.
* **"shared ring"**: exact common-cycle definition above (biconnected block).
* **frame multiplicity**: the parent's "all winning root-fixed permutations"
  mechanism is kept, so automorphic frames contribute with their permutation
  multiplicity; this is what makes the encoder truly permutation invariant.
* **relation normaliser**: `max(1, m)` with `m` the number of undirected bonds
  (not directed events).

## 13. Decision rule (pre-registered)

Primary: Valid whole-molecule RMSE and MAE.  Observe (a) the absolute level of
`ACRA_ADD`, (b) whether `ACRA_COND` improves over `ACRA_ADD`, (c) whether
`ACRA_REL` improves over `ACRA_COND`, (d) the gap to the TRC / GINE reference
records, (e) Train–Valid generalisation gap, (f) matched-pair delta
generalisation.  If only the additive arm works, do not add complex operators;
if only the relation term helps, study structural composition; if the
interaction helps both whole-molecule and cross-molecule deltas, study response
regularities; if everything fails, report the concrete local-model error modes
and stop — no GNN replacement, no extra model.

## 14. Timing / deviations (appended after the runs)

(to be filled with the actual horizon decision, GPU hours and any deviation)