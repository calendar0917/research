# BBPA-v0 — bridge-block port algebra pilot — protocol

Round: `bridge_block_port_algebra_pilot_v0`
Version: 2026-11-23/v1, **frozen before formal training**
Task: `ogbg-mollipo` whole-molecule lipophilicity regression.
Primary metric: whole-molecule **Dev RMSE in original label units**.
Seed: only `seed=0`.
Models: exactly three arms, `BLOCK_BAG`, `LOCAL_PORT`, `TREE_PORT`.

This protocol was written before any formal GPU run of this round. Local CPU
smoke runs used to verify plumbing are engineering tests and are not reported as
results. Any later change is append-only, dated, and recorded in §14.

## 1. Question

Does recursive composition of real structural blocks along the real bridge-bond
connection tree improve whole-molecule prediction, compared with using only the
adjacent block's local port representation, when the atom/bond/block encoders,
parameters, training recipe and graph readout are held identical?

The primary contrast is

```
Δ_comp = RMSE(LOCAL_PORT) − RMSE(TREE_PORT)
```

so a **positive** Δ_comp means recursive tree composition is better.

Secondary questions: does the computation really use connection interfaces and
deeper subtrees; does it distinguish same-composition / different-connection
isomers; does it collapse to a local or global-count model; and is this route
anywhere near ACRA / GINE / the path-count reference?

## 2. Data, split and label isolation

| item | value |
|---|---|
| raw structure cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`, sha256 `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| raw members read | `node_ptr, bond_ptr, atom_feat, bond_feat, bond_u, bond_v, degree, min_ring, in_cycle, graph_n_*, smiles, row_ids, train_idx, valid_idx` — **no `labels` member** |
| Train labels | `tracks/structural-sufficiency/structural_path_composition_route_v1/data/labels_official_train.npz`, sha256 `a44d29deba1fd36e885feee27e788d1fda0247c01f69565c2d30e9c7251114c2` |
| D0 split | `tracks/structural-sufficiency/acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json`, file sha256 `675289f6f997008d44fdabb4ab65350459fe5bccc5e1e5703e31e31fd794e845` |
| D0 split identity | recomputed `sha256_json({train_sub_rows, dev_rows, seed}) = 60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` |
| fit | D0 `train_sub_rows`: **2856** molecules |
| Dev | D0 `dev_rows`: **504** molecules |
| official Valid | structure not used for training; **labels never loaded** |
| official Test | **never touched** |

The training object carries finite labels only on official Train rows. The D0
fit and Dev rows are disjoint and cover exactly the 3360 official Train rows.
Target normalisation is the fit-only z-score:
`mean=2.185815826311436`, `std=1.2139664740467941` (n=2856). All reported
metrics are inverted back to original label units. No Dev label ever enters a
gradient, scaler, category support, vocabulary, or model selection other than
best-Dev checkpoint selection.

Categorical support (which embedding rows may be non-zero) is computed from the
D0 fit rows only; unsupported rows are zeroed. Dev has zero unseen atom/bond
positions under this support.

## 3. Discrete structure: bridge blocks and ports

For each real molecule graph `(atoms, typed bonds)`:

1. Find every **bridge bond** (Tarjan; iterative, edge-id aware, multigraph
   safe) on the raw undirected graph.
2. Delete all bridge bonds. Each remaining connected component is one
   **structural block** `B_i`.
3. A block keeps all its real atoms, all its internal (non-bridge) bonds, and
   its internal graph. Each bridge keeps its original bond attributes and its
   two real endpoint atoms.
4. Contract every block to a node. Bridge bonds become the edges of a **tree or
   forest**. Multiple connected components are processed separately, then
   pooled at the molecule level.

Every `i` has `u_i^global` and, for each incident bridge to neighbour `j`,
`u_(i|j)`. The port atom is a real molecule-local atom index, used only to
locate internal distances and support sets, never as a learnable id.

Required invariants, checked on every D0 molecule:

* each atom belongs to exactly one block;
* each non-bridge bond belongs to exactly one block;
* each bridge bond belongs to exactly one block–block interface;
* expanding all blocks and bridges reconstructs every atom and bond exactly once;
* for every bridge, the two sides after removing that bridge are disjoint and
  their union is exactly the atom set of that connected component;
* `#bridges = #blocks − #components` (forest);
* decomposition is independent of atom numbering and bond-list order.

## 4. Encoders (no message passing on the original graph)

Shared atom encoder: one embedding per raw OGB atom categorical field, summed
and divided by `sqrt(9)`, state width **64**. Shared bond encoder: one embedding
per raw OGB bond categorical field, summed and divided by `sqrt(3)`, state width
**32**. No raw degree / ring / hydrogen feature is used as a learned id.

For a block `B_i`:

* sums / means / standard deviations of all internal atom embeddings;
* sums / means of all internal bond embeddings;
* every internal bond's two endpoint atoms are combined as
  `[a_u+a_v, |a_u−a_v|, a_u*a_v]`, passed through one shared pair MLP, then
  pooled by sum and mean;
* atom count, internal-bond count, cycle rank `b−a+1`, bridge-port count
  (log-scaled and linear).

This gives `u_i^global` (width **96**, shared block MLP).

For a port `p_ij`:

* `u_i^global`;
* the port atom embedding;
* distance-bucketed internal atom and bond sums and counts, with fixed buckets
  `0,1,2,3,4,5+` using true shortest distances in the block's internal graph
  (never across a bridge). A bond is assigned to the closer endpoint's bucket;
* port-atom internal degree, number of ports at that atom, block port count.

This gives `u_(i|j)` (width **96**, shared port MLP). The same block at
different ports can therefore have different representations.

Recommended widths are exactly the frozen widths: atom 64, bond 32, block 96,
directed subtree 96, interface 96. MLP hidden widths: block 128, port 128,
message 96, update 96, Q 96, readout 128→64. Dropout 0.1 in the block/port
encoders and readout; no dropout in the shared composition operators. Exact
parameter count is recorded; target range 100k–500k.

## 5. Shared connection algebra

For a directed block-tree edge `i→j`, `s_(i→j)` is the representation of the
whole `i`-side subtree facing the interface at `j`.

For each neighbour `k ≠ j` of `i`, with bridge bond `ki`, port atoms
`p_ij, p_ik`, and internal distance `d_i(p_ij,p_ik)`:

```
m_(k→i|j) = M( s_(k→i), bond_ki, port_features_i(k), dist_bucket(d_i(p_ij,p_ik)) )
A_(i|j)   = [ sum(m), mean(m), std(m), log1p(count) ]
s_(i→j)   = LN( u_(i|j) + U([u_(i|j), A_(i|j)]) )
```

`M` and `U` are small SiLU MLPs shared over every block, depth and direction.
If `i` has no neighbour other than `j` (leaf), `s_(i→j) = u_(i|j)`. The
aggregation is unordered. Distance is bucketed as `0,1,2,3,4,5+`. Single-atom
blocks give distance 0 between all their ports; ring blocks distinguish nearby
and far substitution positions.

All directed states are computed by two tree-DP passes (post-order child→parent,
then pre-order parent→child), using one shared parameter set. The scheduling
root is arbitrary; predictions must not depend on it (checked explicitly).
No attention, no softmax top-k, no node-id embeddings, no per-layer message
passing on the original atom graph.

For every undirected bridge `e=(i,j)`:

```
q_e = Q( s_(i→j)+s_(j→i), |s_(i→j)−s_(j→i)|, s_(i→j)*s_(j→i), bond_e )
```

`Q` is one shared MLP and the input is symmetric under swapping `i,j`.

The molecule readout uses:

* mean and std of all `q_e` over bridges (exact zeros when there is no bridge);
* mean and std of all `u_i^global`;
* bridge count, block count, atom count, bond count, cycle rank, `log1p` sizes;
* full-graph atom/bond embedding sums (scaled), means, stds.

These go through one shared nonlinear graph readout MLP to the standardised
label. The readout is **not** a strict per-block scalar sum.

## 6. The three arms

All three instantiate the **same modules, shapes, explicit initialisation and
parameter count**. The only difference is the forward computation.

* **`BLOCK_BAG`** — no subtree states, no interface `q`. Uses block-global
  summaries and the public whole-graph inputs. Structural-object baseline; not
  parameter-matched in the sense of using every module.
* **`LOCAL_PORT`** — same modules and port inputs as `TREE_PORT`. In the
  message, the neighbour contribution is `u_(k|i)` instead of the recursive
  `s_(k→i)`. Same `U`, same `Q`, same readout, same training recipe.
* **`TREE_PORT`** — full bidirectional tree DP of §5.

`LOCAL_PORT` vs `TREE_PORT` is the primary controlled comparison. Both record
parameter counts and identical-initialisation module hashes.

## 7. Training recipe (identical across arms)

| item | value |
|---|---|
| objective | whole-molecule standardised MSE only |
| optimizer | AdamW |
| peak LR / min LR | `1e-3` / `1e-5` |
| weight decay | `1e-4` |
| gradient clip | 5.0 |
| batch | 64 molecules |
| warmup | 5 epochs |
| schedule | cosine over the 180-epoch horizon |
| max epochs | 180 |
| eval | fit and Dev every epoch; best-Dev checkpoint |
| early stop | patience 35 after a minimum of 80 epochs, identical for all arms |
| seeds | 0 only |
| auxiliary targets / distillation | none |
| hyper-parameter search | none |

Training batch order is generated from `numpy.default_rng(seed + 101)` and is
identical across arms. Model initialisation is an explicit function of the seed
and is identical across arms. The Dev set is used only to select the best
checkpoint; it is never trained on.

## 8. Engineering checks (must pass before formal training)

`code/bbpa_checks.py` writes `analysis/checks.json`. Required checks:

1. split / raw-cache / row identity / train-only normalisation / no Valid-Test
   label leakage;
2. atom renumbering and bond order invariance (prediction, fp32 tol `1e-5`);
3. block order and neighbour traversal order invariance;
4. batch vs single-molecule prediction invariance;
5. single block, single bridge, four-block chain, star, ring with ports, and a
   two-component forest decompose correctly;
6. on a four-block chain, randomly initialised `TREE_PORT` and `LOCAL_PORT`
   have non-zero state/interface computation difference;
7. on a star (centre + leaf blocks), the two coincide exactly, and this special
   case is recorded without manufacturing a difference;
8. ortho/meta/para disubstituted benzene gives port distances 1/2/3, different
   port inputs, and distinguishable predictions;
9. every bridge's two sides are complementary and non-overlapping;
10. finite forward/backward and non-zero gradients in every used key module;
11. no cross-molecule index confusion; no label leakage;
12. real-molecule bridge coverage, block-size distribution, block-tree diameter,
    no-bridge fraction;
13. learned TREE state variation / near-constant-channel diagnostics;
14. DP schedule root-choice invariance.

A real graph or gradient error is fixed, not bypassed.

## 9. Evaluation and decision rule

Main table columns:
`model, seed, params, train_rmse_at_best, dev_rmse, dev_mae, best_epoch,
epochs_run, wall_seconds, gpu_hours`.

Primary contrast: `Δ_comp = RMSE(LOCAL_PORT) − RMSE(TREE_PORT)`.
Supplemental only:

* paired molecule bootstrap CI for the RMSE and MAE differences;
* error by bridge count and block-tree diameter;
* theoretical TREE-vs-LOCAL difference coverage (block-tree diameter ≥ 3;
  diameter-2 stars are a documented coincidence case);
* TREE state variance, effective rank, near-constant risk;
* worst 10 molecules for TREE;
* historical D0 coordinates (same D0 split, original units, different recipes):
  `GINE_REF 0.5266`, `ACRA_REF 0.5802`, `PATH_COUNT_RIDGE 0.6167`,
  `PSC_POOL_BAG_PAIR 0.6871`.

Route labels (single-seed engineering screen, not statistical significance):

* **PROMISING_PILOT** iff `Δ_comp ≥ 0.025`, `TREE Dev RMSE ≤ 0.64`,
  `TREE Dev MAE ≤ LOCAL Dev MAE + 0.01`, and TREE states are not
  near-constant.
* **NO_COMPOSITION_GAIN** iff `Δ_comp ≤ 0.01` and `TREE Dev RMSE > 0.64`.
* **CAPACITY_WITHOUT_COMPETITIVENESS** iff `Δ_comp > 0.01` and
  `TREE Dev RMSE > 0.64`.
* otherwise **INCONCLUSIVE_SINGLE_SEED**.

The rules are applied in the order above. “Weak range” is defined as
`TREE Dev RMSE > 0.64` (not reaching the competitive threshold). No threshold
is moved after seeing the results.

## 10. Scope and prohibited extensions

Out of scope: GNN / MPNN / Graph Transformer backbones, attention, learned
fragment ids, K-SVD / sparse dictionaries, extra pretraining, pair or
distillation losses, hyper-parameter or width search, word-boundary learning,
new decomposition depths, D1, second seed, official Valid/Test evaluation, and
any follow-up architecture generation. The round stops after this single-seed
screen and its report.

## 11. Resources

Target: ≤ 2 h wall clock and ≤ 4 GPU-hours. `res` has two A100s; at most one
independent training process per GPU. Actual wall and GPU time are measured and
recorded per manifest. If the budget is exceeded, completed runs, facts and the
blocking reason are kept; the model or research definition is not shrunk to fit.

## 12. Deliverables

`PROTOCOL.md`, `code/`, `configs/`, `tests/`, `analysis/checks.json`,
per-model checkpoints/curves/predictions/manifests, `results/summary.json`,
`REPORT.md`.

## 13. Reporting boundaries

The report distinguishes:

* **functional verification** — the computation really uses interfaces and
  subtrees;
* **predictive gain** — whether that lowers Dev error;
* **scientific claim** — a single seed supports only an exploratory judgement.

## 14. Deviations (append-only)

* 2026-11-23: local CPU smoke runs (`_smoke_*`) were used only to verify
  plumbing before the frozen GPU runs; they are excluded from all results.
* 2026-11-23: added an explicit root-choice rescheduling check (`6b`) after the
  decomposition tests, before formal training. It changes no model, data,
  recipe or decision rule.