# Stage-0 information-flow audit: current strong ZINC backbone

Scope: the *current* strong ZINC model family that FSAR will replace —
`compact-v4-smallhead` (B-Full geometry) and its `T=2` recurrent
pair–centre variant (`rec.PatchPathRecurrentPairCentreModel`, "A2 / cell A"),
plus the two frozen references `B-Bag` and `FSAB`.

This document is a **blocking gate** for FSAR (see §6). Its purpose is to
enumerate every tensor that reaches the prediction graph and classify it as

* `A` — chemistry/semantic only,
* `S` — topology only,
* `B` — explicit structure↔attribute correspondence,
* `MIXED` — already entangles topology and chemistry.

Only `A`, `S` or a *rewritten* topology-only projection of a path may survive
into FSAR. Every `MIXED` path must be removed or decomposed before any FSAR
training run.

Code of record (local `main`-lineage commit `4b1c5d1`):

* model: `experiments/luyin16/zinc_patch_path_pooling.py` → `PatchPathModel`
  (`__init__` L1574, `encode` L2520, `forward` L2625);
* recurrent core: `experiments/luyin16/zinc_compact_v4_recurrent_pair_centre.py`
  → `PatchPathRecurrentPairCentreModel._encode_core` L461;
* reference geometry: `experiments/luyin16/zinc_compact_v4_smallhead_e2e.py`
  → `_base_kwargs` L314, `build_baseline` L361;
* config: `configs/luyin16/zinc_compact_v4_topology_hinge.yaml`
  (`patch_hidden 48`→ cell A overrides to 64, `pair_hidden 16`,
  `center_context true`, `topology_mode hinge`, `patch_radius 2`,
  `context_radius 0`);
* feature builders: `_shell_descriptor` L431, `_outer_context_descriptor`
  L503, `_typed_certificate` L568, `_pair_relation` L612, `_graph_record` L669,
  `_encode_records` L1027, `global_feature_views`
  (`zinc_long_range_proxy.py`), `zinc_topology_features.compute_features`;
* local-token encoders: `structural_patch_encoder.py`
  (`SharedStructuralPatchEncoder`, `SharedBagPatchEncoder`,
  `AdaptiveStructureBindingEncoder`, `BindingCompositionEncoder`,
  `FactorizedStructureAttributeBindingEncoder`);
* raw graph view: `graph.atom` (OGB atomic-number category, 28 bins),
  `graph.bond` (OGB bond-type category, 4 bins), `graph.root`, `graph.dist`,
  `graph.patch`, `graph.src` / `graph.dst`, `graph.edge_patch`
  (`zinc_long_range_proxy._data_to_graph` / patch extraction).

---

## 1. Path table

`R` = `unified_graph_width` tensor consumed by the sole graph head.
Cell-A geometry: `patch_hidden H=64`, `pair_hidden Q=16`, `token_width 16`,
`center_context_hidden 60`, small head `(13,13)`,
`topology_mode=hinge` (raw width 25), `context_radius=0`,
`structural_context_mode=none`, `attribute_mode=none`,
`direct_token_readout=false`, `patch_representation` ∈
{`typed_lookup`, `shared_bag`, `shared_structural`, `factorized_binding`}.

| # | path | raw inputs actually read | topology? | atom attrs? | bond attrs? | root role? | mixed? | class | destination in `R` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `data.patch_cont` (SHELL_WIDTH 146) | `_shell_descriptor`: shell-wise atom proportions `[3×28]`, shell-pair bond proportions `[6×4]`, root atom one-hot `[28]`, incident-bond composition `[4]`, size/cycle/degree scalars `[6]` | yes (shell assignment) | yes (28 atom hist) | yes (4 bond hist) | yes (root atom one-hot) | **YES** | MIXED | `patch_encoder` input |
| 2 | `data.patch_context` (CONTEXT_WIDTH 42) | `_outer_context_descriptor`: radius-3 atom proportions `[28]`, radius-2/3 and 3/3 bond blocks `[2×4]`, 6 outer-shell scalars | yes | yes | yes | no | **YES** | MIXED | `patch_encoder` input. Width 0 in cell A (`context_radius=0`); the mechanism exists and must stay removed. |
| 3a | `e_patch` via `typed_embedding(data.typed_token)` (`typed_lookup`) | `_typed_certificate` = canonical colored-incidence certificate of the rooted radius-2 patch (atom type + bond type + root + distance) | yes | yes | yes | yes | **YES** | MIXED | `patch_encoder` input |
| 3b | `e_patch` via `SharedStructuralPatchEncoder` | same `struct_*` primitives, edge-aware MPNN over real patch edges | yes | yes | yes | yes | **YES** | MIXED | `patch_encoder` input |
| 3c | `e_patch` via `SharedBagPatchEncoder` | atom type + root + distance + bond type (no adjacency) | bag only | yes | yes | yes | **YES** | MIXED | `patch_encoder` input (frozen B-Bag reference) |
| 3d | `e_patch` via `FactorizedStructureAttributeBindingEncoder` | strict S (root/dist/untyped adjacency) ⊕ strict A marginals ⊕ centered B | split | split | split | in S | no | A+S+B | `patch_encoder` input (frozen FSAB reference; **only** the local token was factorized — see §4) |
| 4 | `parent_embedding(data.parent_token)` (width 8) | `_typed_certificate` at radius `patch_radius−1 = 1` | yes | yes | yes | yes | **YES** | MIXED | `patch_encoder` input |
| 5a | `data.pair_relation` (RELATION_WIDTH 23), fed to `relation_encoder` | distance one-hot `[5]`, log distance `[1]`, patch-overlap/size `[5]` (topology), boundary overlap `[3]` (topology), `path_bond_mean [4]` (chemistry), log path count `[1]`, `adjacent [4]` bond type (chemistry) | partial | no | **yes** | no | **YES** | MIXED | `q` → centre update + pair readout |
| 5b | `data.pair_bucket` (distance bucket 0..4) | shortest-path hop bucket | yes | no | no | no | no | **S** | `distance_gate` → `q` |
| 6 | `data.global_context` (GLOBAL_WIDTH 62) | `global_feature_views()["global_all"]` = structure short + long (topology) ⊕ atom histogram `[28]` ⊕ bond histogram `[4]` (chemistry) | yes | yes | yes | no | **YES** | MIXED | `global_encoder` → `R` |
| 7 | `data.topology_features` (width 25, `hinge`) | `zinc_topology_features.compute_features`: `cycle_rank`, exact cycle spectrum 3..10, MCB stats, `[L, L²]`, `ReLU(L−3..L−10)` — graph only, no atom/bond types | yes | **no** | **no** | no | no | **S** | `topology_encoder` → `R` |
| 8 | `center_context` (DISTANCE_BUCKETS × (2Q+1) = 165) | `_pool_pairs_to_centres(q, source, target, bucket)`: moments of `q` per centre per bucket | inherited from 5 | inherited | inherited | no | **YES** (via 5a) | MIXED | added to `patch` before unary readout |
| 9 | unary readout `_pool_nodes(patch)` / `_pool_nodes(h^T)` | moments of patch/centre states | inherited | inherited | inherited | inherited | **YES** (all upstream) | MIXED | `R` |
| 10 | pair readout `_pool_pairs(q, bucket)` | moments of `q` per distance bucket | inherited | inherited | inherited | inherited | **YES** (via 5a) | MIXED | `R` |
| 11 | `direct_token_readout` (off in cell A) | raw low-rank code of `typed_token` | yes | yes | yes | yes | **YES** | MIXED | `R` |
| 12 | `attribute_encoder` (`attribute_mode`, off in cell A) | `_AttributeEncoder`: `(atom type, topology role)` and `(bond type, role_left, role_right)` | yes | yes | yes | role | **YES** | MIXED | `patch_encoder` input |
| 13 | `structural_context_mode` (`none` in cell A) | typed/coarse ring signature of the centre | yes | yes | yes | maybe | **YES** | MIXED | token conditioning / `patch_encoder` input |

The *only* pure-`S` path currently reaching prediction is #7
(`topology_features`, hinge). #5b (`pair_bucket`) is `S` but its companion
relation tensor #5a is `MIXED`. Everything else is `MIXED`.

There is **no** explicit `B` path in the current strong model: structure and
attribute correspondence is only implicit inside paths #1/#3/#5a/#12. The
frozen FSAB encoder is the only place where a strict `S`/`A`/`B` split exists,
and it was fed **only** through path #3 (`e_patch`) while #1, #4, #5a, #6, #8,
#9, #10 stayed `MIXED` — this is exactly why the FSAB token could be annihilated
without loss (the `MIXED` paths already carried the signal).

---

## 2. Why each `MIXED` path is `MIXED` (raw-tensor trace)

* **#1 `patch_cont`.** `_shell_descriptor` (zpp L431-501) groups
  `node_types` by *root-relative BFS shell* and `edge_types` by *shell pair*,
  then appends the **root atom one-hot** and the **incident-bond composition**.
  It reads `node_types` (chemistry) and `distances`/adjacency (topology)
  jointly. No untangling is possible post-hoc.
* **#2 `patch_context`.** `_outer_context_descriptor` (zpp L503-566) is the
  same construction at radius 3.
* **#3 local token.** All four non-FSAB encoders consume
  `struct_atom` **and** `struct_root`/`struct_dist`/adjacency together, or the
  exact typed certificate (which hashes atom and bond colors into one byte
  string). `B-Bag` additionally pools bond chemistry into the same token.
* **#4 `parent_token`.** A rooted typed certificate at radius 1 — same object
  as #3, smaller radius.
* **#5a `pair_relation`.** `_pair_relation` (zpp L612-667) concatenates
  topology fields (distance, overlap, boundary, path count) with
  `path_bond_mean` (bond chemistry averaged over shortest paths) and the
  `adjacent` bond one-hot. Trace: `bond_sums` in `_shortest_path_summary`
  (zpp L387-428) is built from `edge_types`.
* **#6 `global_context`.** `global_feature_views`
  (`zinc_long_range_proxy.py`) concatenates topology structure blocks with
  `np.bincount(node_types % 28)` and `np.bincount(bond_values % 4)`.
* **#8/#9/#10.** Functions of `q` and `patch`; `q` is a function of #5a, and
  `patch` is a function of #1/#2/#3/#4/#12.
* **#11/#12/#13.** Typed identity / typed role conditioning.

---

## 3. `topology_features` purity check

`zinc_topology_features.compute_features(graph)` reads only
`graph.nodes` / `graph.edges()` through `_to_networkx` (untyped), then exact
simple-cycle enumeration, MCB statistics and a fixed hinge ladder. It never
touches `node_types` or `edge_types`. Confirmed `S`-pure. It is therefore the
one inherited path that may legally feed the FSAR relational substrate /
readout. All three FSAR modes share it identically, so it cannot confound the
A→SA→SAB nested comparison.

---

## 4. Frozen-reference bypass inventory (what FSAB did **not** remove)

| reference | local token | other predictive inputs | consequence |
|---|---|---|---|
| B-Full | typed radius-2 certificate (path #3a) | #1, #4, #5a, #6, #8, #9, #10 | strong mixed backbone |
| B-Bag | connectivity-free bag token (#3c) | #1, #4, #5a, #6, #8, #9, #10 | strong mixed backbone without patch adjacency |
| A2 / cell A | typed radius-2 certificate (#3a) + `T=2` recurrent pair–centre | #1, #4, #5a, #6, #8, #9, #10, #7 | strong mixed backbone + recurrence |
| FSAB | strict S/A/B split → 16D token | **#1, #4, #5a, #6, #8, #9, #10 unchanged**, #7 | factorization confined to one 16D token; everything else still `MIXED`; token annihilated (93.9 % of encoder params exactly zero) |

FSAB is therefore **not** a factorized backbone. It is the old `MIXED` backbone
plus one optional factorized token. FSAR must move the factorization from
"one token" to "the whole node-initialisation and relation substrate", and
delete every `MIXED` path.

---

## 5. FSAR handling plan (exact)

Per the FSAR brief §§19-21:

| path | FSAR v1 handling |
|---|---|
| #1 `patch_cont` | **REMOVE.** Replaced by explicit `A_v / S_v / B_v` (FSAR owns the radius-2 ego extraction). |
| #2 `patch_context` | **REMOVE** (already width 0; the mechanism is deleted, not merely disabled). |
| #3 `e_patch` (non-FSAB variants) | **REMOVE.** Replaced by `z_v = [A_v; S_v; B_v]` (96D) → `h_v^0`. No 16D bottleneck. Mode A uses only `A_v`; mode SA `[A_v;S_v]`; mode SAB `[A_v;S_v;B_v]`. |
| #4 `parent_token` | **REMOVE.** A radius-1 typed certificate is a `MIXED` identity. |
| #5a `pair_relation` | **REWRITE to topology-only.** Keep `[distance one-hot (5); log dist (1); overlap/size (5); boundary (3); log path count (1)]` = 15D. Delete `path_bond_mean` (4) and `adjacent` bond type (4). Tests must assert invariance to atom/bond attribute changes. |
| #5b `pair_bucket` | **KEEP** (`S`-pure shortest-path bucket). |
| #6 `global_context` | **REMOVE.** No global typed histogram, no mixed descriptor, no token counts. |
| #7 `topology_features` (hinge) | **KEEP as `S`.** Shared identically by A/SA/SAB; graph-level topology-only extra. |
| #8 `center_context` | **KEEP** but recomputed from the topology-only `q` (function of rewritten #5a and `h`). |
| #9 unary readout | **KEEP.** `g = Pool(h^T)` = `[mean, std, log1p(count)]`. |
| #10 pair readout | **REMOVE from `g`.** Pair information reaches the readout only through the `T=2` centre updates (`h→q→h→q→h`), so `g` is a function of final node states alone (brief §§19/22). |
| #11 `direct_token_readout` | **REMOVE.** |
| #12 `attribute_encoder` | **REMOVE.** Implicit `(type, role)` conditioning is a `B`-like `MIXED` path; FSAR's explicit `B_v` replaces it. |
| #13 `structural_context_mode` | **REMOVE / keep `none`.** |

`bond` chemistry may enter **only** through `A_v` (context bond marginal) and
`B_v` (edge alignment), never directly into the relation core (brief §21).
`atom` chemistry may enter **only** through `A_v^{self}`, `A_v^{ctx}` and
`B_v^{node}`.

### 5.1 FSAR channel definitions (binding contract)

* `A_v = [A_v^{self}; A_v^{ctx}]`, 32D total.
  * `A_v^{self} = E_atom(x_v)` — centre atom semantics; explicitly allowed to
    know it is the centre (brief §7.1). This is *not* counted as binding.
  * `A_v^{ctx}` = permutation-invariant marginals over `W_v\{v}` atom
    attributes and over all patch bond attributes: `[μ_a, σ_a, μ_e, σ_e] → MLP`.
  * `A_v^{ctx}` must be invariant to any reassignment of attributes to
    neighbourhood positions (no distance, no degree, no root-attachment, no
    `src`/`dst`, no adjacency propagation).
* `S_v` = topology-only state: `r_{vu}^0 = E_root[1(u=v)] + E_dist[d(v,u)] + base`,
  2 rounds of edge-aware message passing on the **untyped** typed-free
  adjacency, then `[r_{vv}, mean_u r_{vu}, std_u r_{vu}] → MLP`, 32D.
  Invariant to all atom/bond attribute changes (exact 0.0).
* `B_v = F_B(B_v^{node}, B_v^{edge})`, 32D.
  * node: center roles and attributes inside the patch, low-rank product of
    the centered projections (no attention).
  * edge: pure-topology edge role `F_role-edge(r_{vu}+r_{vw}, |r_{vu}-r_{vw}|)`
    ⊙ `E_bond(e_{uw})`, both centered, pooled mean/std.
  * `B_v` is the **only** channel allowed to see which chemistry sits at which
    structural role.
  * No orthogonality / HSIC / MI / decorrelation loss.
* `h_v^0 = F_init(z_v)`, `z_v = [A_v; S_v; B_v]`, `96 → 128 → H`.
* Topology-only relation core `h→q→h→q→h`, `T=2`, weight-tied (reuse the
  proven `refresh` semantics). `q = Q(P(h_i), P(h_j), ρ_ij)` with `ρ_ij` the
  rewritten 15D topology-only relation. Centre update via
  `pool_pairs_to_centres`. Readout `g = Pool(h^2)`.
* Head: one shared simple MLP for all three modes.

---

## 6. Blocking rule (brief §4)

FSAR training must **not** start until the following are demonstrated by tests
on `PatchPathFSARModel`:

1. **No mixed bypass** — a forward-path access audit shows every chemistry
   tensor reaches only `A`/`B` and every local-topology tensor only `S`/`B`;
   and no removed path (#1,#2,#3,#4,#5a-chemistry,#6,#10,#11,#12,#13) exists as
   a model attribute or is read in `forward`.
2. **S chemistry invariance** — arbitrary atom/bond attribute changes leave
   `S_v` bit-unchanged.
3. **A assignment invariance** — reassigning attributes among patch nodes /
   edges leaves `A_v` unchanged while centre attribute and multisets are fixed.
4. **B assignment sensitivity** — same topology / centre / multisets but a
   different attribute↔role assignment gives `A_1=A_2`, `S_1=S_2`,
   `B_1≠B_2`.
5. **Relation purity** — changing chemistry with fixed topology leaves all
   relation descriptors bit-unchanged.
6. **Node-relabel + batch invariance**, and **gradient viability** for all of
   `A`, `S`, `B`, relation core, head.

If any gate fails the workflow stops; FSAR is not trained.

### 6.1 Answers to the brief §42 self-checks for the planned design

* **Q1** (can chemistry be recovered without S/A/B?) → **NO**. After removing
   #1,#2,#3,#4,#5a-chemistry,#6,#10,#11,#12,#13 the only remaining inputs are
   `A`/`S`/`B` and topology-only #5b/#7; no chemistry term survives outside
   `A`/`B`.
* **Q2** (can a topology×chemistry mixed descriptor still be read?) → **NO**.
   The rewritten relation is attribute-invariant and `topology_features` is
   untyped by construction.
* **Q3** (do A/SA/SAB differ only by explicit S/B?) → **YES by construction**:
   one model class, one relation core, one head, one optimizer; modes differ
   only in which of `{A_v},{A_v,S_v},{A_v,S_v,B_v}` initialise `h_v^0`.
* **Q4** (does SAB work without any old Full patch token?) → **YES**: no typed
   lookup, no certificate, no `parent_token`, no `patch_cont` exist in the
   module.

---

## 7. Status

* Stage 0 audit: **complete** (this note).
* Mixed-bypass handling plan: **decided** (§5).
* FSAR implementation: authorised to begin once §6 tests are written with the
  implementation.
* Official ZINC test: **never loaded** (`official_test_loaded = false`).
* This audit itself trained nothing and read no test labels.
