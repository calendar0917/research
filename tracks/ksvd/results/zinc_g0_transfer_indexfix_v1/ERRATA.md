# ERRATA — ZINC G0 transfer audit, edge-key index fix (v1)

This errata corrects a **measurement bug** in the previous audit
(`tracks/ksvd/results/zinc_g0_transfer_audit_v1/`, commit `2284ca1`).  The old
directory is left byte-for-byte unchanged; its hashes are recorded in
`identity.json`.  The corrected recompute lives in
`tracks/ksvd/results/zinc_g0_transfer_indexfix_v1/`.

## E1 — the located bug: missing per-graph endpoint offset

Old `part_c_audit.py::compute_model_keys` concatenated every molecule's nodes
into one global array and stored per-molecule local edges, but indexed the global
arrays with the **local** endpoint ids:

```python
for mi, elist in enumerate(g["edges"]):
    for u, v, ty in elist:
        mu, mv = int(node_mask[u]), int(node_mask[v])      # unoffset (wrong)
        du, dv = node_desc[u], node_desc[v]
```

`u, v` are graph-local ids in `[0, n_i)`, so every molecule's edges were mapped
onto the nodes of the earliest molecules.  The fix uses the graph's node offset:

```python
node_base = int(g["offsets"][mi])
gu, gv = node_base + int(u), node_base + int(v)
mu, mv = int(node_mask[gu]), int(node_mask[gv])
du, dv = node_desc[gu], node_desc[gv]
```

`offsets` has length `n+1` with `offsets[mi]` = start of molecule `mi`
(`offsets[1:] = cumsum(counts)`, `offsets[0] = 0`) — verified before use.

The same fix is applied to **both** the residual-support key and the full
`dict_phi[65]` descriptor key.  The semantic (bond/atom category) keys, the
undirected endpoint ordering and the descriptor round/hash rule are unchanged.

**This is an audit-code bug, not a model-training bug.**  The training path is
correct: `e2e_dictenv_p1.env_collate` builds the PyG `Batch`, takes
`base.ptr`, and calls `_offset_occurrences` to add `ptr[mi]` to
`env_bond_u`/`env_bond_v` (and the `env_occ_*` fields) — see
`e2e_dictenv_p1.py:561-587`.  No training code was modified.

## E2 — what changed (old → new)

Edge structural keys, per fold (cardinalities over the training fold):

| quantity | old (broken) | new A | new B |
|---|---:|---:|---:|
| unique edge support keys | 169 | **1037** | **1095** |
| unique edge descriptor keys | 207 | **2123** | **2123** |
| unique edge joint keys | 317 | **1257** | **1317** |

Held-out class fractions (edge view, residual-support key, meta G0):

| fold | joint_seen old→new | `joint_rare_marginals_seen` old→new | structure_rare old→new |
|---|---|---|---|
| A | 0.99849 → 0.98884 | 0.00080 → **0.00175** | 0.00071 → **0.00941** |
| B | 0.99830 → 0.98534 | 0.00114 → **0.00193** | 0.00056 → **0.01272** |

Edge view, full-descriptor key, meta G0 (structure_rare):

| fold | structure_rare old→new | `joint_rare_marginals_seen` old→new |
|---|---|---|
| A | 0.00080 → **0.01936** | 0.00114 → **0.00355** |
| B | 0.00068 → **0.02242** | 0.00131 → **0.00353** |

The corrected keys are ~6–7× more numerous, and `structure_rare` is ~13×
(residual support) to ~28× (full descriptor) larger than the buggy value.  The
main exposure `joint_rare_marginals_seen` roughly doubles but stays ≈0.2 %
(residual) / ≈0.35 % (descriptor) per molecule.

**Unchanged:** the high group (`≥0.25`) is still **empty** in every fold × view ×
key × split (0 molecules), matching still yields **0 pairs**, and the low group
is still ~3700–3800 of ~3850 graphs.  So the corrected recompute confirms the
previous *high-group/empty-matching* statement, but the old "structure_rare is
essentially unpopulated" characterization was an artefact and must be withdrawn.

Full table: `before_after.csv`; raw: `coverage_fixed.json`,
`class_composition_fixed.json`, `matching_fixed.json`.

## E3 — invariants that certify the fix (all pass, `index_invariants.json`)

1. **Endpoint membership:** all 249,279 real undirected edges have
   `0 ≤ u,v < n_i`, and `offsets[mi]+u`, `offsets[mi]+v` lie inside that
   molecule's global node interval; edge count and bond categories unchanged.
2. **Minimal counterexample:** two 2-node graphs, both with local edge `(0,1)`,
   supports `[1,2]` and `[4,8]`, distinct descriptors.  The broken path returns
   the *same* key twice (`[1,2]`, desc `61,62`); the fixed path returns `[1,2]`
   and `[4,8]` with distinct descriptors.
3. **Independent reference:** for every fold, slicing each graph's node arrays by
   `offsets` and indexing locally reproduces all 249,279 support, descriptor and
   semantic edge keys exactly.  Additionally, 9 non-first graphs (seeded
   `20261003`, sizes 18–37) encode identically per-graph and concatenated
   (support equality on every node).
4. **Molecular-order invariance:** shuffling graph order (seed `20261003`) and
   rebuilding concatenation/offsets/keys reproduces each original graph's edge
   key multiset (support and descriptor) for all 10,000 graphs, and the coverage
   classification matches on the sampled graphs, aligning by stable original id.

## E4 — node keys unaffected

The node view does not use `offsets`, so the recompute reproduces the old
node-view per-molecule rows **byte-identically** (`node_key_invariance.json`:
`node_all_identical = true`; all edge rows differ).  This is the expected "no
collateral change" check: only edge-derived quantities moved.

## E5 — Part A / Part B not touched

Part A (corrected paired bootstrap) and Part B (frozen class-median shrinkage)
do not depend on the edge index; neither was refit or recomputed.  Their files
are unchanged (`part_a_paired_bootstrap.json` sha256 `a77cd831…`,
`part_b_class_median_shrinkage.json` sha256 `eb435a69…`, matching the values
recorded before this task).