# REPORT — ZINC G0 transfer audit, edge-key index fix v1

**Scope.** Fix one located audit-code bug (missing per-graph endpoint offset in
the edge keys), prove the fix with invariants, and recompute Part C under the
**original frozen protocol**.  Local CPU only (≤8 threads, GPU=0).  No training,
no optimizer/backward, no new head/tree fit, no support perturbation, no
hyper-parameter search, no remote job.  Official test never
instantiated/loaded/evaluated; official-valid not loaded/re-read/re-evaluated.

## Phase 0 — inputs and read/write boundary

* Source models/split: `zinc_topology_crossfit_diagnostic_v1` (result `8134630`,
  science `240bd2b`); the two **soup** states `F_A_state.pt` / `F_B_state.pt`
  (not `last_state`), each with its own fold fit blob.  Hashes in `identity.json`.
* Old audit (`zinc_g0_transfer_audit_v1`, commit `2284ca1`) left byte-for-byte
  unchanged; hashes recorded and re-verified.
* **Train-only loading:** `load_split("train")` internally deserialises
  `encoded_valid.pt` (via `zinc_static_dictionary_pair.load_encoded`), so it was
  **not** called.  This round loads only `encoded_train.pt` + the train env cache
  (`attach_env("train")`).  Topology features come from the graph-only cache CSV.
* No new channel-health forward was run; the old health result is referenced
  read-only (`zinc_g0_transfer_audit_v1/part_c_channel_health.json`).

## Phase 1 — fix proved by invariants (`index_invariants.json`, all pass)

| invariant | result |
|---|---|
| endpoint membership (249,279 edges) | pass; all `0 ≤ u,v < n_i`, offset endpoints inside molecule interval |
| minimal counterexample (2×2-node graphs) | broken → same key twice `[1,2]`; fixed → `[1,2]`/`[4,8]`, distinct descriptors |
| independent slice-then-local reference | pass; 249,279/249,279 support + descriptor + semantic keys match, per fold |
| per-graph vs concatenated encoding | pass; 9 non-first graphs (sizes 18–37), all support masks equal |
| molecular-order invariance (seed 20261003) | pass; all 10,000 graphs key multisets + sampled classifications match by stable id |
| node keys unaffected | pass; node-view per-molecule rows byte-identical to old |

Production recompute calls the **same** `edge_keys_for_graph` function that the
counterexample and reference tests exercise; there is no second implementation.

## Phase 2 — recomputed Part C (protocol unchanged)

Only the two frozen structure keys, edge primary / node secondary, frequencies =
distinct canonical training molecules, seen ≥5, the five mutually exclusive
classes, high `≥0.25` / low `≤0.05`, and the fixed covariate matching
(cycle_rank equal, Δnodes ≤2, Δedges ≤3, Δbase-pred ≤0.25, Δstructure_rare ≤0.10,
no replacement, one pair per canonical group) are used unchanged.

### Old → new (edge view; `before_after.csv` has all 18 rows)

| fold | key | split | unique edge keys (sup / desc / joint) old→new | jrm old→new | structure_rare old→new |
|---|---|---|---:|---:|---:|
| A | residual | meta | 169 / 207 / 317 → 1037 / 2123 / 1257 | 0.00080 → 0.00175 | 0.00071 → 0.00941 |
| B | residual | meta | 169 / 207 / 317 → 1095 / 2123 / 1317 | 0.00114 → 0.00193 | 0.00056 → 0.01272 |
| A | descriptor | meta | — / — / — → 1037 / 2123 / 1257 | 0.00114 → 0.00355 | 0.00080 → 0.01936 |
| B | descriptor | meta | — / — / — → 1095 / 2123 / 1317 | 0.00131 → 0.00353 | 0.00068 → 0.02242 |

outer-dev cells are numerically similar (see CSV).  Node view is **identical**
to the old values in every cell.

### Group counts and matching (edge view)

| fold | key | split | high n (≥0.25) | mid n | low n (≤0.05) | matchable pairs |
|---|---|---:|---:|---:|---:|---:|
| A | residual | meta | **0** | 31 | 3820 | **0** |
| B | residual | meta | **0** | 38 | 3813 | **0** |
| A | descriptor | meta | **0** | 67 | 3784 | **0** |
| B | descriptor | meta | **0** | 67 | 3770 | **0** |

The **high group remains empty** in every fold × view × key × split, so the
fixed covariate matching returns **0 pairs** and the matched
`MAE_high − MAE_low` is **not estimable** — reported as "cannot be estimated",
with thresholds left unchanged.

## Phase 3 — interpretation boundary

* **Confirmed after the fix:** the primary exposure `joint_rare_marginals_seen`
  is still ~0.2 % (residual) / ~0.35 % (descriptor) per molecule and the `≥0.25`
  high group is still empty; matching is still not estimable.  This is the
  corrected evidence for the "no high group" statement, replacing the broken-key
  basis.
* **Withdrawn:** the old characterization "coverage saturated / structure_rare
  essentially unpopulated".  With correct indices, `structure_rare` is ~0.9–1.3 %
  (residual) and ~1.8–2.2 % (descriptor) of molecule-edges — real but still far
  from the 25 % high cut.  The proxy still has **no resolution** for the
  pre-registered contrast.
* `dict_phi` with 1182 unique rows is a statement about repeated local-structure
  descriptors in the frozen input; it does **not** by itself imply the learned
  representation collapsed.
* The IHT top-8 is a setting; the support-set count is not the number of full
  code vectors and not the total structural information.  Coefficient magnitude,
  sign and the common coordinate are **not** retained by the support key.
* Node-dead / F_B-edge-near-constant are separate mechanism observations from the
  old health check; its sampling scope (a small batch) must be quoted and is not
  upgraded to a full-G0 causal claim.  Cross-fold-consistent input coverage is
  not the same as a functional replication of a healthy channel.
* This round only fixed the measurement and estimated an association.  It does
  **not** license "fusion is useless", "the representation is sufficient", or
  "regularisation must be changed", and mean-flatness of some frozen forward
  perturbation would not, by itself, buy a representation/fusion/WD change.

### Evidence grade

1. **Mechanism/code (solid):** the bug and fix, the training path's
   `Batch.ptr` offsetting, the four invariants, node-key invariance.
2. **Fixed-unseen association (solid, uninformative):** corrected class
   composition; high group empty; matching not estimable.
3. **Causal improvement conclusions (absent):** would need matched groups and a
   training intervention; not available.

## Deliverables

`REPORT.md`, `ERRATA.md`, `DECISION.md`, `identity.json`,
`index_invariants.json`, `before_after.csv`, `coverage_fixed.json`,
`matching_fixed.json`, `class_composition_fixed.json`,
`node_key_invariance.json`, `per_molecule_fixed.npz` (git-ignored cache, 4.5 MB,
35 s to regenerate), `part_c_indexfix.py`, `build_before_after.py`,
`manifest.json`, `budget.json`.

**Missing:** none for the main edge view and both fixed keys.  Matched-group
tables are empty by construction (no high group) and are reported as such rather
than forced.