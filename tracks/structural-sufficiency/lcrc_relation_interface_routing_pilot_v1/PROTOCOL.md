# PROTOCOL — lcrc relation–interface routing pilot (v1)

Round id: `lcrc-relation-interface-routing-pilot-v1`. Status: architecture
**feasibility pilot**, every number DEVELOPMENT / DEVELOPMENT_TEST.

## 1. Question

1. If every LCRC S1 occurrence is routed to the topology block that owns its
   bond, and every S2 occurrence is routed either to the single block owning
   both endpoint bonds or to the interface between the two owning blocks, is the
   predictive power of the frozen C1 encoding preserved?
2. Does an interface need context from **both** endpoint blocks (R2), or is its
   own routed cross-boundary evidence enough (R1)?
3. Does the block the readout gates out localise the WC_LOGP oracle mass better
   than a matched random connected subgraph?

## 2. Frozen reading of the LCRC encoding

The pilot *consumes* the frozen encoding; it never re-defines it.

* Atom code `a_v` and bond code `b_e` — the frozen C1 atom/bond encoders
  (`cr_models.C1Trunk` module names and shapes, canonical state, seed 0).
* S1 = the 64-d neighbour code `nbr_code` of `molecular_graph.enumerate_s1`
  (root–neighbour dependence, both directions).
* S2 = the typed path record of `molecular_graph.enumerate_s2`, unchanged:
  unordered joint code `path_cat` (400 codes), directed end code `path_end63`
  (80 codes), directed pair `path_u`/`path_v`, joint projection
  `joint_proj(path_u * path_v)`, closure code `path_closure`, and the gate
  `path_gate`. A two-hop count is **not** an acceptable substitute and is not
  used anywhere.
* The local operator is the frozen `dissect_model.LocalContent(...,"C1",...)`
  (`nbr_emb`, `path_emb`, `end_emb`, `path_u/v`, `joint_proj`, `close_emb`,
  `path_gate`, `LocalMLP`) with **one** shared dropout. No new encoder, no new
  normalisation, no new feature.

The only inputs available to the partition and to the selection score are the
molecule's atom types, bond types and bond graph. No property label, no oracle
contribution, no SMARTS/functional-group list, no atom order, no bond id.

## 3. Frozen topology partition

`rr_partition.partition_molecule`, in this order:

1. bridges of the physical bond graph (iterative Tarjan);
2. **cycle blocks** = connected components of the non-bridge subgraph;
3. **chain blocks** = maximal bridge paths whose interior atoms have bridge
   degree 2 and are not cycle atoms;
4. **isolated-atom blocks** = atoms with degree 0.

Invariants (gates 1–5, whole Train/Valid/Test):

* every bond is owned by exactly one block;
* every block is connected (isolated-atom blocks are single atoms);
* blocks may share atoms, never bonds. The atom weight is
  `w(v,i) = #(bonds of v owned by block i) / deg(v)`, so `sum_i w(v,i) = 1`
  for every atom, and an isolated atom has weight 1 on its own block;
* the block order is intrinsic (kind, size, then a 3-round WL hash of the
  labelled block), so relabelling atoms cannot change the model's output.

**Support gate.** Valid is usable only if ≥ 50 % of its molecules have ≥ 2
blocks **and** ≥ 1 cross-boundary S2 occurrence. Otherwise the round stops with
`UNINFORMATIVE_TOPOLOGY_PARTITION`. Observed: 100 % (see
`analysis/partition_support.json`).

## 4. Frozen routing

* **S1**: an occurrence `(v, e, u)` belongs to `owner(e)`.
* **S2**: let `e1`, `e2` be the two physical bonds of the path.
  `owner(e1) == owner(e2)` → the whole occurrence goes to the *internal* S2
  bucket of that block; otherwise the whole occurrence goes to the interface
  `{owner(e1), owner(e2)}`. Each occurrence has exactly one destination and
  `internal + cross == total` on every split.

## 5. Models

Block-conditioned local input (433-d) for atom `v` in block `i`:

```
[a_v,
 sum_{owner(e)=i} b_e,
 sum_{S1 owner=i} nbr_emb(code),
 sum_{internal S2 of i} (path_emb, end_emb, joint_proj, close_emb),
 sum_{internal S2 of i} a_end * path_gate(code),
 atom_scalars_v]
```

`h_(v,i) = LayerNorm(LocalMLP(x_(v,i)))`, one shared dropout,
`z_i = sum_v w(v,i) h_(v,i)`, `Z_i = [z_i, log1p n_bonds_i, log1p n_atoms_i]` (66-d).

Interface evidence `t_ij` (200-d) is the sum of the cross-boundary S2 records
(`path_emb, end_emb, joint_proj, close_emb, content`), passed through
`iface_mlp: 200→64→64`.

```
R1:  c_ij = f_interface(t_ij)
R2:  c_ij = f_interface(t_ij + MLP_context([Z_i+Z_j, |Z_i-Z_j|, Z_i*Z_j]))
     context: 198→64→64
```

Readout: `y = bias + sum_i c_i + sum_ij c_ij + rho_g(g, log1p n_atoms)` with
`c_i = f_block(Z_i)`. `rho_g` is the only full-graph term (standardised 42-d
global vector + `log1p n_atoms` → 64 → 1). There is no `mean(h)` and no
`mean(h^2)`.

Shared width 64, shared initial weights for every shared tensor (44 tensors
bit-identical between R1 and R2), fixed RNG streams, new modules from seed 0, no
weight taken from any trained model.

**Budget.** C1 = 228 674. R1 = 241 860 (+5.77 %), R2 = 258 756 (+13.15 %),
R2/R1 = 1.0699 ≤ 1.10, so the interface comparison is not declared
`CAPACITY_CONFOUNDED`.

## 6. Training

Four units only: R1/R2 × Lipophilicity (official label) and WC_LOGP (RDKit
Crippen per-molecule value, verified against the committed per-atom
contributions). Official `ogbg-mollipo` split 3360/420/420, separate scalers
(Train only), separate checkpoints.

Recipe **Rf**: seed 0, 360 epochs, batch 128, FP32, AdamW (0.9/0.999, eps 1e-8),
peak LR 1e-3, 5 warmup epochs (linear from 0.1·peak), cosine to 1e-5, wd 1e-5,
dropout 0.10, grad clip 5, Train-only normalisation, Train/Valid/Test evaluated
**every** epoch on the full split, selection = lowest Valid RMSE with ties to the
earlier epoch, Test reported as `DEVELOPMENT_TEST`. No AMP, no EMA/SWA/soup, no
early stopping, no hyper-parameter search, one seed.

## 7. Block gate and selection score

Binary gate `m_i ∈ {0,1}`:

```
y(m) = bias + rho_g + sum_i m_i c_i + sum_(i,j) m_i m_j c_ij
delta_i = y(all on) - y(m_i = 0)
score_i = |delta_i| / sqrt(max(1, n_bonds_i))
```

Selected block = `argmax score_i`, ties to the smallest block index. The score
sees only `c_i`, `c_ij`, the interface endpoints and `n_bonds_i` — no label, no
oracle, no Test information (gate 12).

## 8. WC_LOGP external check

`c_v_oracle` is the Crippen per-atom contribution, with
`sum_v c_v_oracle == WC_LOGP(graph)` verified per molecule.

* **Candidate-set capture (unallocated)**: `capture = sum_{v in atoms(block)} |c_v| / sum_v |c_v|`,
  compared with random connected subgraphs of the same molecule having the same
  number of bonds and the same number of atoms (RNG 20261117, up to 16 distinct
  controls; < 8 ⇒ `matched: false`), enrichment `= capture - mean(control)`.
* **Allocated ledger**: `w(v,i)` splits every atom's `|c_v|` across the blocks
  that contain it, and `sum_i allocated_i == sum_v |c_v|` is asserted on every
  molecule. This ledger is bookkeeping; it is never used for the fair
  comparison above.

Coverage and bootstrap CIs are reported; molecules with insufficient controls
are reported separately and never silently dropped.

## 9. Baselines

Read-only, never retrained: C1 (LCRC) and GINE under Rf from
`gnn_route_clarification_8h_v1/analysis/tables/stage1.csv` for Lipophilicity
(`DIRECTLY_COMPARABLE`), and the LCRC-C1 WC_LOGP run under R3 from
`lcrc_lipo_oracle_task_relevance_public122_s0_v1/runs/wc_logp/train.json`
(`RECIPE_DIFFERS`, flagged per row). No GINE WC_LOGP baseline exists.

## 10. Decision scales (pre-declared, pilot-local)

| question | scale |
| --- | --- |
| prediction retention | Val RMSE ≤ 1.10 × C1 on both units |
| interface increment | paired molecule-level bootstrap of RMSE(R2) − RMSE(R1), 2000 draws, seed 20261117, 95 % CI excluding 0 in favour of R2 and P(R2 better) ≥ 0.95 |
| capacity | R2/R1 > 1.10 ⇒ `CAPACITY_CONFOUNDED` |
| oracle signal | ≥ 60 % of matched molecules with positive enrichment and positive mean enrichment, on Valid or Test |
| readout shortcut | mean |rho_g| / mean |sum c_i + sum c_ij| > 4 ⇒ `RHO_DOMINATES_STRUCTURE` |
| ledger | reconstructed within `1e-5 + 1e-6·max(1,|x|)` |

## 11. Gates

Twelve gates run before and after training: bond ownership, block connectivity,
S1 conservation, S2 conservation/single destination, weight sums, relabelling
invariance, ledger closure, all-ones gate equals the forward pass, block-gate
semantics, parameter/config identity, data/scaler/oracle identity, and
selection label/oracle blindness. Output: `analysis/routing_invariants.json`.

## 12. What this round does **not** do

No learnable partition, no new seed, no hyper-parameter search, no extra
dataset, no SMARTS or functional-group prior, no property label or oracle mass in
the definition of structure, no Test-based selection, no retraining of C1, GINE,
the oracle baselines or any historical model, and no claim beyond feasibility.
