# PROTOCOL — `zinc_explicit_bond_object_v1` (ESOC-v1)

**Track:** `structural-sufficiency`
**Round id:** `esoc-zinc-explicit-bond-object-v1`
**Frozen:** before the first formal GPU training second. Numbers marked *measured*
come from the local CPU gate suite (`analysis/cpu_checks.json`) and the GPU
profiling run (`analysis/timing.json`); everything else is a declaration.

---

## 0. The single question of this round

Under an **identical interaction body**, does keeping the *real pair-edge
correspondence* of the key-centred induced local graph — i.e. which two nodes of
`P_b` are joined by which bond type — change performance and generalisation?

Two arms, one difference:

| arm | pair-edge input of the local encoder |
|---|---|
| **REAL** | the real class of each pair `i<j` inside `P_b` |
| **NULL** | the object-level 4-class frequency vector `p_edge`, broadcast to every pair |

Nothing else differs: parameters, initialisation, data order, optimiser, recipe,
readout, relation PE.  NULL is **not** "no structure": it keeps atom types, the
role of every node, the node/pair sets, the local node/edge counts and the
edge-class frequencies.  It removes only the pair ↔ class correspondence.

This round does **not** claim PPGT reproduction, does not claim a new primitive,
and does not claim the model is lossless with respect to the induced graph.

---

## 1. Hard boundaries

1. Official PyG `ZINC(subset=True)`: train 10,000 / valid 1,000, official split,
   raw-`y`, MAE.
2. **Official test is not loaded, not cached, not predicted, not scored.** The
   cache builder refuses `split="test"` (runtime gate
   `provenance.test_split_refused`).  No historical test prediction is read.
3. Single model, ≤ 500,000 inference parameters.
4. No recursive neighbourhood message passing, no MPNN, no hidden-state
   iteration along the local adjacency.  Local encoding is computed once per
   fixed raw graph; tokens are cached as *static indices*, never as learned
   outputs.
5. No external labels, no target-formula components (no SA / logP / cycle
   terms), no distillation, no prediction ensemble / soup, no train-id feature.
6. Reuse is limited to (a) the frozen, label-blind P3 object tensors
   (`atom_feat` for the 21-way atom type, canonical `bond_index`/`bond_type`,
   `global_feat` = the existing legal `global33`), (b) the frozen train-only
   `global33` standardisation from `zinc_arch_discovery_v1/analysis/norm.json`,
   (c) the RRWP construction of `zinc_relation_composition_probe_v1`
   (`rc_data.random_walk_powers` / `bond_relation`), used as an **independent
   reference** for the in-model batched implementation, (d) the frozen
   per-molecule valid table of `zinc_generalization_audit_v1` for the analysis
   groups only.  All four are label-free on the model path.
7. Existing checkpoints, reports, runs and caches of earlier rounds are not
   modified.

---

## 2. Static structural objects (unchanged between arms)

For every undirected physical bond `b = {u,v}` (canonical `u<v`, bond types
`1/2/3` from the raw data):

```
P_b      = {u, v} U N(u) U N(v)                      (measured max |P_b| = 8)
role(i)  = (dist(i,u), dist(i,v)) on the ORIGINAL graph (not the induced one)
edge(i,j)= 0 non-edge / 1,2,3 real bond type          (i<j, i,j in P_b)
```

`h_b = 0.5 * (F(P_b, u→v) + F(P_b, v→u))` with `F` shared.  The reversed
direction swaps the two distance digits of every role; both directions are
actually evaluated and averaged, so the token is **exactly** invariant to
flipping the stored root direction (gate
`invariance.bond_direction_flip_max_delta = 0.0`).

`F` (one shared pass):

```
z_i      = AtomEmb(type_i) + RoleEmb(role_i)                  d_z = 32
node_sum = Σ_i φ_n(z_i)          node_sq = Σ_i φ_n(z_i)^2      d_φ = 64
p_ij     = [z_i+z_j, |z_i-z_j|, z_i*z_j, edge_ij]              (3·32+4 = 100)
pair_sum = Σ_{i<j} φ_p(p_ij)     pair_sq = Σ_{i<j} φ_p(p_ij)^2
h_b      = LN(MLP([node_sum, node_sq, pair_sum, pair_sq,
                   log1p|P_b|, log1p e_b]))                   64 -> 96 -> 64
```

* No normalisation divides by the local object size: the counts are explicit
  inputs, so size information is not removed.
* The pairwise pooling is a deterministic aggregate; **it is not a lossless
  encoding of the induced graph** and no such claim is made.

---

## 3. Global interaction body (identical in both arms)

* Token set: one token per physical bond (≈25 per molecule, max 41).
* Relation: lifted RRWP `R_bc[k] = (M P^k M^T)_bc`, `k = 0..20`, `M` has
  `1/√2` on both endpoints, `P = diag(deg)^{-1} A` on the unweighted raw graph.
  Computed in the model from the raw bond table; parity against
  `rc_data.bond_relation` and against an independent float64 four-endpoint
  enumeration is a gate (max diff 9.3e-8).
* Shared **static** PE stem: `[R, sin πf R, cos πf R for f ∈ {1,2,4}]`
  (84 dims) → 64 → per-head bias (8 heads).  The stem output layer is
  zero-initialised, and the PE is computed once and reused by every layer — it
  never depends on layer state.
* Trunk: 12 pre-norm residual blocks, width 64, 8 heads (head dim 8), FFN
  expansion 2, **AdaRMSNorm** (RMS + DiT-style adaptive scale/shift, no
  BatchNorm; modulation conditioned on a shared rank-4 static graph descriptor,
  projections zero-initialised so the module starts as identity-gain RMSNorm),
  simplified-L2 attention
  `score_ij = (q_i·k_j − 0.5||k_j||²)/√d_head + bias_ij`, attention dropout 0.2,
  graph-level stochastic depth 0.1 (per-sample, inverse-keep scaling).
* Attention is per-molecule only: each molecule is a padded `[E,E]` block, so no
  cross-molecule path exists (gate: perturbing molecule 1 changes the other
  molecules by exactly 0.0).
* URPE is **not** implemented: we could not verify the multiplication position
  of the paper's unitary relative position encoding offline, and an unverified
  "PPGT module" would be worse than an honest omission.  Everything else in the
  body follows the declared recipe.

### Readout (identical in both arms)

```
S1 = Σ_b h_b / 24.9279            S2 = Σ_b h_b² / 24.9279
S3 = Σ_v AtomEmb(type_v) / 23.1664          (shares the local atom table)
counts = [log1p n_atoms, log1p n_bonds, n_bonds/n_atoms, mean |P_b|]
z = [S1, S2, S3, counts, frozen standardised global33]      (197 dims)
y = MLP(197 → 128 → 1)
```

The two reference scales are **fixed train means**; no per-graph normalisation
is applied to any pooled block, so the amplitude difference between molecules is
preserved (gates: `S1` linear / `S2` quadratic under a 3× token amplitude).

**Parameter audit is the only thing that may change depth.**  Measured:
12 blocks = **498,845** ≤ 500,000 → keep 12.  (10 blocks = 429,853; 8 = 360,861.)

---

## 4. Frozen recipe (one trajectory, no sweeps)

| field | value |
|---|---|
| optimiser | AdamW, `betas=(0.9,0.95)`, `eps=1e-8`, weight decay `1e-5` |
| batch | 32 molecules, `drop_last=True` → **312 updates/epoch** |
| LR | peak `2e-3`, 50-epoch linear warm-up (15,600 updates), per-update cosine to `2e-5` |
| horizon | **2,500 epochs = 780,000 updates**, never truncated mid-run |
| loss | raw-`y` L1 |
| clip | global grad norm 5.0 |
| dropout | attention 0.2; graph-level stochastic depth 0.1; nothing else |
| init | from scratch (no SSL encoder, no supervised P3/DCC checkpoint) |
| valid | all 1,000 valid molecules, eval mode, **every epoch** |
| selection | strict `<` on valid MAE; ties keep the earlier epoch |
| checkpoints | `best.pt`, `final.pt`, `last.pt` every 100 epochs, full curve |

Because token set, local encoding, PE and parameter allocation differ, this is
**not** a fair reproduction of the paper's 0.0566 and no such comparison is made.

---

## 5. Pre-training gates (all in `analysis/cpu_checks.json`)

Must pass before the formal run; the overfit gate needs a GPU and is written to
the same file.

| gate | requirement |
|---|---|
| role_vocab | `ROLE_CAP+1 == model role base` |
| invariance | atom renumbering, bond-direction flip, batch vs single, batch reorder |
| local_exactness | 40 molecules × all objects vs an independent networkx enumeration |
| synthetic | empty atom, single bond, tree, triangle, K4, benzene |
| real_null | identical params, **bitwise identical init**, identical optimiser and data order |
| correspondence | class marginals identical; REAL changes under a marginal-preserving pair permutation, NULL changes by exactly 0; REAL/NULL gradients nonzero |
| witness | two synthetic objects, same types/roles/marginals, different pairing: REAL separates, NULL cannot |
| numerics | sL2 and AdaRMSNorm vs independent float64 references; RRWP parity; `R[0]` diagonal 1 / shared 0.5 / disjoint 0 |
| masking | extra padding = 0 change, cross-molecule isolation = 0 change, finite, amplitude preservation |
| budget | ≤ 500k params, optimiser covers every trainable tensor, no BatchNorm |
| provenance | test split refused, static objects provably label-free, caches record no-test flags |
| overfit (GPU) | 48 train molecules: eval-mode train MAE must fall well below the valid level |

---

## 6. Metrics and decision rule

Primary: `best-valid` raw-y MAE per arm/seed.  Also reported: `W20` (minimum of
the 20-epoch rolling **median**), `T100` (median of the last 100 epochs),
`final-valid`, eval-mode `train@best` / `train@final`, and per-molecule paired AE
differences.  The overall valid MAE always uses all 1,000 rows.

Analysis groups are frozen and never enter loss / routing / selection:
radius-2 nested-signature seen (749) / unseen (251) × ordinary (`y > -3`, 935) /
tail (`y <= -3`, 65), `seen+ordinary / unseen+ordinary / seen+tail /
unseen+tail`, ring-rich (`n_rings >= 4`), plus row-172 and the fixed
DCC-L worst-10 sensitivity.

Verdict vocabulary (exactly one is chosen):

* `STRUCTURE_GAIN_AND_COMPETITIVE` — REAL beats NULL across seeds **and** absolute
  performance is at/near DCC-L (valid ≈ 0.076 or better);
* `BACKBONE_GAIN_ONLY` — absolute performance improves but REAL has no stable
  edge over NULL;
* `STRUCTURE_SIGNAL_ONLY` — REAL clearly beats NULL but the absolute level is not
  competitive;
* `UNDERFIT_OR_UNSTABLE` — the body did not train normally; the structure
  question cannot be judged;
* `NO_USEFUL_GAIN` — trained normally, no structural advantage and no meaningful
  absolute gain;
* `COST_INFEASIBLE` — the declared schedule cannot be afforded inside the cap.

DCC-L / DBT numbers are **historical absolute references only**; a cross-body
score difference is not a controlled architectural gain.

### Seed-1 trigger (decided on valid only)

Run the REAL/NULL seed-1 pair **iff** (a) `REAL best-valid ≤ 0.075`, or
(b) `REAL best-valid ≤ 0.079` **and** `REAL − NULL ≥ 0.004` **and** W20 and T100
agree in direction **and** the tail group shows no catastrophic degradation,
**and** the remaining budget covers a full seed-1 pair with margin.
Otherwise stop: no LR / depth / patch-range / readout sweep.

---

## 7. Budget

| item | value |
|---|---|
| GPU cap (this round) | **48 A100 GPU-hours** (both cards summed, incl. profiling, failures, cancels) |
| per-run soft cap | 20 GPU-hours |
| planned | profiling ≤ 0.5 h; seed-0 pair 2 × T; seed-1 pair 2 × T (only if triggered) |
| stop rule | if `2 × projected_full_run > 40 GPU-h` → stop and report `COST_INFEASIBLE` |

A single performance-engineering change is allowed **only** if it does not alter
the mathematics (caching static tensors, avoiding repeated scatter/indexing
backward, better batching), and must be followed by an equivalence check.

### Declared optimisation: static shapes + `torch.compile`

Two changes were made before formal training and are recorded here as
**performance-only**:

1. **Static shapes.** The local-object pair dimension is a fixed slot width
   `EMAX = 41` (the train maximum) instead of the per-batch maximum, and the
   RRWP atom dimension is a fixed `max_atoms = 37` instead of the per-batch
   maximum. Padding rows are masked out of every reduction (verified in gate
   `masking`). This removes recompilation and lets CUDA graphs capture the
   step.
2. **`torch.compile(mode="reduce-overhead")`**, i.e. fused kernels plus CUDA
   graphs. `dynamic=False`, `fullgraph` not forced.

Equivalence evidence (`code/esoc_equiv.py`, `analysis/compile_equivalence.json`,
run on `res` GPU0, torch 2.5.1+cu124):

| check | result |
|---|---|
| initial state bit-identical between the two paths | yes (`state_hash` equal) |
| single-step loss | 1.579736471 vs 1.579736590 → **1.19e-7** |
| train / valid predictions, eval mode | 4.8e-7 / 6.0e-7 max abs |
| 220 gradient tensors (abs diff / gradient scale 0.717) | **1.19e-7** (worst tensor `readout.2.weight`) |
| post-step parameters | 2.16e-8 max abs |
| 10-step drift, loss / params | 1.19e-7 / 2.43e-7 |
| compiled training still stochastic with dropout on | yes (both paths produce spread) |
| fixed bond-slot width vs per-batch width | **exactly 0** difference in RRWP |
| fused `F.rms_norm` vs explicit formula (float64) | **exactly 0** |
| construction seed-determinism | yes |
| REAL/NULL initial state | identical keys, **0.0** difference |

Note on the drift row: an earlier version of this check demanded an exact match
after 30 *stochastic* steps and failed (loss gap up to 0.58). Two causes were
separated: (i) CUDA graphs own the RNG stream, so dropout realisations differ
between the paths — comparing them says nothing about the model, so the
mathematical comparison runs with the two regularisers disabled and the
stochastic path is checked separately; (ii) even with regularisers off, fp32
reduction reassociation accumulates, so the check compares the *function* (one
step, all gradients, post-step parameters) and reports multi-step drift
descriptively. With the regularisers frozen the 10-step drift is 1.2e-7 in loss
and 2.4e-7 in parameters, i.e. the compiled path is numerically the same model.

Measured effect (`analysis/timing.json`): 28.98 s/epoch eager → **6.29 s/epoch**
compiled (312 updates/epoch), i.e. **4.37 GPU-h** for the declared 2500 epochs
instead of 20.1 h. Peak memory 2.18 GB of 40 GB.

---

## 8. Deliverable map

`PROTOCOL.md` (this file) · `code/` · `configs/esoc_v1.json` ·
`analysis/{param_audit,cpu_checks,timing,ledger,main_table,per_molecule_valid,groups,structure_sensitivity}.json|csv` ·
`REPORT.md` · `NEXT_STEP.md`.
