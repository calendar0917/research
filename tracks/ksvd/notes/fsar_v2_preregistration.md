# FSAR-v2 pre-registration — exact attribute marginals + explicit structural basis

Branch `exp/fsar-v2-rich-marginal-explicit-structure-zinc`, created from the
FSAR-v1 verdict HEAD `af8d5c2`.  Official ZINC test is never loaded
(`official_test_loaded = false` in every JSON).

This note freezes the architecture, the mode grid, the parameters, the gates and
the execution schedule **before** any formal FSAR-v2 run is launched.

## Questions (exported from FSAR-v1)

FSAR-v1 established: trainable bypass-free factorized backbone; functional
relational core / pooling; `B` genuinely used; `SAB - SAM = 0.00327` aligned
increment beyond a matched marginal capacity control; but `A` (mean/std of
learned node embeddings) is severely information-starved (`A = 0.157917` vs the
`0.135` guard), and `S` is a *pure-topology but implicit* topology-GNN state.

FSAR-v2 answers exactly two questions:

* **Q1** Does the attribute-marginal information starvation disappear once the
  complete categorical attribute marginal is preserved exactly?
* **Q2** Can the implicit topology-GNN `S` be replaced by a fixed, readable
  **explicit structural basis** while retaining performance and the aligned
  binding gain?

## Exact ZINC schema (audited, not assumed)

`schema_audit` (train + val only):

| split | x field | atom categories observed | edge_attr field | bond categories observed |
|---|---|---|---|---|
| train | single categorical (`x.shape = [n,1]`) | 0..20 | single categorical | 1..3 |
| valid | single categorical (`x.shape = [n,1]`) | 0..15 | single categorical | 1..3 |

Atom / bond attributes are single discrete categorical fields (no multi-field
tuples), so the "complete attribute tuple multiset" reduces to the category
count vector.  The inherited encoder schema is padded to **28 atom / 4 bond**
categories (`zpp.ATOM_CATEGORIES`, `zpp.BOND_CATEGORIES`), which covers all
observed values, so a 28 / 4 count vector is exact.  The runner refuses to run
if the processed split files are missing (would trigger PyG `process()`, which
reads the test split).

## A_exact — lossless categorical marginal

For each rooted radius-2 reference frame `W_v`:

```
raw_v = [ root_one_hot(x_v) (28) , context_atom_counts (28) , bond_counts (4) ]  -> 60-D
A_v   = MLP_A(raw_v):  Linear(60,64) -> SiLU -> Linear(64,64)                    -> 64-D
```

* `root_one_hot` is the complete categorical identity of the centre atom (the
  first `Linear` realizes `E_root-atom`).
* `context_atom_counts[k] = #{u in W_v \ {v} : x_u = k}` — exact integer counts.
* `bond_counts[k]` — exact integer counts over the patch's directed bond entries
  (`struct_bond` grouped by `struct_edge_patch`); an undirected bond contributes
  2, a fixed invertible scaling of the undirected bond multiset.

`A_exact` reads **only** `struct_atom`, `struct_root`, `struct_patch`,
`struct_bond`, `struct_edge_patch`.  It never reads `struct_dist`, `struct_src`,
`struct_dst`, any degree/shell/role, or any adjacency propagation (test
`a_exact_is_blind_to_topology_fields`).

Losslessness (test `exact_marginal_assignment_invariance_and_collision_freedom`,
runner `collision_audit`): two frames collide in `raw_v` iff their root category
and context atom / bond category multisets are identical; distinct marginals
cannot collide.  The audit samples 200 train+valid molecules (4,645 frames):
1,148 unique marginal keys, 1,148 unique raw vectors, 0 collisions, 0
same-key/different-raw mismatches.

## Information-resolution confound removal

`B_implicit` reads the shared categorical semantics
`E_atom = atom_mlp(atom_embedding(struct_atom))` and
`E_bond = bond_mlp(bond_embedding(struct_bond))`.  `A_exact`'s raw counts are
over exactly the categories indexing those embeddings, so the multiset of
`E_atom` / `E_bond` values is fully recoverable from `A_exact`.  Therefore the
only principled information `B` has that `A_exact` cannot recover is **which
attribute belongs to which structural role**.  This is asserted by
`a_exact_reconstructs_full_attribute_multiset` and re-checked in the report.  If
`B` were found to read any raw chemical field outside `A_exact`, the run would
FAIL (`information-resolution confound remains`).

## S_explicit — fixed rooted structural operator basis

No adjacency message passing anywhere on the explicit path (AST test
`explicit_path_has_no_message_passing` + `message is None`/`update is None`).

Node basis (11-D, chemistry-free, `log1p` on integer counts):

```
1[u=v] ; shell one-hot for d(v,u)=0,1,2 ; log1p(indeg_W(u)) ;
log1p(|N(u) ∩ shell_j(v)|), j=0,1,2 ; log1p((A)_{vu}), log1p((A^2)_{vu}), log1p((A^3)_{vu})
```

Edge basis (15-D, undirected, chemistry-free):

```
one-hot unordered shell pair (d(v,u),d(v,w)) over 6 pairs ;
log1p(deg(u)+deg(w)) ; log1p|deg(u)-deg(w)| ; log1p(common neighbours) ;
log1p(n_j(u)+n_j(w)), j=0,1,2 ; log1p|n_j(u)-n_j(w)|, j=0,1,2
```

S_explicit readout (fixed, output 32-D, no sweep, no adjacency message passing):

```
S_v = F_S( φ_vv , mean_u φ_vu , std_u φ_vu , mean_e φ^E_ve , std_e φ^E_ve )
      : Linear(63,32) -> SiLU -> Linear(32,32)
```

Walk order is fixed at `k = 3`; no basis dimension / motif / cycle-vocabulary
sweep.

### B_explicit (centred aligned interaction, same capacity class as B_implicit)

```
c_vu   = U_φ(φ_vu - mean_u φ_vu) ⊙ U_a(a_u - mean_u a_u)
c^E_ve = U_{φE}(φ^E_ve - mean_e φ^E_ve) ⊙ U_e(a_e - mean_e a_e)
B_v    = F_B(mean_u c_vu, std_u c_vu, mean_e c^E_ve, std_e c^E_ve)
```

with `a = E_atom(a)`, `a_e = E_bond(e)`, projections / MLPs of the same widths
as the implicit B.

## Frozen mode grid and parameters

Phase 1 (implicit S): `A`, `SA`, `SAB`, `SAM`.
Phase 2 (explicit S): `SAE`, `SABE`, `SAME`.

`SAM` / `SAME` are marginal-only capacity controls `M_v = MLP_M([A_v, S_v])`;
`M` reads no aligned per-node / per-edge pair.

| mode | total params | S encoder | B encoder | M |
|---|---:|---:|---:|---:|
| A | 68,281 | 0 | 0 | 0 |
| SA | 89,273 | 16,896 | 0 | 0 |
| SAB | 106,873 | 16,896 | 13,504 | 0 |
| SAM | 106,867 | 16,896 | 0 | 9,402 |
| SAE | 75,481 | 3,104 | 0 | 0 |
| SABE | 90,841 | 3,104 | 11,264 | 0 |
| SAME | 90,821 | 3,104 | 0 | 7,148 |

`SAM - SAB = -6` and `SAME - SABE = -20` (< 0.1 %).  The marginal-only `M` is
intentionally smaller than `B` because the widened `[A,S,M]` node-init absorbs
the remainder of the released budget; the *total* is matched, which is the
correct capacity control.  Zero dataset-dependent vocabulary parameters.

## Pre-registered gates (unchanged from the brief)

* `A_exact` classification (seed0 soup):
  strong success `<= 0.135`; partial success `0.135 < A <= 0.145` **and**
  improvement vs FSAR-v1 `A` `>= 0.010`; otherwise failure.
* aligned increment: `Δ_aligned = MAE(SAM) - MAE(SAB) >= 0.001` with functional
  no-B / shuffle-B interventions.
* explicit-stage gate (Q2): `A_exact <= 0.145` **and**
  (`SAB_exact <= 0.132` **or** `MAE(SAM) - MAE(SAB) >= 0.001`).
* explicit classification: parity `SABE <= SAB + 0.002`; moderate cost
  `<= SAB + 0.005`; failure otherwise.
* seed1 confirmation is bought only under the §27 conditions.

Training protocol inherited verbatim from the frozen optimized compact-v4 cell:
Adam, lr 1e-3, weight decay 1e-5, batch 128, grad clip 5.0, max 240 epochs,
patience 40, no scheduler, best-valid checkpoint, fixed equal-weight Top-5 soup,
deterministic algorithms, single ZINC L1/MAE task loss.  No distillation, no
teacher, no auxiliary / reconstruction / binding / shuffle loss.

## Execution schedule (two-GPU rule)

The brief authorises two independent A100 jobs in parallel.  GPU availability is
checked immediately before each wave:

* if both GPUs are free: Wave 1 = `A seed0` on GPU0 and `SA seed0` on GPU1;
  Wave 2 = `SAB seed0` on GPU0 and `SAM seed0` on GPU1; conditional Wave 3 =
  `SAE seed0` / `SABE seed0`; conditional Wave 4 = `SAME seed0`.
* if only one GPU is free: run the wave's jobs **sequentially** on the free GPU,
  never co-tenanting and never touching another user's process.

Each job is a separate process (`launch_remote.sh`) with its own log / pid /
metadata / result directory, same committed revision, same environment, same
data cache/schema, same batch/optimizer/schedule/deterministic setting.

## Files

* code: `experiments/luyin16/fsar_v2.py`, `experiments/luyin16/zinc_fsar_v2.py`
* tests: `tests/test_fsar_v2.py` (12 gates)
* results: `results/fsar_v2/` (gitignored)
