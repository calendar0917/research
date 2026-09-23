# FEC-S0 — path factorization (Stage A / B / C provenance)

Round **FEC-S0** · protocol `fec_s0`.
Pre-registration: [`fec_s0_preregistration.md`](fec_s0_preregistration.md).
Access audit: [`fec_s0_prior_artifact_audit.md`](fec_s0_prior_artifact_audit.md).

Implementation: `tracks/ksvd/experiments/luyin16/fec_s0_factorization.py`.
Every quantity below is rebuilt from **raw molecular primitives** (atom
categories, bond categories and the untyped graph) — never by slicing the
historical tensor.

---

## 1. Local environment — `patch_cont` (146-D)

For root $i$ with radius-2 patch $P_i$ (nodes $V_i$, induced edges $E_i$,
shell $s(v)=\mathrm{dist}(i,v)$, root degree $d_i$):

```
atom_shell[k, a] = ( sum_{v in V_i} 1[s(v)=k] · 1[q_v=a] ) / |V_i|          (3·28 = 84)
bond_shell[p, b] = ( sum_{e in E_i} 1[shellpair(e)=p] · 1[b_e=b] ) / |E_i| (6·4  = 24)
root_atom[a]     = 1[q_i = a]                                              (28)
incident[b]      = ( sum_{e incident to i} 1[b_e=b] ) / d_i                (4)
scalars          = [ log1p|V_i|, log1p|E_i|, boundary_frac, cycle_rank/|V_i|,
                     d_i/4, mean_{v in V_i}(deg v)/4 ]                      (6)
```

`q_v` is the one-hot 28-category ZINC atom primitive, `b_e` the one-hot
4-category bond primitive.  `atom_shell`/`bond_shell` are pure role × primitive
outer-product sums; `root_atom`/`incident` are root-primitive marginals;
`scalars` are pure topology.  The historical 6-class shell-pair schema is kept
verbatim (including the structurally empty `(0,0)` and `(0,2)` classes).

**Result (official-valid, 1 000 molecules, 23 083 patches; train 512): every
raw block is `bit_equal_fraction = 1.0`, `max_abs = 0.0`** (atom_shell,
bond_shell, root_atom, incident_bonds, scalars).

## 2. Standardization (frozen, train-only)

`patch_cont` is standardized by the historical train-fit
`Standardizer(mean, scale)` (`scale = std`, floored at `1e-6`).  The
factorized path uses the **same** standardizer (reproduced with the historical
`zpp.Standardizer.fit(_patch_matrix(train_records))`), never a refit on valid
and never a PEC scaler.

* Reproduction: on full train, `transform(_patch_matrix(train))` vs the cached
  `encoded_train.patch_cont` → `bit_equal_fraction = 1.0`, `max_abs = 0.0`.
* Factorized vs historical standardized `patch_cont` → `max_abs = 0.0`.

## 3. Composition chemistry — `pair_relation` (23-D)

For an unordered centre pair $(i,j)$ with shortest-path summary
$(\text{dist}, \text{count}, \text{bondmean})$ and adjacent bond $b_{ij}$
(defined iff `dist==1`):

```
topology block         = [ onehot(bucket)(5) | log1p(dist)(1) | overlap(5) | boundary(3) ]     (14)
chemistry: path bond   = bondmean(4)      <-- composition-level chemical relation primitive
topology: log count    = log1p(count)(1)
chemistry: adjacent    = onehot(b_ij)(4)  <-- composition-level chemical relation primitive
pair_relation          = concat in historical order                                              (23)
```

The `overlap`/`boundary` coordinates are untyped set operations on the rooted
patches; the two chemistry blocks are explicit relation primitives from raw
bond categories.

**Result (official-valid, 264 776 pairs): every block `bit_equal_fraction =
1.0`, `max_abs = 0.0`, including both chemistry blocks.**

## 4. Global context (62-D) — zeroth-order composition

```
global_context = [ pure-topology structure(30) | atom marginal(28) | bond marginal(4) ]
atom marginal  = sum_v onehot(q_v) / sum_v 1          (normalised)
bond marginal  = sum_e onehot(b_e) / sum_e 1          (normalised)
```

The 30-D structure block is pure topology (node/edge counts, degree moments,
distance moments, eccentricities, clustering/triangles) and is retained and
labelled as such.  The two marginals are explicit zeroth-order composition
statistics derived from raw primitives.

**Result (official-valid): raw and standardized, per block,
`bit_equal_fraction = 1.0`, `max_abs = 0.0`.**

## 5. Pure topology — `topology_features` (25-D) and `pair_bucket`

Both depend only on the untyped graph (exact simple-cycle spectrum + hinge
basis; shortest-path bucket).  Retained verbatim and labelled
`PURE_TOPOLOGY`; the topology standardizer is the historical train-fit one and
reproduces the cache exactly.

## 6. The blocking path — historical alias token (`typed_token`, `parent_token`)

The token is rebuilt from raw primitives by an **explicit binding**: the
rooted typed incidence graph gets node color `(is_root, distance, atom_type)`
and incidence-edge color `(bond_type)`, and the historical token is
`bytes(pynauty.certificate(binding))`.

* Provenance is traceable (the key is a deterministic function of roles and
  primitives).
* **Reconstruction is exact**: on official-valid, the rebuilt certificate bytes
  equal the historical `PatchRecord.typed_certificate` for 1 000/1 000 molecules
  (23 083/23 083 patches), and the vocabulary lookup reproduces the cached
  `typed_token` / `parent_token` ids exactly.
* **But it is not a shared role × primitive factorization.**  The historical
  certificate omits the color labels (the tokenizer module documents this
  explicitly), so it is a **non-injective alias**, and the 6 785-row
  `typed_embedding` owns independent parameters per key (36 420 params).  The
  environment operator is a per-joint-configuration memory, not a shared
  function of local roles and local chemistry.

Aliasing measured on the official-valid audit split:

```
n_patches                         23 083
n_unique_tokens                    2 379
tokens_with_multiple_root_atoms      299  (12.57 % of unique tokens)
occurrences_in_aliased_tokens     15 309  (66.32 % of patch occurrences)
```

So two thirds of the local-token occurrences live in an alias bucket that mixes
different root atom chemistries — a coarsening that no shared role × primitive
factorization would introduce.

## 7. Provenance summary

| path | provenance | factorization verdict |
|---|---|---|
| `patch_cont` | 4× role×primitive sums + root primitives + 6 topology scalars | `FACTORIZED_SHARED` (exact) |
| `pair_relation` | 14+1 topology coords + 8 explicit chemical relation primitives | `FACTORIZED_SHARED` (exact) |
| `global_context` | 30 topology coords + 28/4 primer marginal | `FACTORIZED_SHARED` (exact) |
| `topology_features`, `pair_bucket` | untyped graph only | `PURE_TOPOLOGY` |
| `typed_token`, `parent_token` | explicit binding, **aliased non-injective certificate + per-key table** | `EXPLICIT_BINDING_LOOKUP` (reconstructible, not factorized) |
