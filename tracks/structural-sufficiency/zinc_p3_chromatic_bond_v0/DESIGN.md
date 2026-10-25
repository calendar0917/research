# DESIGN — P3-Chromatic Bond Transformer v0

**Round:** `zinc-p3-chromatic-bond-v0`
**Track:** `structural-sufficiency`
**Status:** CPU-gated (34/34 assertions), frozen before GPU

## 0. One paragraph

This round tests one mechanism on the frozen P3/DCC substrate: a bond token's
attention distribution over all physical bonds is normalised **independently
for each of the 48 output feature channels**, and the fixed structural relation
of a bond pair enters **that channel's** logits. The scalar control uses the
same parameters, the same initial state and the same relation cache but
replaces each head's 12 channel-specific relation logits by their within-head
mean before the key-axis softmax. Nothing else differs. The only performance
question is whether the per-channel normalisation buys anything over the
scalar control, and whether either arm reaches the historical DCC-L absolute
level.

## 1. Prior-work and pre-check (read before implementation)

### 1.1 Public CSA

The mechanism is **not** claimed as this repository's invention. The public
reference is:

- Menegaux et al., *Self-Attention in Colors*, TMLR 2023,
  arXiv:2304.10933.
- `inria-thoth/csa`, `graphgps/layer/csa_layer.py` (commit inspected:
  `main`, file blob `7582c22c…`), and
  `configs/CSA/zinc-CSA+Rings.yaml`.

In the official layer, `E_att` has shape `[..., num_heads, out_dim]`; it is
added to the per-head dot-product scores, the result is reshaped to
`[..., num_heads * out_dim]`, and `softmax(dim=key)` is taken over the full
`num_heads * out_dim` feature axis. The official ZINC config uses 10 layers,
2000 epochs, batch 32, `TypeDictEdge+RWSE` edge features and ring-coalesced
edge types. **This round is not a CSA reproduction**: it uses 5 layers,
600 epochs, batch 128, P3 bond objects, a 21-step lifted atom random-walk
bond-pair relation and a chordless-cycle indicator. The paper's 0.056 is not
an expected score for this model.

### 1.2 Is the same computation already in this repository?

Read: `dcc_models.py`, `dbt_models.py`, `gos_models.py`,
`rc_models.py`, `sta_models.py`, `bcca_models.py`, `nr_models.py`,
`esoc_models.py`, `zps_grit_dense.py`.

| implementation | attention normalisation | structure enters logits? | per-output-channel softmax? |
|---|---|---|---|
| DCC-L (`dcc_models.py`) | no bond attention | — | no |
| DBT (`dbt_models.py`) | `softmax_f` per head | `rel_b -> per head`; `rel_g`/`rel_u` modulate **values** per channel | **no** |
| GOS A (`gos_models.py`) | `softmax_f` per head | `beta(rel) -> per head`; `gamma(rel)` gates values | **no** |
| GOS K-softmax control | intended per-channel weights | `kernel(rel) -> per channel` | **no valid implementation**: the branch was never run (`Ksm` explicitly not run), it masks with a 5-D mask against a 4-D tensor, and it calls `softmax(raw, dim=1)` on the **query** axis, not the key axis. Even if it ran, it is a K-family control, not a trained arm. |
| GRIT dense port | `softmax` over keys with singleton channel axis | `E` enters per-head score; `Aw` reduces to one logit per head | **no** |
| STA / NR / BCCA / RC / ESOC | per-head or single-head | per-head bias | **no** |

Therefore the repository does **not** already contain a valid, trained
equivalent of "per-output-feature-channel key softmax with the structural
relation in that channel's logits". The nearest object is DBT's per-channel
**value** modulation, which is a different computation. The round proceeds.

## 2. Frozen local carrier (identical to DCC-L B-R0)

The P3 encoders (fine-tuned, initialised from the seed-0 train-only masked-SSL
checkpoint `006fabc1…`) produce `a` (96), `b` (96), `w` (96). The DCC-L
`group_composer` and `local_composer` produce

```
c_glob = group_composer([b, Wg, Vg, log1p(n_b), bond_feat])      # 96
c_loc  = local_composer([b, Wl, Vl, log1p(n_b), bond_feat])     # 96
```

where `Wg,Vg` are the P3 global grouping and `Wl,Vl` are the **true junction
ownership** sums (`Wl_b = sum_{j touches b} w_j`, `Vl_b = sum_{j touches b}
w_j^2`), not graph means.

Each physical undirected bond `e` becomes one token:

```
h0_e = LayerNorm(Linear(290,48)([b_e, c_glob_e, c_loc_e,
                                  log1p(line_degree_e),
                                  log1p(n_bonds)]))
```

`line_degree_e` is the number of real junctions incident to `e` (equal to the
line-graph degree for the raw bond graph; checked). `n_bonds` is the molecule's
physical bond count. The three 96-D blocks and the two static scalars are
concatenated in that order; no role is summed away.

This is a one-shot local encoding. There is **no recursive local hidden-state
message passing** on the original graph or on the line graph. The P3 encoders
and both composers are fine-tuned with the supervised loss, as in DCC-L.

## 3. Fixed structural relation `r_ef ∈ R^22`

For every ordered pair of physical bonds `(e,f)` in the same molecule:

1. **Lifted atom random walk (21 dims).** With the raw unweighted adjacency
   `A`, row-stochastic `P = diag(deg)^-1 A`, and the bond-to-atom incidence
   matrix `M` (`M_{b,i}=1/sqrt(2)` if `i` is an endpoint of `b`),

   ```
   R_ef[k] = (M P^k M^T)_ef,   k = 0..20.
   ```

   This is exactly the `zinc_relation_composition_probe_v1` relation
   (`rc_data.bond_relation`), re-implemented here and cross-checked against the
   old implementation. `R[0]` is 1 on the diagonal, 0.5 for two bonds sharing
   one endpoint, 0 otherwise. It is not a line-graph walk.

2. **Chordless-cycle co-membership (1 dim).** `ring_ef = 1` iff there exists a
   chordless cycle of length 3..18 in the raw undirected bond graph that
   contains both physical bonds. For `e=f`, this means "the bond lies in at
   least one chordless cycle". Chordless cycles are enumerated on the raw
   graph with a graph algorithm; no RDKit descriptor and no target-derived
   quantity is used. The original bond graph is never modified.

The relation is computed once, label-free, and stored in a shared cache. Both
arms read the same `[E,E,22]` tensor for each molecule; the batch collation is
byte-identical for C and S.

**Equivariance.** Atom relabelling permutes the bond axes and leaves the values
unchanged; flipping a stored bond endpoint leaves `R_ef` unchanged; permuting
physical bonds permutes both axes. The ring indicator has the same property by
construction. These are CPU gates, not claims about trained behaviour.

**Ring definition boundary.** The public CSA ring feature is defined on atom
pairs and uses ring-membership edge features. This round generalises the idea
to physical bond pairs with chordless cycles. It is not a per-operator
reproduction of CSA's ring encoding.

## 4. Chromatic block (depth 5, width 48, 4 heads, head_dim 12)

Pre-LN, no BatchNorm, attention dropout 0, residual scale `1/sqrt(5)`.

```
x       = LayerNorm1(h) * mask
q,k,v   = Wq x, Wk x, Wv x                    # each 48->48, no bias
QK_efh  = <q_eh, k_fh> / sqrt(12)             # shared by the head's 12 channels

rh_efh  = rel_head(r_ef)                      # 22->4, no bias
rc_efhc = rel_channel(r_ef)                   # 22->48, no bias, zero-initialised
rv_efhc = rel_value(r_ef)                     # 22->48, no bias

# C (Chromatic):
logit_efhc = QK_efh + rh_efh + rc_efhc
# S (Scalar control):
logit_efhc = QK_efh + rh_efh + mean_{c in head h}(rc_efhc)

alpha_efhc = softmax_f(logit_efhc)            # key axis f, independently per channel
message_ehc = sum_f alpha_efhc * (v_fhc + rv_efhc)

h <- h + (1/sqrt(5)) * W_o([message, scale_e * message])   # 96->48
h <- h + (1/sqrt(5)) * FFN(LayerNorm2(h))                  # 48->96->48, SiLU
```

`scale_e = log1p(line_degree_e) / train_mean_log1p_line_degree`
(`train_mean = 1.308120486872196`, computed on the 10k train structures only).

Key padding is the **only** mask. Every physical bond pair in the same molecule
is a key, including self; there is no adjacency/line-graph mask. Non-adjacent
bond pairs have non-zero attention (checked numerically).

`rel_channel` is initialised to all zeros. `rel_head` and `rel_value` use the
frozen seed's default PyTorch initialisation. Therefore C and S have identical
initial predictions. After a non-zero `rel_channel`, C and S differ; a
within-head zero-mean perturbation changes C but not S (CPU gate).

**Parameter sharing.** The three relation projections are per layer and shared
across the four heads (and, for `rel_channel`/`rel_value`, across the 12
channels of a head through the linear map). This is the parameterisation that
makes the ledger exactly 477,017. It is not a claim that C and S have equal
effective capacity; S restricts the same 48 parameters through a within-head
mean and is a strictly smaller function class.

## 5. Reader

Only the new model's own readout is used; the old DCC-L 513→256→256→1 reader
is not retained and no old prediction path is a bypass.

```
z = [ sum_e h_e,            # 48
      sum_e h_e^2,          # 48
      sum_v a_v,            # 96
      sum_v a_v^2,          # 96
      standardized_global33,# 33
      log1p(n_bonds),       # 1
      log1p(n_atoms) ]      # 1
y = MLP(323 -> 128 -> 64 -> 1, SiLU)
```

`standardized_global33` is exactly `(global - train_mean)/train_scale` from the
frozen P3 normalisation. No per-molecule LayerNorm is applied to any pooled
block. No `c_glob`/`c_loc` block enters the reader directly.

## 6. Parameter ledger (analytic; formal audit on `named_parameters()`)

| module | params |
|---|---:|
| P3 encoders (fine-tuned, SSL init) | 202,048 |
| `group_composer` (DCC-L) | 62,976 |
| `local_composer` (DCC-L) | 31,536 |
| token fusion `Linear(290,48)` + `LayerNorm(48)` | 14,064 |
| 5 interaction layers × 23,320 | 116,600 |
| reader `323→128→64→1` | 49,793 |
| **total** | **477,017** |

Per interaction layer: `Q/K/V` 3×2,304 = 6,912; `rel_head` 88;
`rel_channel` 1,056; `rel_value` 1,056; `W_o` 4,656; two LayerNorms 192;
FFN 9,360. Sum 23,320.

The budget is 500,000. The actual count is asserted from the real module, not
from this table.

## 7. What is materially different from the historical models

| model | object | interaction | relation in logits | per-channel softmax | reader |
|---|---|---|---|---|---|
| DCC-L | physical bond | none | none | no | old 513 reader |
| DBT | directed bond | 4 global layers, per head | per head | no | pooled directed tokens |
| GOS A | physical bond | 3 global layers, per head | per head | no | old 513 + token moments |
| RCC | physical bond | fixed quadratic kernel, no pair enumeration | kernel, not softmax | no | 513 + quadratic |
| **this C** | physical bond | 5 global layers, full key set | **per channel** | **yes** | new 323 reader |
| **this S** | physical bond | same | per-head mean only | no (head softmax) | same 323 reader |

The C/S contrast isolates the channel-independent normalisation. The C vs
DCC-L contrast is a whole-model comparison (tokenisation, depth, reader and
loss path all differ), so a win or loss is not attributable to one equation
alone; only Q1 is a controlled mechanism contrast.

## 8. Explicit non-claims

- Not a CSA reproduction; not a new invention of chromatic attention.
- No official test read, no test-driven selection.
- Valid is an exposed development split; no blind confirmation.
- A single seed is a pilot, not a family-level causal claim.
- The scalar control has the same parameter count but not the same effective
  capacity; the report will not claim an "effective-capacity-matched" control.
- Ring support and relation-difference statistics are descriptive; no
  post-hoc predictability R² gate is used to stop the round.
