# DESIGN — zinc-directed-bond-transformer-v1 (DBT)

**Round:** `zinc-directed-bond-transformer-v1`, track `structural-sufficiency`.
**Code:** `tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/`.
**Status:** declared before the first formal GPU training second. Numbers marked
*measured* below come from the local CPU checks and the remote timing run; the
parameter table is the realised instantiation, not an estimate.
**Data:** official PyG ZINC `subset=True`, train 10,000 / valid 1,000 / test
1,000, raw-`y` MAE. Valid drives every development decision; test is read once,
after the single best-valid checkpoint is frozen (already authorised, and the
official test split is already spent in this repository).

---

## 0. One proposition

The working hypothesis is **not proven**: the earlier static models may have
been limited because they did not combine *concrete structural objects together
with their connectivity roles* sample-efficiently. This round tests exactly one
model for that hypothesis:

> Keep both end-point roles of every real bond, and drive multi-layer
> all-graph interaction with explicit endpoint-bound structural relations
> between the directed bond objects. Does a single model generalise better?

It is not an ensemble, not a router, not a concatenation of two full models,
not distillation, and there is no teacher. No claim is made that "endpoint role
loss is the proven bottleneck", and no novelty / WL / score claim is made in
advance.

## 1. What is new relative to earlier implementations in this line

Nothing below is claimed as a new primitive; the table only states what this
computation does that the earlier trunks in this track did not do.

| earlier work | its cross-object computation | what DBT does differently |
|---|---|---|
| P3 / P3B (Null composer + reader) | per-bond `c_b = MLP([b, n_b·mean w, n_b·mean w², log n_b])`, then sum | no directed object exists; the other end of the bond is never a separate role, and there is no interaction between bond objects |
| L / C (dual-channel composer) | parallel local composer over real junctions, then a P3 bypass | one-shot, no multi-layer interaction, junction content summed per bond |
| A (pair composition) | every **atom pair** as an object, one association contraction over a third atom `k`, then per-pair MLP | DBT's objects are the **directed physical bonds**, and atom-pair path counts are only the *relation* between two bond objects, never an object of their own; the interaction is iterated 4× and is not a single contraction |
| B (spectral invariants) | whole-graph matrix-function traces | DBT is structure-resolved, not a global average |
| STA (structural token attention) | additive type/distance/overlap bias on atom-level attention | DBT changes the *token set* (directed bonds with endpoint roles), feeds the relation into the **value** as a per-channel multiplicative + additive term, and uses a 4-block endpoint-bound path-count relation |
| ZSRT / GRIT-style reference | RRWP / degree / node-pair update | DBT's relation is the exact simple-path count of the four endpoint blocks `(u,p)(u,q)(v,p)(v,q)` of two **directed bond** objects; the two objects are not atoms, and no anchor/positional node update is used |
| J (rooted induced paths ≤5) | all rooted paths pooled per root | DBT's paths are **relations between objects**; they are never summarised into a per-root or per-graph bag of path tokens before the interaction |

The only external methodological borrowings are (i) simple-path counts as an
endpoint relation (SPSE-style object, here a **bounded exact** count with
`L = 12`, not SPSE's approximate `L = 20`), and (ii) full-graph
relation-conditioned attention as a workable implementation/training pattern
(GRIT-style mechanism). Neither literature score nor expressivity claim is
inherited.

## 2. Static objects and the shared local encoding

Every undirected physical bond `{u,v}` produces two directed objects
`e = (u→v)` and `reverse(e) = (v→u)`. They are two structural views inside
**one** model with shared weights, not two models. The physical bond table is
canonical (`u < v`), so object `2k` is `(u_k→v_k)` and `2k+1` is `(v_k→u_k)`;
the arithmetic order of indices is used only for indexing, never as a feature.

The shared local encoders are the P3 `atom_encoder` / `bond_encoder` /
`junction_encoder` (202,048 parameters), initialised from the **seed-0,
train-only masked-SSL** checkpoint
`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`
(sha256 `006fabc1…`; pretraining protocol `cscl-bjoc-full10k-pretraining-transfer-v1`,
15 % atom / 15 % bond masking, 7,560 updates; it is **not** a supervised P3
checkpoint). All encoders are fine-tuned during supervised training. No unused
P3 composer/reader is instantiated.

Static directed-bond token (`h_e^0`, width 64):

```
h_(u→v)^0 = LN( InitMLP([ a_u , a_v , b_uv , J_src , J_dst ,
                           log1p(n_src) , log1p(n_dst) ]) )
```

* `a_u`, `a_v` – source and target atom states, concatenated **by role**
  (never symmetrised), so `reverse(e)` naturally swaps the two halves.
* `b_uv` – the physical bond state.
* `J_src = (1/n_src) Σ_{k ∈ N(u)\{v}} w(k,u,v)` – mean of the **real** P3
  junction states centred on the source `u` that pair the target `v` with each
  other neighbour `k` (P3 `(i,j,k)` notation, `j` is the centre). `J_dst` is the
  same at the target: `(1/n_dst) Σ_{k ∈ N(v)\{u}} w(u,v,k)`.
  Junction membership comes from the original graph's P3 incidence table, not
  from node numbering or cache ordering.
* `n_src = deg(u)-1`, `n_dst = deg(v)-1` – real junction counts, kept
  explicitly (`log1p`) so "empty set" is distinguishable from "mean zero".

The five role blocks are projected by five role-specific `96→32` linear maps
before the shared MLP (`162→96→64`). This is the declared, parameter-efficient
form of the InitMLP: the role-ordered concatenation is exact and no information
is symmetrised; only the first layer is block-diagonal instead of dense
(21,856 + 5×3,104 = 37,376 params instead of ~52.6k for a dense `482→96` first
layer). Measured total is 452,523, so the dense variant would still fit; the
block-diagonal variant is kept for the stronger role prior.

This is a **one-shot static composition**: no updated `h` is ever re-injected
along the local adjacency to build the next neighbourhood token.

## 3. Endpoint-bound relations (fixed, label-free)

From the raw graph only, per molecule:

```
S_l(i,j) = number of SIMPLE paths (no repeated node) from i to j with exactly l edges
dist(i,j) = shortest-path distance: 0..12 exact, 13 = connected but > 12, 14 = unreachable
```

`l = 1 .. L_MAX = 12`, diagonal `S_l(i,i) = 0` for `l ≥ 1`; identity is carried
by the endpoint-equality flags instead. The count is a **bounded exact**
enumeration (every simple path up to 12 edges is visited; nothing is sampled,
nothing is truncated per graph). Observed maximum count over train+valid+test
is 12, stored as `uint8` (the builder refuses to write a split that exceeds
255). Cost measured: 8.9 s for the whole 10k train split, 0.9 s for valid, on
one CPU core — so no approximation and no change of `L` was needed.

For directed objects `e=(u→v)`, `f=(p→q)` the fixed raw relation contains, for
each of the four endpoint blocks `(u,p)`, `(u,q)`, `(v,p)`, `(v,q)`:

* the 12 exact path counts (`log1p`, shared `12→8` projection; block position
  keeps the role — no mean/min/max/sort over blocks anywhere),
* the shortest-path distance (shared `15×6` embedding; long-but-connected and
  unreachable are separate buckets),

plus four endpoint-equality flags (one per block) and five deterministic
orientation flags (same physical bond, reverse object, self, same source, same
target). Raw relation width: `4×(8+6) + 4 + 5 = 65`.

The cache holds only the atom-pair tables `S_l` and `dist`; the directed-pair
relation is gathered inside the batch, so no `E²`-sized learned tensor is ever
stored per molecule. Cache version `dbt-path-v1-L12`, file names carry the
version, writes are atomic.

No new feature depends on `y`, on a computed-label SA/logP/cycle component, on
an old oracle target or on any external label. The existing legal `global33`
block is kept and standardised with the frozen train-only statistics.

## 4. Multi-layer full-graph relation interaction

4 layers, width 64, 4 heads (head dim 16), FFN multiplier 2, pre-LN, residual,
dropout 0.05 for attention and FFN (the local encoders keep their SSL default
dropout 0). Shared relation pre-encoder: `RelationMLP(65→32→24)`; `B`/`G`/`U`
are per layer. Per head and per directed-object pair:

```
α_ef      = softmax_f( q_e·k_f/√d_head + B_head(r_ef) )
value_ef  = V_head(h_f) · (1 + tanh(G_head(r_ef))) + U_head(r_ef)      # per channel
context_e = Σ_f α_ef · value_ef
h_{e}^{l+1} = h_e^l + Dropout(O(context_e));  h ← h + Dropout(FFN(LN(h)))
```

The relation therefore enters **both** the logits and the read content, and the
per-channel `G` gate is multiplicative. Attention covers *all* directed objects
of the same molecule; only padding and other molecules are masked. There is no
endpoint-sharing restriction, no line-graph recursion, and no hidden-state
message passing along the local adjacency. Padding rows are re-zeroed after
every layer, so they cannot contribute to any pooled block.

Source degree, target degree, molecule size and bond count are injected into
every layer through one shared `4→64` projection (320 parameters) — declared
and counted, not a novelty claim.

## 5. The readout (one reader, amplitude preserved)

Over the final directed-object states `H = {h_e}`:

```
S1 = 0.5 · Σ_e h_e            # each physical bond counted exactly once
S2 = 0.5 · Σ_e h_e²
z  = [ S1/n_dir_ref , S2/n_dir_ref , Σ_v a_v/n_atom_ref , Σ_v a_v²/n_atom_ref ,
       standardised global33 , log1p(n_atoms, n_bonds, n_dir, n_junc) ]
y  = Reader(357 → 128 → 128 → 1)
```

`n_dir_ref = 49.8558` and `n_atom_ref = 23.1664` are fixed **train** means
(`analysis/graph_scales.json`); the same constants are used for every split.
No per-molecule LayerNorm or standardisation is applied to any pooled block, so
the difference in amplitude between molecules is preserved. LayerNorm inside
the trunk is unrelated to graph-level normalisation. Isolated atoms survive
through `Σ a_v` / `Σ a_v²`; a molecule without bonds gives finite zero bond
moments (checked).

## 6. Parameters (measured, single instantiation)

| module | parameters |
|---|---:|
| P3 atom/bond/junction encoders (SSL-initialised, fine-tuned) | 202,048 |
| directed-token init: 5 role projections + MLP(162→96→64) + LN | 37,504 |
| per-layer trunk ×4 (QKV, out, FFN, 2 LN, B/G/U) | 147,088 |
| shared relation pre-encoder (path 12→8, dist 15×6, MLP 65→32→24) | 3,098 |
| shared static-count injection 4→64 | 320 |
| reader 357→128→128→1 | 62,465 |
| **total inference parameters** | **452,523** |

Cap 500,000; headroom 47,477. Every parameter requires grad and is inside the
optimizer (checked). Nothing is frozen, hidden in the optimizer outside the
model, or cached as a weight.

## 7. Declared recipe (one trajectory, seed 0)

| field | value |
|---|---|
| seed | 0 |
| data | official PyG ZINC `subset=True`, train 10,000 / valid 1,000 |
| loss | raw-`y` L1 |
| batch | 64, last partial batch kept → **157 updates/epoch** (10,000 = 156×64 + 16) |
| optimizer | AdamW, peak LR 3e-4, weight decay 1e-4 |
| schedule | linear warm-up 1,000 updates, then per-update cosine to 3e-6 |
| horizon | **1,500 epochs = 235,500 updates, fixed before launch, never reset** |
| dropout | 0.05 global attention/FFN; 0 in the SSL encoders |
| grad clip | 5.0 |
| valid | every epoch, all 1,000 valid molecules, eval mode |
| selection | strict `<` on valid MAE; ties keep the earlier epoch |
| checkpointing | `best.pt` on improvement; `last.pt` (optimizer + scheduler step + RNG + epoch) every 25 epochs; resume continues the same cosine |

This is a reasonable starting point, not a validated optimum. No LR / WD /
dropout / depth / width / path-length grid is run, and no seed 1 is added.

## 8. Pre-training verification (see `analysis/cpu_checks.json`)

1. path counts match `networkx.all_simple_paths` on 4 graphs, hand-computed
   `K4`/5-path values, are symmetric, have zero diagonal, and are relabelling
   equivariant;
2. atom relabelling, reversed edge-input direction, row order and batch
   composition change the prediction by `< 3e-8`;
3. exactly two directed objects per physical bond; source/target junction
   incidence equals a brute-force enumeration and `deg-1`; padding/cross-graph
   difference `< 2e-8`;
4. with **all token states identical**, the relation still makes the context
   vary across `e` (spread 5.8e-3), zeroing it changes the output by 3.0e-2, and
   shuffling the relation across `f` changes it by 7.7e-3; gradients to
   `rel_b`, `rel_g`, `rel_u` and the relation MLP are finite and non-zero;
5. the encoder weights equal the SSL checkpoint tensor-by-tensor, the optimizer
   covers all 452,523 parameters, and one real step moves an encoder weight;
6. the pooled blocks are reproduced exactly from the trunk states, scale
   linearly/quadratically with token amplitude, and a no-bond molecule gives a
   finite prediction;
7. train labels match PyG exactly, 157 steps/epoch, last batch 16 rows, each row
   exactly once, LR at steps 1 / 1,000 / 235,500 = 3e-7 / 3e-4 / 3e-6 and
   monotone after warm-up.
