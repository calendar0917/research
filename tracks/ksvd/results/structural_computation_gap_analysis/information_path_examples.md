# Information Path Examples

Companion to `notes/structural_computation_gap_analysis.md` (task §23).
Three path classes, described for compact-v4 and for CIN-small from the real
implementations. These are **learned-transformation paths**, not receptive
fields. "Sees the whole graph" is explicitly *not* treated as "can compose the
whole graph the same way".

Notation: `L(x)` = one learned transformation applied to object `x`;
`AGG` = permutation-invariant aggregation (`mean`/`sum`/moment).

---

## Path class 1 — local patch → graph prediction

### compact-v4
```
atom i
  -> patch descriptor (146D, hand-built, no learning)
  -> h_i = L_patch(desc, token, parent)          [1 learned transform]
  -> unary moments AGG_graph({h'_i})             [pooling, no learning]
  -> R
  -> yhat = L_head(R)                            [1 learned transform]
```
Learned depth from a patch to the prediction: **2** (`L_patch` + `L_head`),
plus the one-shot centre update (§2 below). The patch signal reaches the head
only through a permutation-invariant moment, so its individual identity is
gone before the head.

### CIN-small
```
atom i (0-cell)
  -> h_i^0 = Embed_atom(i)                       [lookup]
  -> layer 1: h_i^1 = U_0( B_0(h^0), Up_0(h^0) ) [1 learned transform]
  -> layer 2: h_i^2 = U_0( B_0(h^1), Up_0(h^1) ) [1 learned transform]
  -> rank-0 sum readout -> MLP_R,0 -> final dense
```
Learned depth: **2 message-passing updates + 1 readout MLP**. Crucially the
node state is *re-updated* rather than frozen after one pass, and the update
depends on the node's current context (boundary + upper-adjacency).

---

## Path class 2 — patch A → relation AB → centre B

### compact-v4
```
h_A --pair_projection--> u_A
h_B --pair_projection--> u_B
( u_A , u_B , relation_AB ) --L_pair--> q_AB          [1 learned transform]
q_AB --duplicate to both endpoints--> centre B context
     --AGG(mean,std,log-count per bucket)--> [identity of q_AB LOST]
context_AB_B --L_centre--> delta added to h_B          [1 learned transform]
h_B' --AGG_graph--> unary moments of B --L_head--> yhat
```
The relation object `q_AB` is used **once**; once aggregated, the identity of
`AB` cannot be recovered later in the network. The centre update is a single
shot; there is no path back from `h_B'` to `q_AB`, and `q_AB` is never updated.

### CIN-small
```
edge/atom A (rank p) <-> ring or edge (rank p+1) <-> edge/atom B
each hop is an explicit boundary / upper-adjacency message
h_A^t -> message -> h_sigma^{t+1} -> message -> h_B^{t+1}
```
`A`→`B` can traverse an intermediate cell (the ring), and the intermediate
cell itself carries and updates a state. The incidence `(A, sigma)` stays in
the index tables across layers, so the relation remains queryable.

---

## Path class 3 — higher-order object → lower-order / graph readout

### compact-v4
```
NO higher-order object exists.
cycle/ring information:
atom-level ring membership (implicit in shell descriptor, aggregated)
  + graph-level cycle spectrum/MCB/hinges (25D)
  -> L_topo (552 params) -> z_topology (8D)
  -> concat into R -> L_head -> yhat
```
A ring cannot affect a specific atom's state in a ring-specific way; its
information is either a global scalar (topology channel) or an unstructured
local statistic (shell descriptor).

### CIN-small
```
ring sigma (2-cell) state h_sigma^t
  -> boundary messages to its edges/atoms (higher -> lower)   [learned, every layer]
  -> rank-2 sum readout -> MLP_R,2 -> graph output
```
The ring is a first-class object: it reads from its boundary and writes to it,
and it is read out at the graph level.

---

## Summary

| path | compact-v4 learned depth | CIN-small learned depth |
|---|---|---|
| local patch → prediction | 2 (patch, head) via moments | 2 MP layers + readout |
| patch A → relation AB → centre B | 2 (pair, centre), one-shot, identity lost | ≥2 MP hops, incidence preserved |
| higher-order → lower / graph | none (no higher-order object; global vector only) | 2 MP layers, ring state read out |

Structural receptive field (compact-v4's radius-2 patches + global vector;
CIN-small's whole cell complex) is **not** the same as learned interaction
depth, and neither is the same as object-rank transitions. compact-v4 has a
wide *structural* field but shallow *learned* composition on each object,
because identities are pooled after one pass.
