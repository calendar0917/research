# DESIGN — Statistical Operator Closure (SOC-v1)

## 1. One-paragraph summary

DCC-L computes one static incidence aggregation (junction → bond) and one shared
MLP. SOC-v1 replaces that single aggregation with a **weight-tied, three-step
moment-closure iteration** on the same bond/junction incidence graph: every atom,
bond and junction state is repeatedly updated from the first and second moments
of its explicit neighbours, using one shared operator. The backbone, composers,
reader, data, normalisation, SSL initialisation and recipe are the frozen DCC-L
ones. At initialisation the closure gate is ~0.0067, so the model starts at
DCC-L; the operator can only move away from it if that reduces the training L1.

## 2. Why moments, and why weight tying

The prior rounds establish two constraints:

1. **Extra capacity is expensive.** Every added branch (COC, BAMF, native
   relation, DBT) paid 0.004–0.006 MAE in optimization cost, even when the
   branch carried real signal. A new module must be tiny and start at the
   champion.
2. **The missing information is local multi-hop context.** DCC-L already has
   radius-2 junction states and a global moment channel. What it lacks is a
   *repeated* propagation of bond context through the explicit incidence
   structure; the frozen-encoder iteration probe shows that bolting iteration on
   after training does not work, so it must be learned jointly.

Moments (mean, second moment, count) are the smallest sufficient statistics for
a permutation-invariant neighbourhood update. Weight tying (one operator for all
roles and all steps) keeps the parameter count at 18,695 and makes the
computation a genuine fixed-point map rather than K independent layers. There is
no attention: no query/key/value, no softmax, no data-dependent pairwise
weights. There is no global token and no cross-molecule mixing.

## 3. Exact computation

Let `a ∈ R^{Na×96}`, `b ∈ R^{Nb×96}`, `w ∈ R^{Nw×96}` be the P3 encoder states.
The batch provides:

* `bond_index ∈ Z^{Nb×2}`: the two endpoint atoms of each bond.
* `junc_index ∈ Z^{Nw×3}`: the three atoms of each junction patch (centre second).
* `junc_bond ∈ Z^{Nw×2}`: the two bonds incident to each junction.

For an object `o` with state `h_o`, define its two incidence groups:

| object | group 1 | group 2 |
|---|---|---|
| atom | incident bonds `{b_e : v ∈ e}` | empty |
| bond | endpoint atoms `{a_u, a_v}` | incident junctions `{w_j : e ∈ j}` |
| junction | patch atoms `{a_i, a_j, a_k}` | incident bonds `{b_e : e ∈ j}` |

For a group `G` with `n_G` members:

```
m(G) = (1/n_G) Σ_{x∈G} x
s(G) = (1/n_G) Σ_{x∈G} x ⊙ x
c(G) = n_G
```

The shared operator is:

```
x_o = [ h_o, m(G1), s(G1), m(G2), s(G2), log1p(c(G1)), log1p(c(G2)), r_role ]
u_o = W2 · SiLU( W1 · LayerNorm(x_o) )
h_o' = h_o + σ(γ_role) ⊙ u_o
```

with `W1 ∈ R^{24×490}`, `W2 ∈ R^{96×48}`, `r_role ∈ R^8`, `γ_role ∈ R`.
One Jacobi step updates all three object types simultaneously from the same
pre-step states. The same `W1, W2, r, γ` are reused for `t = 1..K`, `K = 3`.
The final states `(a^K, b^K, w^K)` enter the frozen DCC-L composers and reader
unchanged. Arm T instead enters the time mean `(1/(K+1)) Σ_t (a^t,b^t,w^t)`.

### 3.1 Parameter table

| block | shape | params |
|---|---:|---:|
| shared projection `W1` | 490 × 24 + 24 | 11,784 |
| shared core `W2` | 48 × 96 + 96 | 4,704 |
| core hidden | 24 × 48 + 48 | 1,200 |
| LayerNorm | 490 × 2 | 980 |
| role embedding | 3 × 8 | 24 |
| role gates | 3 | 3 |
| **closure total** | | **18,695** |
| DCC-L backbone | | 494,193 |
| **SOC total** | | **512,888** |

CTRL (no closure, reader hidden 274) = **513,021** (+0.026% vs SOC).

### 3.2 Initialisation and identity

The backbone is built by the frozen `dcc_models.build_initial_state` and copied
into the SOC model; only the closure block is new. `γ = -5` gives
`σ(γ) = 0.006693`. Forcing `γ = -30` gives `σ(γ) ≈ 9.4e-14`, and the measured
max abs prediction difference vs DCC-L is **0.0** on 16 real molecules for both
readouts. CTRL's extra reader units are initialised dead (zero outgoing columns),
so CTRL also reproduces DCC-L to 2.4e-7.

The gate is not exactly zero, so the closure receives a gradient from step 1 and
the update is a small perturbation of the champion, not a cold start.

## 4. What SOC is not

* **Not attention.** No pairwise scores, no softmax, no global tokens. The
  neighbour weights are uniform within each incidence group; only the moments
  are learned functions.
* **Not ESA/ESA+/GRIT/CSA/Flownet/GPM.** Those use attention or edge-conditioned
  token mixing and (in ESA+) a much larger pretraining/label budget. SOC uses
  moment closure over a fixed bond/junction incidence graph and is trained from
  the same 10k labels as DCC-L.
* **Not "no message passing".** It is an explicit learned iterative propagation;
  the report must say so.
* **Not a relation partition.** There is no per-relation parameter allocation
  (BOSI/COC/ESOC negative results).
* **Not a label-decomposition model.** It predicts raw `y` only; no cycle
  component, no label-derived inference feature.

## 5. Falsifiable failure modes

1. **Optimization cost**: the 18.7k extra parameters overfit the 8k/10k labels,
   and Monitor/valid is worse than DCC-L. (Most likely, given BAMF/COC.)
2. **Moment blur**: mean/second-moment pooling destroys the bond-specific
   information DCC-L already encodes, so even a well-optimised closure cannot
   help. (The CPU iteration probe is consistent with this.)
3. **Capacity, not mechanism**: CTRL matches the gain. Then the result is a
   capacity effect and must be reported as such.
4. **Screen-only gain**: the Fit8k gain does not survive 10k labels. Then the
   operator is not sample-efficient at the target label count.
5. **Instability**: repeated application amplifies the state. Gates and finite
   checks cover this; a non-finite run is a failed run and is recorded.

## 6. What would make the mechanism credible

* Both seeds improve on the matched Monitor rows, not just the mean.
* The improvement is larger than CTRL's.
* The full-10k official-valid gate passes at >= 0.004 seed-mean improvement.
* The terminal test is < 0.060 with a single model, no ensemble, no test tuning.

Anything less is reported as a negative or a near-miss, with the measured
numbers and the exact code revision.
