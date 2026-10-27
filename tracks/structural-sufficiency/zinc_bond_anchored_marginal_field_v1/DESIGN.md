# DESIGN — BAMF-v1: bond-anchored marginal field

## 1. One-sentence design

Keep DCC-L's local composition exactly as it is; add a 12-dimensional per-bond
feature computed from a four-channel binary factor graph whose node and edge
potentials are shared neural functions of the P3 atom/bond states, and compare
**exact whole-molecule marginals (G)** with **independent per-edge marginals (I)**
under identical parameters and initialization.

## 2. What is copied from DCC-L, and what is new

DCC-L's forward is preserved:

```
a, b, w = P3Encoders(batch)
c_glob  = group_composer([b, Wg, Vg, log1p(n_b), bond_feat])
c_loc   = local_composer([b, Wl, Vl, log1p(n_b), bond_feat])
z       = sum_b c_glob
zl      = sum_b c_loc
zl2     = sum_b c_loc^2
yhat    = reader([z, sum_a a, sum_a a^2, global, zl, zl2])
```

Only two things change:

1. `local_composer.lins[0]` is widened from 296 to 308 input columns. The first
   296 columns are copied bit-for-bit from the frozen DCC-L initial state; the
   last 12 are zero. All other DCC-L tensors (including `reader`, whose last 192
   columns are the historical zero-init local/reader expansion) are copied
   unchanged.
2. Two shared potential heads are added:

```
log_phi_v = tanh(Linear(96, 8)(a_v)).reshape(4, 2)
raw_psi_e = tanh(Linear(96,12)(b_e)).reshape(4, 3)
log_psi_e = symmetric 2x2 from (raw_00, raw_01, raw_11)
```

Parameter audit: 494,193 (DCC-L) + 776 + 1,164 + 960 = **497,093**.

At step 0 the marginal columns are zero, so G and I are identical and both
equal DCC-L (verified to 4.8e-7 against the trained historical checkpoint).
The potential heads are the only new randomly initialized tensors; G and I
draw them from the same RNG stream and have bit-identical initial states.

## 3. Why a factor graph, and what “global” means

For one molecule, create one binary variable `s_v` per atom and one pairwise
factor per physical bond. For each of four independent channels:

```
E(s) = sum_v log_phi_v[s_v] + sum_{e=(u,v)} log_psi_e[s_u, s_v]
```

The exact marginal of bond `e` under `p(s) ∝ exp(E(s))` is a sum over all
`2^n` assignments. G computes it without enumerating assignments, by exact
variable elimination on a min-fill junction tree:

- eliminate variables in a min-fill order;
- each elimination creates a clique and a separator;
- upward pass: child clique → parent separator via log-sum-exp;
- downward pass: parent → child, subtracting the child's own upward message;
- calibrated clique belief = node/edge potentials + all incoming messages;
- node/edge marginals are log-sum-exp reductions of the calibrated belief;
- logZ is the log-sum-exp of a root belief, summed over connected components.

This is **algorithmic global inference**. It is not a learned recurrent layer,
and it is not “no propagation”: information travels across the whole molecule
through the elimination messages. It only avoids recursive learned hidden-state
message passing of the kind DCC/BOSI style GNNs use.

Induced width is a property of the molecular graph, not a hyper-parameter. On
all 10,000 official train graphs, a fresh min-fill computation gives maximum
width 3 (counts `[0,55,9906,39]` for widths 0..3), so `smax=4`, state table
size `S=16`, and maximum clique size 4. The plan is built once per corpus and a
batch selects its own sub-forest.

### 3.1 Exactness checks

- float64 full `2^n` enumeration on random connected/disconnected graphs with
  `n=2..8`: logZ, node marginals, and edge marginals agree to ≤1.8e-15.
- batched multi-graph inference equals per-graph inference, including empty,
  single-atom, single-bond, tree, ring, dense, and disconnected graphs.
- invariance to atom renumbering, edge endpoint swap, and bond permutation.
- finite and normalized marginals at potential scale ±100.
- first derivative checked by finite differences; second-order autograd through
  the elimination (including the downward pass) checked by a directional
  finite difference.
- train/eval forward identical (dropout = 0).

## 4. G and I differ only in consistency

Both arms use the same `log_phi`, `log_psi`, and the same feature formula.

**I** treats each bond as an isolated two-node model:

```
p_e = softmax(log_phi_u + log_phi_v + log_psi_e)
```

**G** uses the exact whole-molecule conditional:

```
p_e = exact edge marginal of p(s) ∝ exp(E(s))
```

For a six-node chain, changing only the potential of atom 5 changes G's
marginal on edge (0,1) (five hops away) by 2.9e-5 under the Phase-0 test, while
I's marginal changes by exactly 0. That is the clean mechanism check: G can see
long-range consistency, I cannot.

The per-bond feature is

```
m_e = [ mu_u + mu_v - 1,
        |mu_u - mu_v|,
        4 (mu_uv - mu_u mu_v) ]       # per channel, 4 channels -> 12
```

`mu_u+mu_v-1` is a centered pair mean, `|mu_u-mu_v|` is a disagreement, and
`4(mu_uv-mu_u mu_v)` is the centered pair covariance (for Bernoulli variables
the factor 4 makes it the centered product).

## 5. Gradient path and the zero-init gates

The marginal computation is differentiable end-to-end: no `detach`, no
`no_grad` around the elimination, no cached detached marginals. The graph is

```
a,b -> log_phi,log_psi -> exact marginals -> m_e -> local_composer -> reader -> L1
```

The first two forwards have zero gradient into the potential heads because:

- the reader's new local columns are zero at DCC-L init, so `c_loc` does not
  reach the loss at step 0;
- after the reader columns become non-zero at step 1, the local composer's new
  columns are still zero, so `m_e` does not reach the loss at step 1.

Both new-column groups become non-zero after their first updates; the potential
heads first receive non-zero gradient on the third forward (0-based step 2).
Phase 0 records the exact gradient norms. This is a deliberate consequence of
the frozen zero-init requirement, not the SFC-v0 `log_q.detach()` bug; double
backward through the marginals is separately tested.

## 6. Relation to prior work (no novelty claim for the operator)

Factor graphs, variable elimination, junction trees, and exact edge marginals
are classical. “Strong structure encoding can reduce/replace message passing”
is also not new (Eijkelboom 2023; GRIT/CSA/EGT/MoSE/SPSE/GPM). The contribution
under test is narrower and empirical:

- DCC-L's local content is held fixed;
- a **binary factor graph with learned shared potentials** is attached to it;
- the only experimental variable is exact global marginal inference vs the
  independent per-edge null with the *same* potentials;
- the question is whether that difference buys sample efficiency at fixed
  10k labels on unseen ZINC structures.

Prior internal evidence to respect:

- SFC-v0's exact logZ contractions were computable but gave no sample-efficiency
  evidence; SFC-v0 also had train/eval feature mismatch and a detach bug, which
  this design avoids.
- The DCC information audit found correspondence-aware joint statistics
  non-redundant but not predictive.
- Shared-factor/T-operator gains were around 0.002–0.004, below the practical
  bar and near run-to-run noise.
- The Phase-0 redundancy probe here is sobering: DCC local input predicts G
  features with mean test R² 0.791 and G−I with mean R² 0.525. A large part of
  the field is locally reconstructible. The round proceeds because the protocol
  was frozen to test the residual, not because the redundancy is small.

## 7. Complexity and deployment

Per batch, exact inference costs

```
O( C * sum_cliques 2^{|clique|} + C * batch_plan_selection )
```

with `C=4`, `|clique|≤4`, and ~23 cliques per molecule on average. The plan is
precomputed on CPU; each training batch selects its cliques and moves small
tables to the GPU. The G arm therefore adds a modest, width-controlled cost,
not an all-pairs attention or a dense `n×n` tensor.

The model has 497,093 parameters (single model, no ensemble). The I arm is the
same code path with the exact-marginal call replaced by the two-node softmax;
it is a strong control for “more computation” and “more features”, because it
uses the same 12 features, same potential heads, same parameter count, and same
optimizer budget.

## 8. Failure modes this design is watching

1. **Redundancy:** the marginal features may be reconstructible from DCC local
   moments; Phase 0 quantifies this and the report must not hide it.
2. **Delayed potential learning:** two zero-init gates delay potential gradients
   by two updates; the fixed 47,400-update budget must be enough.
3. **Optimization noise:** a 0.001–0.003 difference is not a result. The gates
   require 0.004–0.006 and both seeds.
4. **GPU overhead:** exact inference on tiny tables can be launch-bound. The
   timing gate stops if a G run is projected above 2.5 A100·h.
5. **Unseen-structure gain:** average Monitor MAE can improve through easier
   rows. The any-unseen no-penalty subgroup and top-row influence checks are
   mandatory before any Phase-2 recommendation.
