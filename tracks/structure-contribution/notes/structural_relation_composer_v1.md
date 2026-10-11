# cscl-structural-relation-composer-v1 — BJOC, structures and results

Track: `structure-contribution` · Protocol: `configs/structural_relation_composer_v1.yaml`
(frozen before any formal GPU run) · Driver:
`code/run_structural_relation_composer_v1.py` · Objects:
`code/structural_relation_objects_v1.py` · Model:
`code/structural_relation_composer_v1.py` · Tests:
`tests/test_structural_relation_composer_v1.py`

Status: Phase A (audit) done · Phase B (architecture) done · Phase D
(synthetic) done · Phase E–G in progress.

---

## 0. The question

Full-Y pools atom-pair states into **graph-level distance-bucket moments**.
Two structurally different organisations that share the same multiset of
local pair states (and the same distance-bucket histogram) produce the
**same** reader input.  The round asks whether retaining the *ownership*
between local relations and their structural anchors — which junction
belongs to which bond — buys generalisable predictive value.

Candidate hypothesis **H-ORG**: the organisation of relations matters.
Competing hypothesis **H-REP**: the local chemical representation is the
binding constraint, organisation is second-order.

This is tested with **two arms that share every parameter tensor and differ
only in how junctions are attached to bonds**:

* `organized` — true junction-to-bond ownership,
  `W_b = Σ_{j touches b} w_j`, `V_b = Σ w_j²`;
* `null` — every bond gets the graph-mean junction content scaled by its own
  junction count, `W_b = n_b · mean_graph(w)`, so that
  `Σ_b W_b` and `Σ_b V_b` are **exactly conserved**.

The null arm is *not* a zeroed or smaller model: it keeps all atoms, all
bonds, all junctions, all trainable weights and all graph-level moments.

## 1. Architecture (BJOC)

One-shot compute DAG (no message passing, no attention, no recurrent state):

```
raw graph (x, edge_index, edge_attr)
  -> atom features (fixed graph algorithms, label-blind)         108 raw dims
  -> a_i  = ATOM(raw_i)                                          shared MLP
  -> b_uv = BOND(a_u, a_v, raw bond)                             symmetric
  -> w_ijk = JUNC(a_i, a_j, a_k, b_ij, b_jk, raw junction)       symmetric
  -> static incidence grouping (no learned parameters)
  -> c_b  = PSI(b_b, W_b, V_b, log1p n_b, raw bond)
  -> Z    = [Σ_b c_b, Σ_i a_i, Σ_i a_i², global topology]
  -> ŷ    = READER(Z)
```

Object layers:

| object | content | count (10k train) |
|---|---|---|
| atom | type one-hot (28), degree, ring membership and min ring size, incident bond-type histogram, radius-1/2 neighbour-type histograms, length-2 paths, triangles, fusion/terminal roles | 231,664 |
| physical bond | every unique undirected bond once (PyG double storage deduplicated and validated), type, ring role flags, endpoints' atom embeddings | 249,279 |
| junction | unordered neighbour pair `(i,j,k)` at centre `j` once, explicit centre, both real bond embeddings, symmetric bond-type pair, endpoint adjacency/common-neighbour/ring features | 346,358 |

Parameter inventory: atom 26,336 · bond 62,816 · junction 112,896 ·
composer 62,976 · reader 148,481 · **total 413,505** (Full-Y comparison:
408,651).

Why it is not a GNN: `a` is computed from fixed raw features; `b` from `a`;
`w` from `a` and `b`; grouping and the composer never feed back into an
earlier object.  The dependency DAG has depth one per object class and there
is no neighbour-to-neighbour state refresh.

Why it is not the historical FullY-StaticTriplet: the triplet branch only
concatenated `[Σq, Σq²]` of *all* triplets into the reader; here junctions
are grouped by their real bond anchors and only then composed per bond, so
the ownership pattern is represented, not just the junction multiset.

## 2. Phase A — CPU audit (`results/structural_relation_composer_v1/phase_a_audit.json`)

**Full-Y information flow.** Reader input `[unary 289 | pair moments 5×97 |
global 32 | topology 8] = 814`; pair pooling is `sum(h)`, `sum(h²)`,
`log1p(count)` per hard distance bucket (C6_MASK zeroes counts).  After that
summation the reader cannot recover (a) which pair states shared a centre
atom, (b) which bonds the states belonged to.

**Concrete counterexample (new).**  Four bonds `{P,P,Q,Q}`, four junctions
`{X,X,Y,Y}`, every bond touching exactly two junctions:

| system | per-bond junction tokens |
|---|---|
| A | P:{XX} P:{YY} Q:{XY} Q:{XY} |
| B | all four bonds {XY} |

Bond multiset, junction multiset and per-bond junction counts are identical;
the ownership differs.  With an untrained BJOC: organized `W_b` differ by
3.3e-2 while null `W_b` are bit-identical (0.0).  The incidence target
(A: 56, B: 54) differs.

**Negative control.**  The alternating molecular relabel `X-Y-X-Y` vs
`Y-X-Y-X` is *not* observable at graph level: reversing the chain also swaps
the two junction roles, so the sorted per-bond content multiset is unchanged.
Recorded so that ownership claims are not over-generalised.

**Label audit (official train only).**

* 9,996 unique canonical SMILES; 4 duplicate groups (8 rows).
* 1 duplicate group has inconsistent `y` with spread 0.0197
  (`O=C(N[C@H]1CCCCC[C@H]1C(=O)[O-])c1cc(F)ccc1Br`, rows 941/2264).
* 3 one-WL collision groups; **0** with inconsistent `y`.
* y: mean 0.0153, std 2.011, min −42.04, p05 −3.55.

**Historical internal-dev error structure (2,000 rows, Full-Y soup).**
MAE 0.11589; corr(|e|, y) = −0.319; the `y ≤ −3` tail (6.4% of rows) carries
17.8% of error mass.  By atom count: q1 (≤20) 0.1497, q2 0.1097, q3 0.0834,
q4 0.1284.  By cyclomatic number ≥3 group: 0.1181 and 59.5% of error mass.
The dominant error source is therefore *not* just the low-y tail.

## 3. Phase D — synthetic known-truth experiment (`synthetic_phase_d.json`)

Object-level incidence systems (`random_object_instance`, 4–8 bonds, 4–10
junctions, 3 tokens).  Target standardisation uses train mean/std.  150
epochs, widths 16–48, 3 seeds for the fitting grid, 5 seeds for the final
incidence grid; CPU only.

| target | organized MAE | null MAE | direction |
|---|---:|---:|---|
| incidence (ownership-dependent) | 0.0761 / 0.0946 / 0.1172 / 0.0803 / 0.1155 | 0.3992 / 0.3737 / 0.2914 / 0.4293 / 0.3525 | **O wins all 5 seeds** |
| marginal-only (control) | 0.0289 / 0.0338 / 0.0426 / 0.0217 / 0.0545 | 0.0658 / 0.0280 / 0.0213 / 0.0583 / 0.0462 | tie / N slightly ahead |

Random *molecular* graphs with a molecular ownership target: organized
0.0186 vs null 0.0296.

Interpretation: the null arm is not capacity-limited (it fits marginal
targets at least as well); it loses specifically when the label depends on
which junction belongs to which bond.  This proves BJOC **can represent**
the ownership function and that the two arms are genuinely different
instruments — it does **not** show that ZINC properties reward ownership.

## 4. Phase E — new internal split and label boundary

`split_v1.json`: grouped by canonical SMILES, group order
`sha256("cscl-structural-relation-composer-v1"|smiles)`, greedy fill.
Actual sizes are recorded in the file (fit ≈ 8000, monitor ≈ 1000, locked
dev = remainder).  Fit-only preparation: global-topology standardiser,
Full-Y patch/ctx/topo/anchor statistics, K-SVD dictionary and common
subspace, all fit on the **fit** rows only.  Locked-dev labels are read only
in the `eval-locked` stage.  Official valid / official test are never
loaded.

## 5. Phase F–G — formal arms and results

(to be filled after the frozen GPU runs)

## 6. Phase I — mechanism diagnostics

(to be filled)

## 7. Evidence boundary

* single seed per arm is not training-stability evidence;
* locked dev is an isolation split inside official train, not a fresh blind
  test;
* the synthetic result shows representability, not chemical causality;
* no attribution claim (α/δ/γ) is made in this round: the reader is a plain
  MLP, so intermediate vectors are **not** additive contributions.
