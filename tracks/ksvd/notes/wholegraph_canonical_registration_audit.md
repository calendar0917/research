# Whole-graph canonical coordinates vs. a graph-specific registration `P_G` (ZINC)

**Question.** For ZINC-sized attributed graphs, after strict canonical labeling
and placement into a shared whole-graph sparse dictionary, are the canonical
coordinates stable enough on their own — or is a graph-specific registration
matrix `P_G` needed to fix slot correspondence across different molecules?

**Status.** Representation analysis only. No model was trained, no predictor was
fit, no architecture experiment was run. This note does **not** modify
`STATE.yaml`, any pre-registration, or any model implementation.

---

## 0. One-line answer

> **Registration is too unconstrained to be useful** — in the tested form.

Canonicalization *does* fully remove the input node-order nuisance (Q1 = yes),
but it does **not** by itself provide a usable cross-graph coordinate system
(Q2 = no). The identity-anchored near-canonical `P` produces **exactly zero
discretizable correction** (its Hungarian rounding is the identity on 100% of
pairs) and its apparent soft improvement is the *same* on random pairs as on
similar pairs (Δ_near ≈ 0.27 both). A freer registration does reduce mismatch on
similar pairs a little more than on random pairs (Δ_free_disc 0.51 vs 0.42), but
it simultaneously wipes out 42% of the mismatch of *arbitrary* graph pairs — i.e.
it mostly imposes similarity rather than recovering it. So the current `P_G`
form is **not** justified, and canonical-only is **not** sufficient by itself.

---

## 1. Provenance and data discipline

| item | value |
|---|---|
| repo commit (analyse revision) | `0e3c02a6db70181486b6826d1a0b9fabee36f252` |
| worktree at run start | clean except the new analysis script (untracked) |
| splits used | **official `train` (10 000) + `val` (1 000)** |
| official `test` | **never read / instantiated / referenced** |
| target `y` | **never used** |
| data source | `data/ZINC` (`torch_geometric.datasets.ZINC`, `subset=True`) |
| loader | `tracks/ksvd/experiments/luyin16/zinc_long_range_proxy._load_zinc` + `_data_to_graph` |
| Python / numpy | 3.12.14 / 2.1.3 |
| canonicalizer | `pynauty` **2.8.8.1** (pinned), colored-incidence `canon`, `pynauty.canon_label` |
| script | `tracks/ksvd/code/run_wholegraph_canonical_registration_audit.py` |
| results | `tracks/ksvd/results/wholegraph_canonical_registration/` (git-ignored) |
| seed | 20260918 |
| command | `PYTHONPATH=. uv run python tracks/ksvd/code/run_wholegraph_canonical_registration_audit.py` |

Reproduce the run with `--quick` for a ~80 s smoke version.

### 1.1 Actual graph encoding (not the "textbook" ZINC definition)

Read from the repo's own loader, per graph `G=(X,E)`:

* `x` is an integer **atom type** tensor, shape `(n,1)`; observed categories
  `{0,…,20}` → **21 node categories**.
* `edge_attr` is an integer **bond type**; observed categories `{1,2,3}` →
  **3 bond categories**.
* adjacency is stored as `edge_index` (both directions) and flattened to an
  undirected bond map by `_data_to_graph`; bonds are simple (no self-loops).
* graph size over train+val: `n_min=9`, `n_median=23`, `n_95%=31`, `n_99%=34`,
  `n_max=37`. **Padding `q=37`.**

Tensors used below: node one-hot `X_G ∈ {0,1}^{q×22}` (21 atom channels + one
explicit `EMPTY`/padding channel), and pair relation
`R_G ∈ {0,1}^{q×q×4}` (channel 0 = "no edge", channels 1–3 = bond types; pairs
involving a padded slot are separated by an explicit `valid` mask).

---

## 2. Exact canonicalization

For a permutation matrix `Π`, we need `C(ΠG) = C(G)`. WL hashes / sorting
heuristics are **not** used. Instead the attributed graph is converted to a
colored incidence graph

```
atom_i —— bond_e —— atom_j
```

* atom vertex colour `("n", atom_type)`,
* bond vertex colour `("e", bond_type)`,
* incidence edges.

`pynauty.canon_label` returns a canonical vertex order; the relative order of
the atom vertices is the canonical atom ordering, and
`certificate + canonical colour sequence` is a complete isomorphism key. This is
the same exact colored-incidence construction already used in the repo by
`tracks/ksvd/experiments/luyin16/typed_patch_tokenizer.py`. `C(G)` (canonical
node-type sequence + canonical bond tensor) is invariant by construction; the
tests and Sanity check A verify it.

**Not used:** RDKit canonical ranking (RDKit is not installed), any sorting
heuristic, any WL hash as an "exact" label.

---

## 3. Sanity check A — random relabelling invariance (implementation test)

500 train graphs × 20 random node permutations = **10 000 checks**.

| metric | value |
|---|---|
| canonical representation match rate | **1.0000** |
| failures | 0 |

Exact canonicalization of the *attributed* graph is permutation-invariant. ✓

## 4. Sanity check B — small perturbations

For 400 graphs, one random instance of each perturbation; nodes are tracked by
original identity. `q=37` is used to normalise displacement.

| perturbation | slot preservation (mean) | normalised displacement (mean) | Kendall τ (mean) |
|---|---|---|---|
| change 1 node type | 0.358 | 0.043 | 0.823 |
| change 1 bond type | 0.491 | 0.034 | 0.859 |
| delete 1 non-bridge edge | 0.487 | 0.033 | 0.864 |
| add 1 leaf | 0.429 | 0.029 | 0.899 |

A single local edit reshuffles a large fraction of canonical slots (only ~36–49%
of atoms keep their slot), even though the *relative order* is largely preserved
(τ ≈ 0.82–0.90). Canonical coordinates are therefore **sensitive to small
structural changes** but are not "chaotic".

## 5. Permutation control (validates the tensor transforms)

For a graph and its non-canonicalized random permutation `ΠG`:

| quantity | mean | median | p90 |
|---|---|---|---|
| `d_I` (raw, not canonicalized) | 0.483 | 0.480 | 0.562 |
| `d_free` soft (raw) | 0.008 | 0.002 | 0.025 |
| `d_free` discrete (raw) | 0.015 | 0.002 | 0.047 |
| `d_I` after canonicalization | **0.000** | 0.000 | 0.000 |

The transform direction (`X→PX`, `R→P R Pᵀ`), the Sinkhorn solver and the loss
are correct: raw identity is far from 0, free registration recovers ≈0, and
canonicalization gives exactly 0. ✓

---

## 6. Similar vs. random pairs and canonical correspondence

**Pair selection (no target, no canonical tensor).** A 2000-graph train+val
pool is fingerprinted with a typed **WL subtree kernel** (rounds 0–3, per-round
normalised colour histograms; bond types enter the refinement). Top
nearest-neighbour pairs with `subtree-cosine ≈ 0.98–0.99` and **different
canonical keys** form 350 *similar* pairs; 345 *size-matched* random pairs
(atom count ±1) form the control. The node-colour-only histogram was rejected as
degenerate (54% of graphs share an identical histogram); the subtree kernel is
not.

**Correspondence.** Without RDKit/MCS we use the allowed auxiliary: joint typed
WL on the disjoint union, matching atoms with equal WL colour (tie-broken by a
Hungarian assignment on neighbour-colour histograms). WL rounding `k=1` and
`k=2` are both reported.

| WL rounds | group | pairs | matched atoms | exact-slot agreement | norm. slot displacement | Kendall τ |
|---|---|---|---|---|---|---|
| 1 | similar | 350 | 14.2 | 0.166 | 0.062 | 0.714 |
| 1 | random | 330 | 9.0 | 0.148 | 0.070 | 0.739 |
| 2 | similar | 283 | 5.8 | 0.180 | 0.060 | 0.572 |
| 2 | random | 121 | 4.1 | 0.118 | 0.071 | 0.627 |

For atoms that are *WL-equivalent across the two molecules*, the canonical slot
agrees only **15–18%** of the time, with a normalised displacement of ~6% of `q`
and relative-order τ ≈ 0.57–0.71. Similar pairs are only marginally better than
size-matched random pairs. **Canonical slots are not a stable cross-graph
correspondence.**

---

## 7. Registration necessity test

Alignment loss (fixed for every experiment):

```
L_align(G,H)[P] = L_node + L_edge
L_node = mean_s 0.5·||X_G[s] − (P X_H)[s]||²                ∈ [0,1]
L_edge = mean_{valid pairs} 0.5·||R_G − P R_H Pᵀ||²         ∈ [0,1]
```

(both are *per-item means*, so the O(q²) pair term cannot swamp the O(q) node
term). Registration `P = Sinkhorn(θ)` is optimized per pair from
`θ₀ = 4·I` (so `P⁽⁰⁾≈I`), Adam lr 0.3, 200 steps, 30 Sinkhorn iterations.

* **A. Identity** — `P=I`.
* **B. Near-canonical** — `ρ=2.0` (identity penalty) and `λ=2.0` (diagonal
  transport bias `Σ P_ij |i−j| / q²`).
* **C. Free** — `ρ=0`, `λ=0` (upper bound, **not** a candidate model).

In addition to the soft loss we report an honest **discrete** loss obtained by
Hungarian-rounding `P` to a permutation — this removes soft-smearing slack.

| group | d_I | d_near | Δ_near (soft) | Δ_near **disc** | d_free | Δ_free (soft) | Δ_free **disc** | transport(C_near) | perm-disp(free) |
|---|---|---|---|---|---|---|---|---|---|
| similar (n=350) | 0.288 | 0.208 | 0.269 | **0.000** | 0.058 | 0.795 | **0.514** | 0.0048 | 0.068 |
| random (n=345) | 0.351 | 0.252 | 0.278 | **0.000** | 0.092 | 0.737 | **0.415** | 0.0066 | 0.073 |

Bootstrap 95% CI for the similar − random gap:

| metric | gap | 95% CI |
|---|---|---|
| Δ_near (soft) | −0.009 | [−0.015, −0.002] |
| Δ_near (discrete) | **0.000** | [0.000, 0.000] |
| Δ_free (discrete) | +0.099 | [+0.084, +0.114] |

**Control (Section 10) reading.** If near-canonical `P` encoded real semantic
correspondence, Δ_near should be *much larger* on similar than random pairs.
Instead `Δ_near_disc = 0` for **100%** of both groups (the rounded near `P` is
the identity), and even the soft Δ_near is marginally *worse* for similar pairs.
Meanwhile the free `P` attains 51% (similar) vs 42% (random) discrete reduction
— a real but small (~10 pp) preference sitting on top of a 42% reduction that it
achieves on completely unrelated molecules.

---

## 8. Answers to the four required questions

* **Q1 — Does exact canonicalization fully solve the input node-order nuisance?**
  **Yes.** 10 000/10 000 relabelling checks match (rate 1.000), and the
  permutation control gives exactly 0 identity distance after canonicalization.
* **Q2 — Do canonical slots provide stable cross-graph correspondence for
  similar (non-isomorphic) molecules?** **No.** For WL-equivalent atoms the
  exact slot matches only 15–18% of the time (norm. displacement ~6%, τ≈0.6),
  barely above size-matched random controls. Canonicalization solves
  *invariance*, not *correspondence*.
* **Q3 — Does near-canonical `P_G` preferentially reduce similar-pair mismatch?**
  **No.** Δ_near is indistinguishable between similar and random (soft 0.27 vs
  0.28, gap CI excludes 0 only in the *wrong* direction), and the discretizable
  gain is exactly 0 for both. The soft gain is relaxation slack.
* **Q4 — If `P_G` helps, what capacity does it need?** Only a **free**
  registration helps, it needs only small displacements (mean |i−perm(i)|/q ≈
  0.07 ≈ 2.6 slots of 37), yet it helps random pairs almost as much as similar
  ones. So the required capacity is "almost arbitrary small permutation", not a
  bounded near-diagonal correction.

## 9. Architecture implication

The pre-registered heuristic in the task request maps to:

* **Case D** (registration helps random graphs about as much as similar ones):
  the current `P` form is too free and would swallow graph identity.
* with a **Case C** component (only free registration moves the number at all;
  canonical coordinates alone are not a usable coordinate system).

Recommendation for the whole-graph dictionary:

1. **Do not add the tested near-identity `P_G`.** It has zero discretizable
   effect and no similar-vs-random preference.
2. **Do not add a free soft-permutation `P_G` either.** Its random-pair reduction
   (42%) shows it imposes rather than recovers correspondence.
3. Since canonical coordinates do not yield cross-graph correspondence, prefer a
   **permutation-invariant whole-graph representation** (e.g. a bag/multiset of
   canonical sub-structures, or a permutation-invariant reconstruction
   objective) instead of slot-indexed coordinates. If a registration is still
   wanted, it must be *correspondence-aware and constrained by structure*
   (GW/FGW-style), and its quality must be checked against the random-pair
   control — the near-diagonal Sinkhorn variant does not clear that bar.

---

## 10. Limitations

* **No RDKit / MCS.** RDKit is not installed and `networkx`'s maximum-common-
  subgraph search is exponential (it hung on unrelated pairs), so the gold atom
  correspondence is approximated by *WL-equivalent* matching. This can only
  *under*-state true correspondence; the similar-vs-random conclusion is robust
  to using `k=1` or `k=2`.
* **"Similar" is a fingerprint proxy** (typed WL subtree kernel), not chemical
  MCS similarity. Similar and random pairs are ~0.99 vs lower in this metric;
  the selected pairs are genuinely near-neighbours, but no chemometric
  ground truth is available.
* The alignment loss is a binary one-hot mismatch; alternative continuous
  feature distances could change the soft magnitudes (but not the discrete
  conclusion, since Δ_near_disc is exactly 0).
* Registration is optimised per pair independently with a fixed budget; the
  free case is an *optimised upper bound*, not a candidate architecture.
* One seed / one pool. The gaps are large relative to the bootstrap CIs, but
  this is a single pooled run.

## 11. Artifacts

* `tracks/ksvd/code/run_wholegraph_canonical_registration_audit.py` — analysis script.
* `tracks/ksvd/tests/test_wholegraph_canonical_registration.py` — 5 data-free correctness tests (all pass).
* `tracks/ksvd/results/wholegraph_canonical_registration/` (git-ignored):
  `results.json`, `pair_metrics.csv`, `perturbation_metrics.csv`, `run.log`,
  `plots/` (graph-size, perturbation displacement, correspondence displacement,
  Δ_near/Δ_free similar-vs-random, Δ vs transport cost).
