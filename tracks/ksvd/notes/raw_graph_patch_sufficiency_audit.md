# Raw-Graph → Patch-System Sufficiency Audit

**Date:** 2026-09-26
**Scope:** zero-full-training / representation-family audit
**Module:** `tracks/ksvd/experiments/luyin16/zinc_raw_graph_patch_sufficiency_audit.py`
**Results:** `tracks/ksvd/results/raw_graph_patch_system_sufficiency/`
**Tests:** `tracks/ksvd/tests/test_raw_graph_patch_sufficiency_audit.py` (20 pass)
**Official valid:** never loaded. **Official test:** never loaded.

**Verdict: Decision Case D — `CURRENT_PRE_NEURAL_PATCH_SYSTEM_SUFFICIENCY_NOT_REFUTED`.**

Neither a material hard pre-neural aliasing signal (`LB(P3) = 0.0 < 0.002`) nor
a material soft geometry degradation (`Delta_pre = -0.0878 < 0`) is present.
The raw structural neighbourhoods are *less* target-local than the current patch
system, by both generic raw references.  The "where did information get lost?"
line is closed; the justified next question is inductive bias / sample
efficiency, not another structural module.

---

## 1. Motivation

Every local architectural repair in the compact-v4 line has returned a clean or
sub-threshold NO-GO (P1 learned composer, P2 one-shot relation refresh, cycle
cells, covariance, triad, endpoint association, attribute factorization, FM
head, function basis, broad frozen-state screen).  The stagewise
representation-collision audit then found **no clear stagewise collision**
*after* `X_patch`.  The one boundary that had never been audited is the one
*before* the learned computation:

```
G_raw  ->  F_pre(G)  ->  X_patch  ->  h0 -> ...
          ^^^^^^^^^^
          this audit
```

The question is whether compact-v4's conversion of a raw molecular graph into
its current patch / relation / global structural object system already causes
material hard aliasing or a stable target-relevant geometry degradation.

## 2. Why checkpoint optimization was deprioritized

The Top-5 checkpoint-aggregation audit established a real but modest
estimator-stabilization effect (`Delta_soup ~ +0.0045` official-valid, locked
development confirmation).  That is **protocol hygiene**, not a representation
hypothesis.  It cannot explain the ~0.02 architecture-scale gap and is not used
here as a main line.  No K sweep, no EMA/SWA, no weighted soup, and no new
backbone training were performed.

## 3. What the previous stagewise audit did not test

It started from `X_patch`, which is already a human structural abstraction.  It
could therefore only detect geometry destroyed by the *learned* modules.  It
could not test whether the abstraction itself (`G_raw -> X_patch`) is
sufficient, because `X_patch` was taken as the input.  This audit moves the
boundary back to the raw atom/bond graph.

## 4. Raw molecular graph definition

For the official PyG ZINC `subset=True` train split (10,000 molecules):

* `G = (V, E, X_V, X_E)`;
* `V` = atoms, `X_V` = the single dataset-provided scalar `data.x` = atomic
  number;
* `E` = bonds, `X_E` = the single dataset-provided scalar `data.edge_attr` =
  bond type;
* `data.y` = penalized logP (target, read only in the target phases).

No external chemistry (Morgan / RDKit / fingerprints / target-derived features /
external embeddings) is used anywhere.  See `raw_graph_inventory.json`.

## 5. Exact current preprocessing pipeline

Reconstructed from the real code path (`zinc_patch_path_pooling.py`,
`zinc_post_v4_residual_audit._extract_v4_records`, `zinc_topology_features.py`):

1. every atom is a patch centre; the patch is the radius-2 ego-net;
2. `typed_certificate = pynauty.certificate(rooted typed incidence graph)`
   (historical tokenizer; the bare certificate does **not** encode the vertex
   colouring); `parent_certificate` = radius-1 certificate;
3. `shell_descriptor` = 146D continuous shell/root/size-cycle descriptor;
4. every unordered patch pair ↔ one relation object (23D): distance bucket
   one-hot + `log1p(distance)` + 5 overlap features + 3 boundary features + 4
   path bond-composition features + `log1p(path count)` + 4 adjacent-bond
   features; plus the integer distance bucket;
5. `global_context` = 62D graph size/degree/distance/clustering + atom/bond
   histograms;
6. `topology_features` = 25D cycle spectrum / longest cycle / MCB statistics /
   hinge basis;
7. token vocabularies and patch/global/topology standardizers are fit on the
   full official train, target-independently.

## 6. Model-accessible pre-neural inventory

`pre_neural_input_inventory.json` enumerates every deterministic field that can
affect the forward pass before the learned modules: `typed_token`,
`parent_token`, `patch_cont` (146D), `patch_context` (width 0 in
compact-v4-hinge), `pair_index`, `pair_relation` (23D), `pair_bucket`,
`global_context` (62D), `topology_features` (25D), `num_nodes`.  Learned state
(`h/q/R`) is explicitly excluded — it belongs to the post-pre-neural stages.
The topology channel carries a **provenance caveat**: it was historically
motivated by a long-cycle target audit; it is still a genuine model-accessible
input and therefore belongs in P3.

## 7. P0 / P1 / P2 / P3 signatures

* **P0 (token-only):** sorted multiset of `(typed_certificate,
  parent_certificate)` per molecule.
* **P1 (full patch-object system):** sorted multiset of `(typed_certificate,
  parent_certificate, shell_descriptor bytes)`.
* **P2 (patch + relation system):** canonical key of the labelled
  patch-relational object system — patch nodes labelled by P1, every unordered
  patch pair an edge labelled by `(relation descriptor bytes, bucket)`.
* **P3 (full model-accessible pre-neural system):** P2 + `global_context` bytes
  + `topology_features` bytes.

Only P3 can support a hard expressivity claim; P0/P1/P2 localise which
deterministic channels resolve earlier ambiguity.

## 8. Permutation-invariant canonicalisation

Signatures must not depend on PyG node ordering.  The raw graph and the P2
object system are encoded as **vertex-coloured incidence graphs** (edge labels
become edge-vertices, exactly the construction already used by the repository's
typed tokenizer).  The canonical key is

```
pynauty.certificate(graph)  ||  repr(canonical semantic colour sequence)
```

which is an **exact** complete invariant of the coloured graph (the bare
certificate omits colour labels; appending the canonical colour sequence
repairs this).  Invariance under node relabelling is verified on 20 real
molecules (0 failures) and on random coloured graphs; see
`integrity_tests.json`.

## 9. Hard collision protocol

For each level: group all 10,000 official-train molecules by signature; report
unique-signature count, non-singleton classes, collision mass, max class size,
and signature digest.  A class whose members are pairwise raw-graph isomorphic
is a dataset-ordering duplicate, **not** a representation collision.  Only
raw-non-isomorphic classes count.

## 10. Functional-equivalence verification

A functional check (same weights, forward both molecules, compare `R` and
`yhat`) is required only if a raw-non-isomorphic P3 collision exists.  P3 has
**zero** such collisions, so the check is vacuous and recorded as such in
`functional_collision_checks.json`.  The three P3 non-singleton classes are
exact (VF2-confirmed) copies of the same raw molecule.

## 11. Empirical aliasing L1 lower bound

For each non-singleton class `C`, the optimal deterministic constant predictor
is `m_C = median{y_i : i in C}`, and

```
LB(P) = (1/N) * sum_C sum_{i in C} |y_i - m_C|.
```

`N = 10,000`.  Targets are read only after the hard signatures are hashed and
locked (`hard_signature_lock.json`).

## 12. Hard aliasing result

| level | unique | non-singleton classes | molecules | collision mass | raw-non-isomorphic classes | LB |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 9,995 | 5 | 10 | 0.00100 | **2** | 7.95e-5 |
| P1 | 9,997 | 3 | 6 | 0.00060 | 0 | 0.0 |
| P2 | 9,997 | 3 | 6 | 0.00060 | 0 | 0.0 |
| P3 | 9,997 | 3 | 6 | 0.00060 | 0 | 0.0 |

The two P0 raw-non-isomorphic classes are `{941, 2264}` (targets −3.786 /
−3.806) and `{7805, 9382}` (targets 1.900 / 1.125).  Both are **dissolved at
P1**: the continuous 146D shell descriptor distinguishes the rooted structures
that the historical `pynauty.certificate` token merged.  The three surviving
P3 classes are exact duplicates with **identical targets**, hence `LB(P3) = 0`.

`LB(P3) = 0.0 < 0.002`, so **no material hard pre-neural aliasing signal**.

## 13. Why exact collisions may be insufficient

Exact injectivity is not required for good learning, and absence of collisions
does not by itself prove a representation is well organised for finite-data
learning.  A representation can preserve information while organising it badly.
Phase S therefore asks the *soft* question.

## 14. Raw WL reference (locked)

Edge-aware typed 1-WL subtree count, rounds **0..4** (locked, never swept).
Node label = exact raw atom type; edge label = exact raw bond type; update
`l_v <- hash(l_v, multiset{(l_e, l_u)})`.  `Phi_WL` is the concatenated colour
count over rounds, L2-normalised; distance = `1 - cosine`.

## 15. Raw shortest-path reference (locked)

For every unordered atom pair (canonically ordered endpoint types) record
`(x_u, x_v, d_G(u,v))`, plus atom-type counts and bond-type counts.  Sparse
histogram, L2-normalised, cosine distance.  No target-aware chemistry.

## 16. Current patch-system soft representation

Four real blocks of the current preprocessing:

* **B1 identity:** sparse count histogram over `(rooted token, parent token)`;
  cosine.
* **B2 patch numeric:** 217D projected-quantile sketch (24 fixed Gaussian unit
  projections × 9 quantiles + `log1p(count)`) of the deterministic 146D shell
  descriptor multiset.
* **B3 pair relation:** 217D projected-quantile sketch of the deterministic 23D
  pair-relation multiset.
* **B4 global/topology:** z-scored 87D `[global_context(62); topology(25)]`.

`d_PATCH = sqrt(mean_b d~_b^2)` with equal semantic weight; no target-dependent
or kNN-tuned block weighting.  Progressive variants use B1+B2 (LOCAL),
B1+B2+B3 (PAIR), B1+B2+B3+B4 (FULL).

## 17. Unlabeled Phase U

7200 reference / 2000 primary / 800 replication official-train split (identical
to the frozen stagewise manifest).  WL, SP, B1–B4, all distances and all
neighbour manifests are built **without reading any target**.  Block scales are
medians over 50,000 deterministic random reference pairs
(`distance_scale_stats.json`); B2/B3/B4 z-scoring uses only the 7200 reference
graphs.

## 18. Neighbour-manifest lock

For every representation and both query splits, the k=8 nearest reference
molecules are written to `neighbors_*_{probe,selection}.jsonl.gz` and SHA-256
hashed **before** targets are read (`neighbor_manifest_hashes.json`,
`locked_before_target_read = true`).  k=4/16 are stored as robustness only.

## 19. Target-scoring Phase Y

```
V_8(Z) = (1/N) sum_i (1/8) sum_{j in N8(i;Z)} |y_i - y_j|
eta(Z) = V_8(Z) / V_rand
V_rand from a fixed-seed random 8-neighbour manifest
```

No target is touched before the manifest lock; `tau_far` = 80th percentile of
`|y_a - y_b|` over fixed random reference pairs (`tau_far = 3.356`).

## 20. Raw-vs-patch target locality

Primary 2000 queries (`random_neighbor_baseline.json`: `V_rand = 2.092`):

| representation | V8 | eta | collision rate |
|---|---:|---:|---:|
| RAW (combined) | 1.326 | 0.6337 | 0.0846 |
| WL | 1.560 | 0.7455 | 0.1079 |
| SP | 1.260 | 0.6024 | 0.0806 |
| PATCH_LOCAL | 1.227 | 0.5865 | 0.0623 |
| PATCH_PAIR | 1.232 | 0.5888 | 0.0637 |
| **PATCH_FULL** | **1.142** | **0.5459** | **0.0464** |

800 replication agrees (PATCH_FULL 0.5502 vs RAW 0.6410).  Both generic raw
references are *worse* than the patch system on target locality:
`eta(PATCH_FULL) - eta(WL) = -0.1996` and `eta(PATCH_FULL) - eta(SP) = -0.0565`.
The collision rate also falls from 0.0846 (RAW) to 0.0464 (PATCH_FULL).

## 21. Progressive PATCH_LOCAL / PAIR / FULL analysis

PATCH_LOCAL already beats RAW (0.5865 < 0.6337); adding the pair block is
neutral (0.5888); the global/topology block supplies the large additional
improvement (0.5459).  The correct reading is therefore **not**
"local patch abstraction is a bottleneck": local abstraction alone already
organises ZINC better than the generic raw metric, and the existing global
channels recover further.  No redesign of the local patch is warranted by this.

## 22. Bootstrap and replication

Paired bootstrap over the 2000 query molecules (`B = 2000`, fixed seed):
`Delta_pre = eta(PATCH_FULL) - eta(RAW) = -0.0878`, 95% CI
`[-0.0999, -0.0748]`, `P(>0) = 0`.  The 800 replication is
`Delta_pre = -0.0909` (same direction).  `knn_predictor_diagnostics.json`
(supportive only): NN8 median MAE 1.002 (RAW) → 0.878 (PATCH_FULL).

## 23. Hard-vs-soft interpretation

* Hard: `LB(P3) = 0.0 < 0.002` → no material hard aliasing.
* Soft: `Delta_pre = -0.0878 < 0` → no material soft degradation; the patch
  system is more target-local than the generic raw references.
* Collision metric agrees with `eta` (PATCH_FULL collision rate lower), so
  `metric_inconsistent = false`.
* WL and SP agree in direction, so `raw_reference_unstable = false`.

Evidence supports **neither** hard expressivity loss **nor** soft
factorization/sample-efficiency loss at the pre-neural boundary.

## 24. Existing NO-GOs that remain binding

No result here reopens: P1 learned composer, P2 relation refresh,
compact-v4-cell cycle cells, covariance/triad/endpoint witnesses, corrected
tokenizer as a performance fix, larger patch radius, FM head / function basis,
or attribute factorization.  The corrected-tokenizer performance NO-GO remains
in force; the P0 token alias found here is *resolved* by the shell descriptor,
so it does not resurrect the tokenizer-expressivity candidate.

## 25. What is and is not proven

**Proven (official train only, zero training):**

* the model-accessible P3 pre-neural representation has zero
  raw-non-isomorphic collisions on the 10K official-train molecules;
* the historical tokenizer alias is real at P0 (2 raw-non-isomorphic classes)
  but is fully resolved by the continuous patch descriptor channel at P1;
* on a frozen target-independent split, the current patch-system geometry is
  more target-local than either generic raw-graph reference view;
* all Phase-U integrity gates and all 18 listed integrity tests pass.

**Not proven / not claimed:**

* that the patch system is information-theoretically lossless;
* that no alternative raw metric could reverse the soft ranking;
* that official valid or official test behave the same;
* that the representation is optimal for finite-data learning (that is the
  *sample-efficiency* question);
* that a new architecture would help.

## 26. Implication for representation redesign

**No representation-family redesign is authorized.**  `top1_pre_neural_hypothesis.json`
has `authorized_for_design = false`, `full_training_authorized = false`.  The
pre-neural boundary is not refuted.

## 27. Implication for sample-efficiency research

Per the protocol, once the complete pre-neural representation shows neither
material hard aliasing nor stable soft degradation, the scientifically
justified next question is

```
why does the same information need more samples under this factorization?
```

and the correct next stage is an **inductive-bias / sample-efficiency audit**,
not a new structural feature search.

## 28. Final verdict

**Decision Case D — `CURRENT_PRE_NEURAL_PATCH_SYSTEM_SUFFICIENCY_NOT_REFUTED`.**

```
P3 hard LB           0.0        (< 0.002 material gate)
Delta_pre            -0.0878    (95% CI [-0.0999, -0.0748])
replication          -0.0909    (same direction)
WL / SP              both say patch is more target-local
collision metric     consistent
```

Stop hunting for missing structural information.  Next: inductive bias /
sample efficiency.

---

## Q1–Q20

* **Q1** Raw ZINC node feature = atomic number (`data.x`); edge feature = bond
  type (`data.edge_attr`); no external descriptors.
* **Q2** `pre_neural_input_inventory.json`: tokens, 146D patch descriptor, pair
  index/relation/bucket, 62D global, 25D topology.
* **Q3** P0 token multiset; P1 patch-object system; P2 + relation system; P3 +
  global/topology.
* **Q4** Yes — every field in the inventory is covered by P3; the controlled
  perturbation test changes the expected signature level for 100/100 sampled
  molecules on all seven field families.
* **Q5** P0: 5 non-singleton classes, mass 0.00100, LB = 7.95e-5, 2
  raw-non-isomorphic classes.
* **Q6** P1: 3 classes (all raw-isomorphic), mass 0.00060, LB = 0.0.
* **Q7** P2: identical to P1; LB = 0.0.
* **Q8** P3: identical to P1; LB = 0.0.
* **Q9** No raw-non-isomorphic P3 collisions.
* **Q10** Functional check vacuous (not triggered).
* **Q11** `LB(P3) >= 0.002`? **No** (0.0).
* **Q12** `eta(WL) = 0.7455`.
* **Q13** `eta(SP) = 0.6024`.
* **Q14** `eta(RAW combined) = 0.6337`.
* **Q15** PATCH_LOCAL 0.5865 / PATCH_PAIR 0.5888 / PATCH_FULL 0.5459.
* **Q16** `Delta_pre = -0.0878`.
* **Q17** Bootstrap CI `[-0.0999, -0.0748]`, lower < 0 → no material
  degradation support.
* **Q18** 800 replication `-0.0909`; WL/SP same direction; collision metric
  consistent.
* **Q19** Neither hard expressivity nor soft factorization/sample-efficiency:
  sufficiency not refuted.
* **Q20** Decision Case D.

---

## Figure pointers

* `figures/figure1_architecture_boundary.png` — audit boundary.
* `figures/figure2_hard_aliasing.png` — P0–P3 collision mass and LB.
* `figures/figure3_target_locality.png` — `eta` for all views.
* Figure 4 (paired `Delta_pre` bootstrap) is deliberately not rendered because
  no material signal exists.
