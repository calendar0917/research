# Explicit-Support Structural Composer (vocab-free) — ZINC Cell A

**Date:** 2026-09-15
**Branch:** `exp/explicit-support-structural-composer`
**Base revision:** `5b19132` (integrated Agent A / Agent B / identity audit + B-full + B-bag + rank-1 audit)
**Formal training commit:** `c9984cb` (analysis-only diagnostics key fix: `6ea6e9a`)
**Run tag:** `exco-s1` (GPU1, detached), seed1 only
**Remote:** `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`
**Verdict:** **STOP** — seed1 Top-5 soup valid **0.133289** vs B-full seed1 **0.118126** (Δ = **+0.015162**, regression ≫ 0.003). No second seed. Official ZINC test **never loaded**.

---

## 1. Question and representation hypothesis

The completed B-full encoder derives a patch token by propagating hidden node
states over the real rooted typed patch graph. A zero-training audit showed its
16-D `e_struct` is effectively **rank-1** (effective rank 1.107/1.202; rank-1
reconstruction changes the 2-seed soup by ≈ +1.6e-5). The B-bag control (same
primitives, no adjacency) then cost ≈ +0.0013 soup.

This study tests a different representation hypothesis, not a wider encoder:

> **Do not encode the whole patch implicitly through GNN hidden-state
> propagation. Maintain explicit structural objects with exact atom/bond
> supports and learn only which legal local objects are worth composing.**

Core principle: **topology defines which compositions are legal; neural
parameters learn which legal compositions are useful.**

## 2. Pre-registered architecture (exactly one)

```
round 0 : atom primitives                    support = {atom}, bonds = {}
round 1 : legal atom+atom objects            support = 2 atoms, bonds = {bond}
          (one per real patch bond)
round 2 : legal level-1 + level-1 objects    support = exact parent union
          (overlap OR a real new connecting bond; union <= 4 atoms;
           deterministic support de-duplication)
pooling : activity-weighted mean / std over the surviving objects
          + log1p(object count) -> fusion MLP -> e_struct in R^16
```

- **Object latent width** `latent_dim = 8` (pre-registered 4-8); **edge width** 8.
- **Symmetric, order-invariant composition features.** Level-1 input
  `[z_u+z_v ; |z_u-z_v| ; bond_emb ; size_u+size_v ; |size_u-size_v| ;
  root_u+root_v ; |root_u-root_v|]`. Level-2 input
  `[z_a+z_b ; |z_a-z_b| ; overlap ; n_new_conn ; Σ new-conn bond_emb ;
  union_size ; root_a+root_b ; |root_a-root_b| ; a_a+a_b ; |a_a-a_b|]`.
- **Scorer / content**: `a_k = sigmoid(score_MLP(...))` (scalar activity) and
  `z_new = a_k · content_MLP(...)`; support is retained regardless of `a_k`
  (spec §6). Score and content use **independent** MLPs per round.
- **Pooling weight = learned activity `a_k`.** Level selection per patch:
  if the patch has round-2 objects, pool those; else round-1; else round-0.
- **Nothing hidden defines structure:** the composer never reads `struct_src` /
  `struct_dst`; the only structural input is the precomputed, target-free
  candidate table. There is no GNN / GINE / GCN / attention / soft clustering.
- **Everything downstream is inherited bit-exactly from cell A / B-full:**
  radius-2 patch definition, relation system, pair descriptors, `T=2`
  weight-tied recurrent pair–centre, parent embedding, global/topology
  channels, `R = 334`, small head and the frozen optimized protocol.

## 3. Structural object representation (support / provenance)

A structural object is a first-class record `O_k = (S_k^V, S_k^E, z_k, a_k,
parents)`. Supports are discrete, exact and **not latent**:

- **Atom support** — `expand_atom_support(graph, level, index)` recursively
  replays the provenance DAG: atom → `{atom}`; level-1 → `{u, v}`;
  level-2 → union of the two parent supports.
- **Bond support** — `expand_bond_support`: atom → `{}`; level-1 → `{bond}`;
  level-2 → parent bond supports ∪ the **new** connecting bonds (parents' own
  bonds excluded), recorded in `l2_conn`.
- **Provenance** — `parents(graph, level, index)` returns the exact
  `(level, index)` parents; every level-2 object recurses to level-1 and then to
  atom primitives.

The candidate search space is built once, target-free, by
`explicit_support_composer.build_explicit_candidate_graph` and cached
(`explicit_support_candidates_v1`). Real molecular adjacency is used only to
(1) decide which pairs are legal, (2) form the exact support union, and
(3) provide the connecting bond primitives.

## 4. Legal composition rule

`O_a + O_b` is a candidate iff their atom supports **overlap** or there is at
least one **real bond between the supports**, with `|S_a^V ∪ S_b^V| <= 4`.
A candidate is de-duplicated by its exact union support (first ordered pair
wins). `a_k` never creates or deletes support; hard top-k is never used.

## 5. Integrity / adversarial tests (all pass)

`sanity` on the first 128 official-valid molecules: **24/24 pass**
(`results/explicit_support_composer/sanity.json`), including

1. node-relabel invariance (on an ambiguity-free support; unit-tested),
2. every atom support exactly recoverable, 3. every bond support exactly
recoverable, 4. composition support == exact parent union, 5. provenance
recurses to atom primitives, 6. only topology-legal pairs compose, 7. changing
connectivity changes the legal candidate set, 8. no hidden-state message
passing (forward succeeds with `struct_src`/`struct_dst` removed; no
message-passing / attention modules exist), 9. no vocab-sized
`typed_embedding`, 10. zeroing `typed_token` changes nothing, 11. overlap is
allowed and recorded, 12. containment is a deterministic support function,
13. `e_struct` width 16, 14. `h/q/T = 64/16/2`, 15. forward/backward finite,
16. deterministic candidate enumeration, plus recurrent weight tying, parent
path preserved, and the parameter budget.

**Adversarial pair.** Two 6-node rooted trees with the **same**
atom/root/distance/degree primitive multiset and the **same** bond-type
multiset but different connectivity (`0-1,0-2,1-3,2-4,2-5` vs
`0-1,0-2,1-3,1-4,2-5`): the legal composition signature differs
(`connectivity_changes_candidate_set = true`), proving connectivity explicitly
changes the composition search space. A second typed pair (same topology, same
bond-type multiset, different bond-type placement) moves the pooled output by
`1.83e-3`, while the tree pair's pooled output is representation-equivalent
(`1.5e-8`) because its local-object descriptor multiset coincides — the
candidate space still differs.

`tracks/ksvd/tests/test_explicit_support_composer.py`: **18 pass**.
Affected existing suites (B-full, B-bag, rank-1 audit): 46 pass total.

## 6. Parameter accounting (exact)

| Block | Params |
|---|---|
| primitive embeddings (atom/root/dist/degree/bond) | 344 |
| level-1 scorer | 6,001 |
| level-1 content | 7,408 |
| level-2 scorer | 6,601 |
| level-2 content | 8,008 |
| object fusion | 6,782 |
| **composer total** | **35,144** |
| candidate model total | **84,487** |
| cell-A baseline | 85,763 |
| B-full total | 84,495 |
| Δ vs B-full | **−8 (−0.0095 %)** |

The released typed lookup is 36,420 params; the composer is 35,144, a
budget-matched replacement concentrated in the composition scorer / content /
fusion (the spec's §8/§11 intent). No vocab-sized table; no final-head padding.

## 7. Formal one-seed result (seed1)

Deterministic-A100 (`torch.use_deterministic_algorithms(True)`), Adam
`lr=1e-3`, `wd=1e-5`, batch 128, `max_epochs=240`, patience 40, no scheduler,
L1, clip 5, fixed equal-weight Top-5 soup. Seed1 on GPU1 (GPU0 was occupied by
the still-running B-bag seed0; determinism makes GPUs interchangeable).

- seed1 raw (best checkpoint) **0.139464** @ epoch 153; 193 epochs;
  wall **2527.5 s** (13.10 s/epoch); peak GPU **442 MB**.
- seed1 fixed Top-5 soup **0.133289** (soup gains −0.006175 over raw;
  top-5 epochs 153/177/186/169/183).
- Determinism: 6-epoch `repro` run twice on GPU1 → bit-identical
  (`selection_state_sha256 = 3bafae7e8e11c94a…`, best valid 0.380334).

### Matched single-seed comparison (seed1 soup)

| Variant | seed1 soup | Δ vs B-full | params |
|---|---|---|---|
| **B-full** (connectivity-aware encoder) | **0.118126** | — | 84,495 |
| B-bag (connectivity-free ablation) | 0.120229 | +0.002103 | 84,511 |
| A2 (no identity, shared adapter) | 0.122134 | +0.004008 | 85,740 |
| **Explicit composer** | **0.133289** | **+0.015162** | 84,487 |

Δ vs A2 **+0.011154**; Δ vs B-bag **+0.013059**. The composer is worse than
every matched reference, including the connectivity-free B-bag.

## 8. Structural diagnostics (256 official-valid molecules, seed1)

- round-1 candidates/patch **5.29**; round-2 candidates/patch **10.35**;
  selected objects/molecule **238.4**.
- mean round-1 activity **0.499997**; mean round-2 activity **0.521508**;
  overall activity mean **0.5142**, **std 0.01017**, range **[0.49999, 0.52151]**.
- support-size histogram of selected objects `[0, 0, 0, 34522, 26518]`
  (all selected objects are size 3-4).
- atom coverage **1.000**, bond coverage **1.000**;
  pairwise overlap rate **0.936**, containment rate **0.131**;
  duplicate-support rate **0.000**; singleton fraction **0.000**;
  whole-patch-support fraction **0.0042**.
- `e_struct` effective rank **1.045** (top singular fraction **0.9935**).

**Collapse report (spec §13).** Collapse A (singleton atoms) **no**;
Collapse B (whole patch) **no**; Collapse D (duplicate supports) **no**;
Collapse C (constant activity): the pre-registered `std < 0.01` criterion is
**not** formally tripped (std 0.0102), but the activities are *effectively
saturated*: every activity lies in `[0.49999, 0.52151]`, i.e. the scorer
outputs are ≈ 0 and `sigmoid ≈ 0.5`. This is a borderline/near Collapse C and
is the honest mechanism of the regression, not a clean pass.

**Representation-magnitude check (same 256 patches).** The composer's patch
token is essentially constant and small: per-dimension std mean **0.0052**
(max 0.0152), mean token norm **0.0263**, range 0.146. B-full's token is much
larger and more variable: per-dim std mean **0.0556** (max 0.1801), mean norm
**0.1452**, range 1.535. Both are near rank-1 (composer more so: 1.045 vs
1.107). So the composer learned an almost input-independent patch token.

**Interpretation.** The explicit objects, supports, provenance and legal
search space behaved exactly as designed (coverage 1.0, no duplicates, no
singleton/whole-patch collapse). What failed is the *learned composition*:
the activity gate saturated near its initialization and the pooled object
representation collapsed to a near-constant, low-magnitude vector, so the
patch token carries almost no per-patch structural information. This is a
negative result for **this exact composer** under the frozen protocol; it is
**not** evidence that exact-support composition is impossible.

### Training curve

valid MAE at epochs 20/40/80/120/160 = 0.2849 / 0.2618 / 0.1952 / 0.1959 /
0.1467; best 0.139464 @153, then rising to 0.16657 by 193. The model trained
and plateaued — it did not diverge — but converged to a worse solution than
B-full.

## 9. Compute comparison (spec §17)

| Metric | Composer | B-full |
|---|---|---|
| structural-encoder forward latency (A100, batch 128) | 0.00384 s | 0.00755 s |
| ratio | **0.51×** | 1.0× |
| peak GPU memory (profile stage) | 154 MB | — |
| mean level-1 objects/patch | 5.33 | — |
| mean level-2 objects/patch | 10.46 | — |
| peak GPU memory (formal training) | 442 MB | (same order) |
| training wall / epoch | 13.10 s | (same order) |

Explicit composition did **not** cause a combinatorial blow-up: the candidate
set is bounded (mean 10.5 level-2 objects/patch) and the composer forward is
≈ 2× *faster* than B-full's message-passing encoder on GPU. Compute is
acceptable; the failure is representational/optimization, not cost.

## 10. Relation-construction preview (zero training, spec §18)

Over 128 official-valid molecules, 453,670 internal-object pairs:

- overlap rate **0.823**, containment rate **0.230**,
  adjacency (min support graph distance ≤ 1) **0.969**,
  boundary-sharing (distance 1, no overlap) **0.146**.
- min graph-distance histogram `{0: 373171, 1: 66312, 2: 13697, 3: 489, 4: 1}`.
- relations are **deterministic** and all relations are constructible
  (`relations_deterministic = true`, `feasible_for_outer_relations = true`).

So learned internal objects *can* yield explicit `r_ij` (overlap / containment
/ adjacency / boundary-sharing / graph distance) at this scale. This is an
engineering-feasibility result only; the internal objects were **not** wired
into the `h/q` backbone.

## 11. Historical distinction (spec §19)

- **Not B-full.** B-full keeps a fixed patch support and encodes it with
  hidden GNN propagation into an opaque vector. Here the structural objects and
  their supports are explicit, and no hidden state is propagated.
- **Not KSVD motif-slot.** The old route maps a handcrafted/fixed patch vector
  to a KSVD dictionary id / sparse code and transports motif occurrences. Here
  raw atom/bond primitives feed end-to-end learned composition decisions with
  exact support unions — no motif vocabulary, no dictionary id, no certificate
  lookup.
- **Not DiffPool-style clustering.** There is no soft `N×K` assignment, no
  coarsened adjacency `Sᵀ A S`, and no partition requirement; objects may
  overlap and contain one another (recorded, not penalised).

## 12. What was NOT done (spec §20)

No official ZINC test; no MolHIV training; exactly one promotion seed; no GNN
inside the composer; no Transformer/attention; no soft clustering / DiffPool;
no certificate lookup / hash identity; no KSVD dictionary; no hard learned
top-k sweep; no latent-width / round-count / temperature sweep; no `h/q/T`
change; no optimizer sweep; no extra hand-crafted chemistry.

## 13. Verdict and next step

**STOP** (regression **+0.015162 > 0.003** vs B-full seed1, and the activity
gate saturated in practice). Per the pre-registered stop rule, do **not**
change rounds, latent width, activity temperature, candidate rule or pooling,
and do **not** buy a second seed. Do not lift these internal objects into outer
`h/q` relational objects.

**Single next step:** close the explicit-support-composition route in this form
and keep B-full as the frozen ZINC Cell A reference. If the exact-support idea
is revisited, it must be a *fresh pre-registered* design that addresses the
observed saturation directly (e.g. a composition objective/supervision that
prevents the activity gate from collapsing to its initialization), with its own
frozen witness — not a parameter sweep of this composer.

### Artefacts

- `results/explicit_support_composer/` — `parameter_accounting.json`,
  `sanity.json`, `diagnostics.json`, `compute_profile.json`,
  `relations_preview.json`, `decision.json`, `report.json`,
  `soup_exco_seed1.json`, `runs/exco_seed1.json`, curves/states/repro.
- `tracks/ksvd/tests/test_explicit_support_composer.py` (18 pass).
- `records/claims/claim-explicit-support-structural-composer-stop-20260915.yaml`
- `records/decisions/decision-explicit-support-structural-composer-stop-20260915.yaml`
