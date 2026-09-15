# Explicit Structural Basis with Learned Rank-1 Valuation — ZINC Cell A

**Date:** 2026-09-15
**Branch:** `exp/explicit-structural-basis-valuation`
**Base revision:** `93b3ebc` (integrated Agent A / Agent B / identity audit + B-full + B-bag + rank-1 audit + explicit-support composer STOP)
**Formal training commit:** `6f572f8`
**Run tag:** `esb-s1` (GPU1, detached), seed1 only
**Remote:** `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`
**Verdict:** **PARTIAL_GO (numeric band) / non-promotion** — seed1 Top-5 soup valid **0.120194** vs B-full seed1 **0.118126** (Δ = **+0.002068**). It is *statistically tied with the connectivity-free B-bag* (**Δ vs B-bag = −0.000035**), so the "clearly better than B-bag" condition of the pre-registered Partial-GO gate is **not** met. No second seed. Official ZINC test **never loaded**.

---

## 1. Question and representation hypothesis

The frozen B-full encoder derives a patch token by propagating hidden node
states over the real rooted typed radius-2 patch graph, and a zero-training
audit showed its 16-D `e_struct` is effectively **rank-1** (effective rank
1.11/1.20; a frozen rank-1 reconstruction moves the 2-seed soup by ≈ 1.6e-5).
The previous explicit-support *composer* failed because its learned activity
gate saturated (`notes/explicit_support_structural_composer.md`).

This study tests a narrower hypothesis, not a wider encoder:

> **The graph defines what structural objects exist; the network only learns
> what each explicitly present object is worth for the task.**

Nothing in the model can create, delete, gate or re-weight object existence,
and the structural channel is **rank-1 by construction**. The hypothesis is
that the task needs one learned *structural valuation axis*, not a high
dimensional latent patch manifold.

## 2. Pre-registered architecture (exactly one)

```
B0  atom objects              support = {v}
B1  bond objects              support = {u, v},        bonds = {(u, v)}
B2  centred 3-atom objects    support = {u, v, w},     centre = v,
                              bonds = {(v, u), (v, w)} (+ closure (u, w))
valuation  t_r = f_r(explicit object) in R   (one shared MLP per order, signed)
statistics A(P) = [mean(t_r), std(t_r), log1p(count_r)] for r in {0,1,2}  -> R^9
scalar     s(P) = fusion(A(P)) in R
channel    e_struct(P) = b + s(P) * v        with trainable b, v in R^16
```

- Object primitive fields are only graph-observable: atom type, root flag,
  distance from root, within-patch degree, boundary flag, bond type. No RDKit,
  no fingerprint, no target-derived chemistry, no certificate id, no motif
  vocabulary.
- `f0/f1/f2` are signed scalar MLPs (**no final sigmoid, no gate, no soft
  selection, no top-k**). Aggregation is the fixed invariant statistic set
  (**no attention pooling, no learned object weighting, no max selection**).
- B1 features are endpoint-swap symmetric (`z_u+z_v`, `|z_u−z_v|`, bond emb).
  B2 features are endpoint `u↔w` symmetric (centre, endpoint sum/diff, incident
  bond sum, closure flag + closure bond emb).
- `(u-v-w)` and `(v-u-w)` are distinct *centred* objects even with the same
  atom support: the centre is part of the object's semantic role.
- **No message passing**: `struct_src`/`struct_dst` are read only by the
  target-free enumeration `build_explicit_basis_graph`; the encoder's forward
  never touches them (`inspect` check + a `_NoEndpoints` wrapper test).
- Everything downstream is inherited bit-exactly from cell A / B-full:
  radius-2 patches, relation system, pair descriptors, `T=2` weight-tied
  recurrent pair–centre, parent embedding, global/topology channels, `R = 334`,
  the fixed small head, and the frozen optimized protocol.

## 3. Explicit object support (exact, auditable)

| Order | atom support | bond support |
|---|---|---|
| B0 | `{v}` | `{}` |
| B1 | `{u, v}` | the single real bond `(u, v)` |
| B2 | `{centre, u, w}` | `(centre,u)`, `(centre,w)` and the closure `(u,w)` when real |

Support functions (`atom_object_support`, `bond_object_support`,
`triple_object_support`, `triple_object_bond_support`, `triple_object_center`)
are exact and are replayed in tests and in the sanity stage. No latent
membership or soft assignment exists anywhere.

## 4. Why this is not a GNN, not clustering, not a lookup

- **Not message passing / GNN.** The structural module contains no
  `MessagePassing`/GCN/GIN/GraphConv/attention/Transformer; the forward source
  contains no `struct_src`/`struct_dst`; it succeeds with edge endpoints
  removed. Topology is used only to decide *which explicit objects exist*.
- **Not learned clustering / DiffPool / learned support.** There is no `N×K`
  assignment, no coarsened adjacency, no existence gate, no sigmoid, no top-k.
  Every object is a deterministic function of the raw patch graph.
- **Not a certificate lookup.** There is no `typed_embedding`; zeroing
  `typed_token` leaves the model output bit-identical; OOV structures are
  naturally encodable.
- **Not KSVD / motif dictionary / SBCI / v6 attribute branch.** No dictionary,
  no object id, no soft assignment, no side attribute branch.

## 5. Parameter accounting (exact, budget-matched)

| Block | Params |
|---|---|
| primitive embeddings (atom/root/dist/degree/boundary/bond) | 924 |
| atom valuation `f0` | 4,049 |
| bond valuation `f1` | 10,673 |
| 3-atom valuation `f2` | 17,481 |
| scalar fusion | 2,025 |
| rank-1 projection (`b`, `v`) | 32 |
| **encoder total** | **35,184** |
| **candidate total** | **84,527** |
| cell-A baseline | 85,763 (typed lookup 36,420) |
| B-full total / encoder | 84,495 / 35,152 |
| Δ vs B-full | **+32 (+0.038 %)** |

Widths (`node_dim 20`, `edge_dim 16`, `degree bins 8`, valuation/fusion hidden
184) were chosen **once by parameter accounting** to match the B-full encoder
budget; they were **not** selected on validation. No width/round sweep was run.

## 6. Integrity / adversarial tests (all pass)

`sanity` on the first 128 official-valid molecules: **34/34 pass**
(`results/explicit_structural_basis/sanity.json`), covering every numbered
requirement of the brief: node-relabel invariance; exact B0/B1/B2 atom and bond
supports; explicit centre; B1 count = real bond count; B2 count =
`Σ_v C(deg(v),2)`; B1 endpoint-swap invariance; B2 `u↔w` swap invariance;
atom/bond type changes move the matching valuation; no message-passing modules;
no attention; no vocab lookup; no certificate dependence; no learned
support/existence gate; rank ≤ 1 by construction; width 16; `h/q/T = 64/16/2`;
forward/backward finite; parameter budget.

**Adversarial connectivity witness (7 atoms).** Two graphs on one atom type
and one bond type with the **same** degree multiset `{4×6, 2}`, the **same**
rooted distance multiset `{0,1,1,1,1,2,2}` and the **same B1 symmetric
endpoint-primitive multiset**, but different edge sets. The explicit centred
**B2 basis differs** and the pooled patch output moves
(`output_delta = 9.3e-6`). This is the conceptual integrity witness:
*connectivity changes which explicit structural objects exist* — it is not
hidden propagation.

**Honest boundary.** A second 6-atom-tree witness (`descriptor_coincident_pair`)
has a *different identified B2 basis* but a *coinciding B2 descriptor multiset*,
so its pooled output is representation-equal (delta 0.0). Mean/std statistics
over a population can coincide under relabelling; this is a property of the
fixed aggregation, recorded rather than hidden.

`tracks/ksvd/tests/test_explicit_structural_basis.py`: **21 pass**. Affected
existing suites (B-full, B-bag, composer, rank-1 audit): **67 pass** total.

## 7. Formal one-seed result (seed1)

Deterministic-A100 (`torch.use_deterministic_algorithms(True)`), Adam
`lr=1e-3`, `wd=1e-5`, batch 128, `max_epochs=240`, patience 40, no scheduler,
L1, clip 5, fixed equal-weight Top-5 soup. Seed1 on GPU1.

- seed1 raw (best checkpoint) **0.127628** @ epoch 177; 217 epochs; wall
  **2920 s** (13.46 s/epoch); peak GPU **389 MB**.
- seed1 fixed Top-5 soup **0.120194** (soup gain −0.007434; top-5 epochs
  from the run summary).
- Determinism: 6-epoch `repro` run twice on GPU1 → bit-identical
  (`selection_state_sha256 = fe142098fa16a263…`, best valid 0.336504).

### Matched single-seed comparison (seed1 soup)

| Variant | seed1 soup | Δ vs B-full | params |
|---|---|---|---|
| **B-full** (connectivity-aware GNN encoder) | **0.118126** | — | 84,495 |
| **Explicit structural basis** | **0.120194** | **+0.002068** | 84,527 |
| B-bag (connectivity-free ablation) | 0.120229 | +0.002103 | 84,511 |
| A2 (no identity, shared adapter) | 0.122134 | +0.004008 | 85,740 |

Δ vs A2 **−0.001940**; Δ vs B-bag **−0.000035**. The explicit basis lands
essentially exactly on the **connectivity-free bag**, i.e. it recovers almost
none of B-full's connectivity advantage over B-bag (+0.002103) — while being
0.00194 better than the identity-free A2 adapter.

## 8. Mechanism diagnostics (frozen, 256 valid molecules for distributions)

### Object valuations (trained seed1)

| Order | mean | std | range | note |
|---|---|---|---|---|
| `t_atom` (B0) | 0.001318 | **0.0** | constant | atom valuation collapsed to a constant |
| `t_bond` (B1) | −0.5032 | **3.590** | [−11.90, 5.17] | strongly active, wide signed range |
| `t_triple` (B2) | 0.3841 | **1.090** | [−2.47, 2.71] | active |

Initial (untrained, same seed): `t_atom` std 0.382, `t_bond` std 0.301,
`t_triple` std 0.281, `s` std 0.062.

### Patch scalar `s(P)`

- **trained** `s` mean −0.1239, **std 1.0338**, range [−3.259, 3.657] →
  **not collapsed** (the pre-registered collapse criterion `std(s) ≈ 0` fails).
- initial `s` std 0.062, range [−0.625, −0.253].
- `e_struct` effective rank = **1.000** by construction (norm mean 0.354).

The B0 valuation becoming constant is a *learned allocation*, not the
pre-registered collapse: `s(P)` retains large variance, and B0 still
contributes through the invariant `log1p(count0)` term. It says the model finds
no per-atom value axis, only bond/triple value axes.

### Frozen order ablation (official valid, no retraining)

Zero one order's three aggregated statistics in the fusion input:

| ablation | soup valid | Δ soup vs baseline | raw valid | Δ raw |
|---|---|---|---|---|
| none | 0.120194 | — | 0.127628 | — |
| remove B0 | 0.122451 | **+0.002257** | 0.130276 | +0.002648 |
| remove B1 | 0.139060 | **+0.018866** | 0.150689 | +0.023061 |
| remove B2 | 0.121279 | **+0.001084** | 0.129077 | +0.001449 |

**B1 (explicit bond objects) carries almost all of the learned structural
value**; B0 contributes through count; B2 (centred 3-atom objects) adds only
+0.0011. This is consistent with the local nature of the basis.

### Candidate scalar `s(P)` vs frozen B-full PC1 (matched 23,083 valid patches)

| statistic | value |
|---|---|
| Spearman ρ | **0.0962** |
| Pearson r | **0.0875** |
| linear R² | **0.0077** |

The explicit-basis scalar does **not** recover B-full's functionally effective
scalar axis. The two representations use different learned axes; the explicit
basis lands where B-bag lands, not where B-full lands.

### Existing deterministic descriptors → `s(P)` (fixed ridge α = 1, train→valid)

| statistic | value |
|---|---|
| valid R² | **0.2755** |
| features | 224 pre-existing target-free patch descriptors |
| train / valid n | 231,664 / 23,083 |

So `s(P)` is only partially a reparameterization of the existing hand-crafted
statistics (~27.5 % of its variance is linearly predictable); the rest is a new
combination induced by the explicit object basis.

## 9. Compute diagnostics

| Metric | Explicit basis | B-full |
|---|---|---|
| structural-encoder forward latency (A100, batch 128) | 0.01397 s | 0.00552 s |
| latency ratio | **2.53×** | 1.0 |
| peak GPU memory (profile) | 126 MB | — |
| peak GPU memory (formal training) | 389 MB | same order |
| training wall / epoch | 13.46 s | same order |
| mean objects/patch | atoms 6.12, bonds 5.33, triples 5.94 (**17.39** total) | — |

Complexity is `O(V_patch + E_patch + Σ_v deg(v)²)`: total triples 1,374,482
over 231,664 train patches (5.93/patch; max 309 per molecule). **No
combinatorial blow-up**, but the object-materialising encoder is ~2.5× the
B-full GNN encoder latency on GPU (it runs three MLPs over ~17 objects/patch).
Compute is acceptable but not free.

## 10. Conceptual comparison (spec §25)

- **Historical identity model:** `patch → certificate id → embedding lookup`
  (opaque categorical memory, vocab-sized, OOV-blind).
- **B-full:** `patch graph → hidden GNN propagation → opaque ~rank-1 vector`
  (35,152-param encoder, connectivity-aware).
- **Failed learned composer:** `candidate supports → learned existence/activity
  → near-constant gates → collapse` (STOP).
- **This model:** `explicit graph-derived objects → every object exists
  deterministically → learned scalar valuation → fixed invariant aggregation →
  rank-1 patch channel`.

> **We do not learn what the structure is. We learn the value of structures
> that are explicitly present.**

## 11. Historical exclusions (spec §26)

This is **not**: the v6 topology-attribute side branch; KSVD motif dictionary;
motif-slot transport; SBCI; DiffPool / learned clustering; a subgraph GNN; a
certificate embedding. Explicitly: no dictionary, no object id, no soft
assignment, no hidden message propagation, no learned support.

## 12. Official-test discipline (spec §27)

**ZINC official test was never loaded** (`official_test_loaded=false` in every
artifact; `research doctor` data check only inspected train/val sizes). No
second seed was purchased. No MolHIV training.

## 13. Verdict

Pre-registered gates:

| gate | threshold | candidate |
|---|---|---|
| Strong GO | ≤ 0.116126 | 0.120194 ✗ |
| Method GO | ∈ [0.117126, 0.119126] | 0.120194 ✗ |
| Partial GO | (B-full+0.001, B-full+0.003] **and clearly better than B-bag** | in band, but Δ vs B-bag = −0.000035 (a tie) → condition not met |
| STOP | > 0.121126 or `s(P)` collapse | neither ✗ |

**Verdict: PARTIAL_GO on the numeric band only; not promoted.**

The candidate is +0.002068 behind B-full (≈ one training-noise unit) and
statistically indistinguishable from the connectivity-free B-bag. The
mechanism diagnostics are decisive and mutually consistent:

1. the explicit basis **reproduces B-bag**, not B-full (Δ vs B-bag = −3.5e-5);
2. its scalar does **not** align with B-full's PC1 axis (ρ = 0.096);
3. B1 dominates the ablation while the information it carries is *local*
   (bond endpoint primitives), exactly the information the bag also has;
4. no collapse, exact supports, sane compute — the architecture behaved as
   designed, so this is a **representational** limit (no propagation beyond
   fixed low-order supports), not an optimisation/collapse failure.

A second seed is **not** purchased. Even if the +0.0021 were confirmed on a
second seed, the candidate would remain tied with the connectivity-free bag,
which is not a useful promotion for a B-full replacement.

## 14. Single next step (spec §29 #18)

**Do not buy a second seed.** Close this *exact* basis as a B-full replacement
and keep B-full as the frozen ZINC Cell A reference. If the "explicit basis +
learned valuation" idea is revisited, it must be a fresh pre-registered design
that adds a mechanism able to carry information *beyond* fixed low-order local
supports (so that it can leave B-bag performance), with its own frozen witness —
not a parameter sweep of this basis.

### Artefacts

- `tracks/ksvd/results/explicit_structural_basis/` — `parameter_accounting.json`,
  `sanity.json`, `diagnostics.json`, `mechanism.json`, `compute_profile.json`,
  `decision.json`, `report.json`, `soup_esb_seed1.json`, `runs/esb_seed1.json`,
  `repro_detA_seed1.json`, `repro_detB_seed1.json`, curves/states/soup_states.
- `tracks/ksvd/tests/test_explicit_structural_basis.py` (21 pass).
- `records/claims/claim-explicit-structural-basis-valuation-20260915.yaml`
- `records/decisions/decision-explicit-structural-basis-valuation-partial-nopromotion-20260915.yaml`
