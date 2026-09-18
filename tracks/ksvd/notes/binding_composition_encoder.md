# Binding Composition Encoder (BCE) on ZINC patches

**Date:** 2026-09-16
**Branch:** `exp/binding-composition-encoder-zinc`
**Commit (formal run):** `c686c5d6030cfb68e74a1f6c1432b8f33e266641`
**Run tag:** `bce-seed0` (remote GPU 1)
**Remote:** `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`
**Protocol version:** `binding_composition_encoder_v1`
**Verdict:** **STOP (Case D — composition branch dead / constant; plus pre-registered seed0 guard exceeded).**
The single authorised seed (seed0) reaches Top-5 soup valid **0.1332753**, above
the pre-registered guard `B-Bag + 0.003 = 0.1268055` by **+0.0064698**. The
mechanism also collapsed: the shared `Compose` operator became an exactly
constant function of the decomposition interface (`composition_disagreement`
6.7e-9 at the selection checkpoint vs 0.1859 at init), 53.7% of encoder
parameters (including **all** atom/bond/distance/root embeddings and 99.7% of
`Compose`) were annihilated to ~0 by Adam weight decay, and the 16-D
`e_struct` is near rank-1 (effective rank 1.053). All live computation is
carried by `SupportUpdate` and `Fusion`. **No seed1, no promotion, no sweep.**
Official ZINC test **never loaded**.

---

## 1. Question

Can minimal explicit **structure–attribute bindings** (`Object = (exact support,
learned state, provenance)`), combined by **one shared composition operator**
over every legal explicit parent decomposition, recursively form higher-order
explicit objects (size-3 and size-4 connected supports) inside each rooted
radius-2 patch, and does that explicit composition use structure better than
the frozen aggregate B-Bag / B-Full references on ZINC Cell A?

This is a **composition study, not attention.** The hypothesis is that making
the decomposition of a support explicit (support = unordered node set; induced
edges = the real patch induced bonds; provenance = the recorded decomposition
parents) gives the model a structural signal that scalar patch pooling cannot.

Pre-registered gates: one architecture only; `MAX_SUPPORT_SIZE = 4` fixed;
radius-2 patches unchanged; `Compose` shared across all patches/supports/
decompositions and symmetric in the two parents; aggregation over the
decomposition multiset = **mean + std + count** (no attention / no soft-gate /
no top-k); `object_dim=32, compose_hidden=48, activation=SiLU/GELU,
output_dim=16`, candidate total ≤ 120,000 params; **dataset-dependent params
must be 0**; standard-scale init (no tiny-init residual); seed0 must satisfy
`seed0 Top-5 soup valid ≤ B-Bag + 0.003 = 0.1268055`, otherwise **stop without
re-tuning**; official test locked.

## 2. Pre-registered architecture (exactly one)

`patch_representation="binding_composition"`, implemented by
`BindingCompositionEncoder` in
`tracks/ksvd/experiments/luyin16/structural_patch_encoder.py`, wired through
`zinc_patch_path_pooling.py` (`bce_*` kwargs). The decomposition chart is built
by `tracks/ksvd/experiments/luyin16/binding_composition_support.py`; the runner
is `tracks/ksvd/experiments/luyin16/zinc_binding_composition_encoder.py`.

Locked widths: `object_dim=32`, `compose_hidden=48`, `update_hidden=48`,
`fusion_hidden=48`, `activation=silu`, `output_dim=16`.

**Object contract.** An object is `(exact support, learned state, provenance)`.
The support is an unordered node set; its induced edge set is exactly the
original patch induced bonds (`src < dst`, undirected), not a learned mask.

- **Primitives.** Size-1 object = atom primitive:
  `h = atom_mlp([atom_emb(a), root_emb, dist_emb])`. Size-2 connected support =
  bond-binding primitive: `h = bond_binding_mlp([bond_emb(b),
  z_u + z_v, |z_u - z_v|, ...])`. A size-2 support is **not** re-composed from
  its two atoms; it is a primitive binding.
- **Support chart.** For every rooted radius-2 patch, enumerate every connected
  induced node subset of size ≤ 4 (all subsets, not sampled). Exact supports
  are keyed by `frozenset` so relabelling cannot change them.
- **Decomposition rule.** For each support `S` of size 3 or 4, enumerate every
  **unordered** pair `(A, B)` with `A ∪ B = S`, `A ≠ ∅`, `B ≠ ∅`, `|A| < |S|`,
  `|B| < |S|`, `A` and `B` connected induced supports, and with a **non-empty
  intersection** `A ∩ B` (overlap) **or** at least one real crossing bond
  between `A \ B` and `B \ A`. Every legal decomposition is used; there is no
  "first decomposition wins" and no lexicographic parent selection. Symmetric
  difference is legal; complement-only pairs are not (they would carry no
  shared information).
- **Compose (shared).** One `Compose` MLP consumes a **symmetric parent-pair
  interface** built from the two parent states and the decomposition geometry:
  `hA + hB`, `|hA - hB|`, parent supports' size features (`|A|+|B|`,
  `||A|-|B||`, `rootA+rootB`, `|rootA-rootB|`), overlap size, crossing-bond
  count and the mean/std of the crossing bond types, and the induced-bond-type
  moments. Parent storage order carries no semantics (proved by
  `node_relabel_invariant`, `decomposition-order invariance`, and the
  order-swap test).
- **Decomposition aggregation.** For each support, the multiset of candidate
  child states is reduced by `mean + std` and a `count` feature; no softmax /
  attention / top-k / learned weighting over decompositions.
- **Support update (shared).** `SupportUpdate` maps
  `[previous support state, candidate mean, candidate std, size embedding]` to
  the new support state with one shared MLP for all supports and all sizes.
- **Patch fusion (shared).** `Fusion` pools the final object states of the
  patch (size-conditioned mean/std plus atom/bond/attribute moments) to the
  16-D `e_struct` consumed by the frozen downstream pair system, T=2
  pair–centre recurrence, and graph head.

## 3. Invariants and forbidden-architecture compliance

Enforced by code and checked in `tracks/ksvd/tests/test_binding_composition_encoder.py`
(13 tests) and the `sanity` stage (13 checks, all pass):

- **Exact induced support** — induced bond ids equal the real patch bonds among
  the support nodes (test D/E; brute force reference agrees on every fixture).
- **Complete decomposition coverage** — chart decompositions equal an
  independent `brute_force_decompositions` reference on all fixtures (test E).
- **Node relabel invariance** (exact 0.0), **edge-order invariance**
  (≤ 1.12e-8, only on the deliberately ambiguous fixture), **decomposition-order
  invariance**.
- **Provenance recursion** — every size-4 support decomposes recursively to
  size-1/2 primitives (test F).
- **Binding sensitivity / connectivity sensitivity** — changing bond type or
  connectivity changes the output (tests G/H; sanity).
- **No attention / gate / top-k** — AST audit of the encoder source finds no
  `softmax`/`attention`/`sigmoid`/`gumbel`/`topk`/`MultiheadAttention`
  identifiers and no `attn`/`gate` submodules (test K, sanity
  `no_gat_or_attention`).
- **No vocabulary parameters** — `dataset_dependent_params = 0`,
  `typed_token_embedding` absent (`no_vocabulary_parameters`).
- **Gradient viability at init** — `atom_mlp`, `bind_mlp`, `compose`,
  `support_update`, `fusion` all have non-zero gradients.
- **Batch invariance** — a molecule encoded alone equals the same molecule
  encoded inside a concatenated batch (max diff 0.0).

## 4. Parameters

| Component | Params |
|---|---:|
| `atom_mlp` | 2,112 |
| atom/root/distance/bond embeddings | 1,184 |
| `bond_binding_mlp` | 6,224 |
| `Compose` | 11,216 |
| `support_size_embedding` | 160 |
| `SupportUpdate` | 6,320 |
| `Fusion` | 5,584 |
| **BCE encoder total** | **32,800** |
| inherited frozen downstream | 49,343 |
| **candidate total** | **82,143** |
| reference baseline A0 total | 85,763 |
| dataset-dependent params | **0** |

Budget respected (82,143 ≤ 120,000); the candidate is 3,620 params **smaller**
than the A0 baseline and uses no typed/vocabulary table. No width or
`MAX_SUPPORT_SIZE` sweep was run.

## 5. Tests and sanity (local, CPU)

- `tracks/ksvd/tests/test_binding_composition_encoder.py`: **13/13 pass**
  (relabel, edge order, decomposition order, exact induced support, brute-force
  decomposition coverage, provenance, binding sensitivity, connectivity
  sensitivity, no-attention audit, gradient viability, batch invariance,
  vocabulary/budget).
- `sanity` stage: **13/13 checks pass**; max invariance diff 1.12e-8 (all other
  invariants exactly 0.0); init gradient norms: compose 7.88e-4, support_update
  7.02e-3, fusion 7.21e-2, atom_mlp 1.75e-2, bind_mlp 5.97e-3.
- Related existing suites re-run unchanged: 105 tests pass
  (`test_shared_structural_patch_encoder`, `test_shared_bag_patch_encoder`,
  `test_adaptive_structure_binding_cell`, `test_explicit_object_relational`,
  `test_explicit_structural_basis`, `test_explicit_support_composer`) plus 52
  patch-path tests.

## 6. GPU smoke and memory safety

Remote `smoke --device cuda --steps 20` (64-molecule batch, commit `c686c5d`):

- Loss **decreased** 1.2465 → 0.7425 over 20 steps; all losses finite.
- Gradients non-zero for `compose`, `support_update`, `bind_mlp`, `fusion`;
  object states non-trivial; composition disagreement **0.1859**; `e_struct`
  effective rank **2.80** at init (not constant).
- **Peak GPU memory 329,655,296 B ≈ 314 MiB.**
- `candidate_audit` forward/backward probe (32 molecules, 44,564
  decompositions): 0.882 s, **peak 169,954,304 B ≈ 162 MiB**, `probe_ok`.
- Formal seed0 training **peak 697,091,072 B ≈ 665 MiB** (recorded in
  `runs/bce_seed0.json`).

All peaks are ~1.7% of the 40 GB A100, so no OOM risk and no protocol change
was required. **Note:** the first launch was placed on GPU 0, which was already
87% occupied (35.7 GB, 96–100% util) by another job; it was aborted within
seconds and relaunched on GPU 1. This is recorded for reproducibility.

## 7. Scale statistics and combinatorial audit

Full chart cache rebuilt on remote in 141.8 s (cache 102 MB train + 10 MB valid
gzipped; `candidate_audit` wall 592.3 s):

| | train | valid |
|---|---:|---:|
| molecules | 10,000 | 1,000 |
| patches | 231,664 | 23,083 |
| supports | 5,411,595 | 539,320 |
| decompositions | 13,511,399 | 1,346,736 |
| supports / patch (mean, p99, max) | 23.4, 50, 107 | 23.4, 50, 99 |
| decompositions / patch (mean, p99, max) | 58.3, 1257, 3015 | 58.3, 1254, 2499 |

Supports by size (train): size1 1,418,500 / size2 1,232,844 / size3 1,368,984 /
size4 1,391,267. A 128-molecule batch carries ~170–185k decompositions; the
heavy per-patch tail is bounded (max 3,015) and does **not** cause a
combinatorial explosion. This is an explicit audit result: `MAX_SUPPORT_SIZE=4`
is affordable.

## 8. Formal protocol and runs

Protocol inherited verbatim from the frozen optimized Cell A regime
(`zinc_compact_v4_smallhead_e2e.OPTIMIZED_PROTOCOL`): Adam, lr 1e-3, weight
decay 1e-5, batch 128, L1 loss, gradient-clip 5.0, max 240 epochs, patience 40,
no scheduler, best-official-valid checkpoint selection, fixed Top-5 weight
soup. No optimizer, schedule, width, or `MAX_SUPPORT_SIZE` change was made.

Seeds: **only seed 0 was run**, because the study is guard-gated on seed0 and
the guard failed. A second seed would have been purchased only on a guard pass.

## 9. Results and paired comparisons (seed 0)

| model | params | valid (2-seed soup / seed0) |
|---|---:|---:|
| B-Bag (frozen) | 84,511 | 0.1238055 (seed0 0.1273818) |
| B-Full (frozen) | 84,495 | 0.1189722 (seed0 0.1198180) |
| A2 (frozen) | 85,740 | 0.1219140 (seed0 0.1216936) |
| **BCE seed0 best checkpoint** | 82,143 | **0.1343159** (epoch 209) |
| **BCE seed0 Top-5 soup** | 82,143 | **0.1332753** |

- Pre-registered guard: `0.1268055`. **FAIL** by **+0.0064698**.
- vs B-Bag 2-seed soup: **+0.0094698** (worse).
- vs B-Bag seed0 0.1273818: **+0.0058935** (worse); even against B-Bag
  seed0 + 0.003 = 0.1303818 the candidate still fails by +0.0028935.
- vs B-Full seed0 0.1198180: **+0.0134573** (worse).
- vs A2 seed0 0.1216936: **+0.0115817** (worse).
- Top-5 soup improves over the best checkpoint by only 0.0010406
  (top-5 epochs 209/238/217/236/226), i.e. the selection curve is flat, not a
  lucky-dip rescue.

No 2-seed paired comparison is possible by design (seed1 not purchased). The
candidate clearly fails the pre-registered seed0 gate.

## 10. Mechanism diagnostics (decisive)

At the seed0 selection checkpoint (epoch 209):

- `composition_disagreement_mean = 6.69e-9`. At init it was **0.1859** — a
  ~7-order-of-magnitude collapse. Per-size on the 128-molecule diagnostic set:
  size3 mean 6.83e-9 (n=1336, max 1.91e-8), size4 mean 6.58e-9 (n=883, max
  3.41e-8). `candidate_across_decomposition_std_mean = 0.0`, i.e. **all legal
  decompositions of a support now produce the identical candidate.** The
  decomposition multiset aggregation is a no-op relative to a single
  decomposition.
- `e_struct` symmetric-`Compose` **effective rank 1.053** (participation ratio
  1.018, top singular fraction 0.991) — the 16-D structural output is near
  rank-1, the same failure shape as the Z1 / explicit-object-relational runs.
- `candidate_norm_mean = 0.03275`; size3 object norm mean 0.340 / std 1.115,
  size4 mean 0.229 / std 0.880; `output_norm_std = 0.244`. The final object
  states vary across supports, but that variation is produced by
  `SupportUpdate` + `Fusion` on top of a constant composition candidate.

**Weight audit** (seed0 selection state; encoder 32,800 params, 17,616 = 53.7%
have `|w| < 1e-30`):

| module | zero params | surviving norm |
|---|---:|---:|
| atom_embedding | 100% | 0 |
| bond_embedding | 100% | 0 |
| distance_embedding | 100% | 0 |
| root_embedding | 100% | 0 |
| atom_mlp | 98% | bias 0.00094 |
| **Compose** | **99.7%** | only last bias 0.03275 |
| bind_mlp | 49% | 0.0084 |
| size_embedding | 60% | 0.0265 |
| SupportUpdate | 0% | 1.26 |
| Fusion | 0% | 1.25 |

`compose.2.weight = 0`, so `Compose` outputs exactly its constant last bias
(norm 0.03275 = `candidate_norm_mean`). The entire primitive/binding/composition
pathway was annihilated (Adam with coupled weight decay drives a
vanishing-gradient parameter by ~lr per step); only `SupportUpdate` and
`Fusion` remain alive. This reproduces the Z1 mechanism exactly.

Trajectory (per-epoch curve): disagreement is 0.0198 at epoch 1 and
`candidate_across_decomposition_std` is non-zero only in 5.8% of the 240
epochs, reaching exactly 0 and staying there; compose grad norms fall to
1e-8–1e-7. `valid_mae` bottoms at 0.134316 (epoch 209) while `train_mae`
continues to 0.085 — the model overfits using the `SupportUpdate`/`Fusion`
residual pathway and the composition mechanism is unused.

## 11. Interpretation

The hypothesis was **not cleanly tested**: the shared `Compose` operator
degenerated to a constant before it could be used, so BCE is effectively a
size-conditioned support-state encoder with a dead composition branch. This is
**not** evidence that explicit decomposition composition is worthless; it is
evidence that this pre-registered implementation does not realize it. The
performance gap (≈ +0.0095 vs B-Bag soup) is consistent with the frozen
findings that patch-local connectivity beyond aggregate primitives is not the
task-relevant ZINC Cell A signal (B-Full near rank-1; EOR inner-encoder
collapse; ESB connectivity-free tie). Two compounding causes are visible and
match Z1: (i) the composition interface features did not carry enough
task-gradient to survive, and (ii) standard-init parameters on the
primitive/compose path have vanishing gradients once the residual
`SupportUpdate`/`Fusion` path fits the data, so Adam weight decay erases them.

## 12. Deviations from the brief

The brief locked `object_dim=32`, `compose_hidden=48`, activation
SiLU/GELU, `output_dim=16`, `MAX_SUPPORT_SIZE=4`, radius-2, mean+std+count
aggregation, dataset-dependent params 0, and the seed0 guard. All were
honoured. **Recorded deviation:** `update_hidden` and `fusion_hidden` were not
explicitly specified in the brief; both were set to **48** (chosen once, not
swept). Also recorded: the first remote launch briefly shared GPU 0 with an
unrelated job before being aborted and relaunched on GPU 1 (no data effect).
No architecture, optimizer, or protocol change was made after the guard result.

## 13. Verdict

**STOP — Case D (composition branch dead / constant), with the pre-registered
seed0 guard exceeded.** Do **not** promote; do **not** buy seed1; do **not**
sweep widths, `MAX_SUPPORT_SIZE`, aggregation, activation, or optimizer; do
**not** open the official test. A single pre-registerable repair direction is
recorded as a *candidate only* (see §16); it requires its own fresh witness and
is not authorized here.

## 14. Report questions (Q1–Q27)

1. **Hypothesis?** Minimal explicit structure–attribute bindings composed by one
   shared operator over explicit decompositions form higher-order explicit
   objects that help ZINC prediction.
2. **Object contract?** `(exact support, learned state, provenance)`; support is
   an unordered node set; induced edges are the real patch bonds.
3. **Decomposition semantics?** All unordered connected parent pairs with
   non-empty overlap or a crossing bond; both parents strictly smaller; every
   legal decomposition used.
4. **"First decomposition wins" / lexicographic parents?** Absent; all
   decompositions are enumerated and aggregated.
5. **Attention / soft gate / top-k / GAT?** Absent; mean+std+count aggregation.
6. **Compose shared and symmetric?** Yes — one MLP, symmetric interface
   features, storage order irrelevant; relabel/order invariants are exact.
7. **Radius / max support changed?** No: radius-2, `MAX_SUPPORT_SIZE=4`, no
   sweeps.
8. **Size-1/2 handling?** Size-1 atoms and size-2 connected bond bindings are
   primitives, not recomposed.
9. **Vocabulary / dataset-dependent params?** Zero; no `typed_embedding`.
10. **Widths / budget?** `object_dim=32, compose_hidden=48, update_hidden=48,
    fusion_hidden=48, output_dim=16`; encoder 32,800; candidate 82,143 ≤
    120,000.
11. **Init?** Standard-scale (no tiny-init residual, no 1e-3 blow-up); init
    gradients non-zero for every module.
12. **Tests?** 13/13 correctness tests; 105 + 52 related existing tests pass.
13. **Sanity?** 13/13 checks; batch/node/edge invariants; no-GAT audit passes.
14. **GPU smoke?** Loss 1.2465→0.7425, gradients non-zero, disagreement 0.1859,
    rank 2.80, peak 314 MiB.
15. **Peak GPU memory (formal)?** 665 MiB; no OOM; no batch-size or protocol
    change.
16. **Combinatorial scale?** 5.41M supports / 13.51M decompositions on train;
    max 3,015 decompositions per patch; bounded, no explosion.
17. **Protocol?** Inherited frozen Cell A protocol, unmodified.
18. **Seeds?** Seed 0 only (guard-gated); seed 1 not purchased.
19. **Commit / provenance?** `c686c5d6030cfb68e74a1f6c1432b8f33e266641` on
    `exp/binding-composition-encoder-zinc`; remote commit verified.
20. **Seed0 result?** Best checkpoint 0.1343159 (epoch 209), Top-5 soup
    0.1332753.
21. **Guard?** `0.1268055`; **FAIL** by +0.0064698.
22. **Paired comparisons?** vs B-Bag +0.0094698 / seed0 +0.0058935; vs B-Full
    seed0 +0.0134573; vs A2 seed0 +0.0115817 (all worse).
23. **Mechanism alive?** No — composition disagreement 6.7e-9 (init 0.1859),
    candidate cross-decomposition std 0.0.
24. **Collapse evidence?** `e_struct` effective rank 1.053; 53.7% of encoder
    weights zeroed including 100% of embeddings and 99.7% of `Compose`.
25. **What is doing the work?** `SupportUpdate` + `Fusion` (0% dead), acting on
    constant primitive/compose inputs.
26. **GO / PARTIAL / STOP?** **STOP** (Case D + guard exceeded); no promotion.
27. **Durable record?** This note, the linked claim and decision records, and
    the STATE update; next direction is paradigm-level, not another local
    structural patch.

## 15. Provenance and reproduction

- Branch/commit: `exp/binding-composition-encoder-zinc` @ `c686c5d`.
- Remote: `res`, `/home/hxy/cy/research`, run tag `bce-seed0` on GPU 1.
- Local results: `tracks/ksvd/results/binding_composition_encoder/`
  (`parameter_accounting.json`, `candidate_audit.json`, `sanity.json`,
  `smoke.json`, `diagnostics.json`, `disagreement.json`, `support_examples.json`,
  `decision.json`, `report.json`, `runs/bce_seed0.json`, `soup_bce_seed0.json`,
  `curves/bce_seed0_curve.csv`, `states/`, `soup_states/`).
- Commands: `python -m tracks.ksvd.experiments.luyin16.zinc_binding_composition_encoder
  {params,preprocess,candidate_audit,sanity,smoke,diagnostics,disagreement,
  support_examples,decide,report}` locally;
  `train_queue --seeds 0 --device cuda` on the remote.
- Official ZINC test was **never** loaded at any stage.

## 16. Candidate repair direction (NOT authorized)

A single pre-registerable repair is recorded but not started: give the
composition branch a non-vanishing task gradient (e.g. a loss/architecture that
forces the decomposition candidate to be used before the `SupportUpdate`
residual can fit; or excluding the primitive/compose parameters from Adam
weight decay), then re-run the seed0 guard with a fresh frozen witness. This
changes forward semantics / regularization and therefore must not be treated as
a continuation of this commit.
