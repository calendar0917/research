# Compact-v3: Context-Conditioned Patch Representation (ring/cycle context)

Single-variable experiment on the verified compact-v2 baseline.
Question: does making the *same local pattern mean something different under
different structural contexts* move the ZINC plateau — with the model kept at
~100k parameters and no ring message passing?

- protocol: `zinc-context-gap` (seed 0; official PyG ZINC subset split)
- baseline: Compact-Hybrid-v2 (`configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml`),
  valid 0.184158, test 0.1353615, params 98,549 selection / 99,613 refit

## Hypothesis

Compact-v2 forms a *context-free* patch representation: `Encode(patch)`.  The
claim under test: the semantics of a local molecular pattern depend on its
higher-order structural context, i.e. the same radius-2 patch should not have
the same representation when it sits in a ring vs. a chain (or in different
rings).  The experiment is therefore `Encode(patch | structural context)`,
implemented as a **small conditioning module that modifies the exact patch
embedding output**; the rest of the v2 pipeline (descriptor branch, pair
encoding, pooling, center-context update, head, optimizer, horizon, split,
selection rule) is untouched.

## Difference from CIN

There is **no** CIN machinery anywhere in this experiment:

- no ring/cell nodes, no atom-ring incidence graph, no ring→patch→ring
  message flow, no cellular message passing, no higher-order GNN rewrite,
  no attention;
- the ring/cycle information is only a *conditioning signal*: after the
  extractor produces per-node context keys, it becomes a small embedding /
  4D vector that modulates the patch representation;
- no new computation node is created in the model graph.

The design is intentionally the other way around from CIN:
CIN = higher-order *object* → message passing; here = higher-order *context*
→ modifies the local representation.

## What changed vs v2 (exact list)

1. `tracks/ksvd/experiments/luyin16/structural_context.py` (new): cycle
   enumeration (bounded, configurable `max_cycle_len=10`) + per-node
   coarse + typed ring context.
   - coarse: `[in_cycle, cycle_count, min_cycle_len, max_cycle_len]` (4D).
   - typed: center-anchored canonical typed cycle signature
     (`node_type, edge_type, ...` around the cycle, direction-canonical,
     lexicographically minimal of forward/reverse), one token per distinct
     ring context of the centre node; multiple cycles → sorted, combined key.
2. `PatchPathModel` (in `zinc_patch_path_pooling.py`): optional
   `structural_context_mode` (`none`/`coarse_ring`/`typed_ring`),
   `structural_context_fusion` (`condition`/`concat`),
   `structural_context_dim=4`, `structural_context_embedding_rank=1`,
   `structural_context_condition_rank=8`.
   - conditioning: `e_patch' = e_patch + W_out(tanh(W_p e_patch) *
     tanh(W_c e_ctx))`, all projections bias-free, `W_out` zero-initialised
     (model ≡ compact-v2 at init), delta masked to **zero** for NO_RING.
   - concat (ablation only): `e_ctx` appended to the patch-encoder input.
3. Feature pipeline: context extracted once per graph during feature build
   (0.4 ms/graph measured; no per-epoch cost); context vocabulary fit on
   official **train only** for the selection phase, refit on train+valid for
   the terminal phase (same protocol rule as the patch vocabularies);
   token 0 = NO_RING, 1 = UNK_CONTEXT, known keys 2+.
4. New configs under `configs/luyin16/zinc_compact_v3_*.yaml` (4 variants);
   no existing config/code was modified (v2 config unchanged, historical
   tests still pass 208→220).

## Data leakage audit

- context key = pure function of the molecule graph (node types, edge types,
  topology) — no labels, no RDKit-only annotation (graph edge attrs already
  carry bond/aromatic categories in the pipeline; nothing new was invented);
- vocabulary built from train records only in the selection phase
  (`_phase_data` fits on `fit_records` = official train); validation/test
  keys not seen on train → UNK_CONTEXT (1);
- terminal phase refits the context vocabulary on train+valid per the
  `zinc-context-gap` protocol, exactly like the patch vocabularies; test
  labels never participate in any fit or selection;
- `context_mode=none` replay is bit-identical to compact-v2 (parameters,
  valid trace and terminal test 0.1353615), so the v3 code path introduces
  no measurement drift.

## Context statistics (official split, cap 10)

| quantity | train | val | test |
|---|---|---|---|
| node rings (in_cycle) | 63.32% | 63.60% | 63.04% |
| NO_RING | 36.68% | 36.40% | 36.96% |
| mean cycle count per ring node | 1.31 | 1.29 | 1.29 |
| max cycle count per ring node | 15 | 6 | 6 |
| min cycle length (mean) | 5.68 | 5.67 | 5.66 |
| max cycle length (mean) | 6.55 | 6.45 | 6.48 |
| typed context vocabulary (selection) | 9,700 keys (+NO_RING+UNK=9,702) | OOV 3.88% | OOV 3.46% |
| refit vocabulary (train+val) | 10,243 keys (+2 = 10,245) | — | test OOV 3.01% |

Cycle length distribution (train, cap 10): 3:629, 4:166, 5:8,519, 6:18,055,
7:363, 8:301, 9:2,278, 10:1,385 (≈31.7k bounded cycles; a cap of 8 would
silently drop 12% of cycles including most 9/10-cycles — measured, hence cap
10; documented and configurable).

Cycle extraction: ~0.4 ms/graph → ~4.5 s for the whole train split; caches on
the graph record; no training-time cost.  Runtime of the full runs stayed in
the v2 band (318–348 s selection; ~600–650 s terminal).

## Parameter audit

| block | v2 refit | v3 none | v3 coarse | v3 typed condition | v3 typed concat |
|---|---:|---:|---:|---:|---:|
| typed_token_embedding | 37,484 | 37,484 | 37,484 | 37,484 | 37,484 |
| parent_token_embedding | 256 | 256 | 256 | 256 | 256 |
| patch_encoder | 14,192 | 14,192 | 14,192 | 14,192 | 14,704 |
| pair_projection | 768 | 768 | 768 | 768 | 768 |
| relation_encoder | 1,360 | 1,360 | 1,360 | 1,360 | 1,360 |
| distance_gate | 80 | 80 | 80 | 80 | 80 |
| pair_encoder | 5,328 | 5,328 | 5,328 | 5,328 | 5,328 |
| center_context | 15,888 | 15,888 | 15,888 | 15,888 | 15,888 |
| global_encoder | 3,136 | 3,136 | 3,136 | 3,136 | 3,136 |
| graph_head | 21,121 | 21,121 | 21,121 | 21,121 | 21,121 |
| **structural_context_embedding** | 0 | 0 | 16 | 10,248 | 10,248 |
| **context_patch_projection** | 0 | 0 | 128 | 128 | 0 |
| **context_condition_projection** | 0 | 0 | 32 | 32 | 0 |
| **context_delta_output** | 0 | 0 | 144 | 144 | 0 |
| **total (refit)** | **99,613** | **99,613** | **99,933** | **110,165** | **110,117** |
| total (selection) | 98,549 | 98,549 | 98,869 | 108,558 | 108,510 |

Added params: coarse +320 (embedding 16, W_p 128, W_c 32, W_out 144);
typed condition +10,552 (embedding 10,248 = 10,244 rank-1 codes + 4-dim
projection; conditioning 304); typed concat +10,504 (embedding 10,248 +
wider patch encoder 256).  The typed budget note: the refit vocabulary is
protocol-mandated (train+valid); rank is already 1 and the dim/rank
combination minimal, so the documented budget was set to 111,000 and the
measured totals are 110,165 / 110,117 (+0.15% / +0.1% over the 110k ideal).
No parameter is spent on any ring message-passing layer.

## Training setup

Identical across all four variants and to compact-v2: official PyG ZINC
subset 10k/1k/1k, seed 0, Adam lr 1e-3, weight decay 1e-5, batch 128, L1
loss, 60-epoch ceiling, patience-12 early stopping on valid MAE, best-valid
checkpoint, CPU, 4 threads, center-context update on, head 294→64→32→1,
hybrid rank-4 token embedding (768 full rows) + rank-2→rank-4 rare table,
radius-2 patch, 146D descriptor, 23D relation, 5 distance buckets, moments
readout.  Only the context module differs.

## Results

Selection phase (60 epochs, `test_access=blocked`):

| model | params (sel) | best val MAE | Δ valid vs v2 | selected epoch |
|---|---:|---:|---:|---:|
| compact-v2 (existing run) | 98,549 | 0.184158 | — | 56 |
| v3 `context_mode=none` | 98,549 | 0.184158 | 0.000000 | 56 |
| v3 coarse-ring condition | 98,869 | 0.186304 | +0.002146 | 53 |
| v3 typed-ring condition | 108,558 | 0.196666 | +0.012508 | 53 |
| v3 typed-ring concat | 108,510 | 0.201215 | +0.017057 | 60 |

Terminal phase (single official test, checkpoint frozen from valid):

| model | params (refit) | test MAE | Δ test vs v2 |
|---|---:|---:|---:|
| compact-v2 (existing run) | 99,613 | 0.1353615 | — |
| v3 `context_mode=none` | 99,613 | 0.1353615 | 0.000000 |
| v3 coarse-ring condition | 99,933 | 0.1422880 | +0.0069265 |
| v3 typed-ring condition | 110,165 | 0.1600276 | +0.0246661 |
| v3 typed-ring concat | 110,117 | 0.1377695 | +0.0024080 |

Run directories (runs/, git-ignored): context-none screen
`20260907-234719-33edde08` + terminal `20260908-001318-34985928`;
coarse screen `20260907-235240-07953a09` + terminal `20260908-002324-96388cd3`;
typed-condition screen `20260907-235817-8a673dbb` + terminal
`20260908-004655-3215c2f3`; typed-concat screen `20260908-000348-8ab3386d` +
terminal `20260908-005718-1bfd071c`.  Two intermediate terminal attempts
(`20260908-003353-8a97a551`, `20260908-003957-76872c5c`) failed the refit
budget at 110,165/110,117 > 110,000 before the budget was documented at
111,000 for the typed variants; the retries are the reported runs.

## Interpretation

1. **Ring context has signal, but not as implemented.**  The coarse 4D ring
   vector alone (is-in-ring, cycle count, min/max length) does not help
   (+0.0021 valid, +0.0069 test).  The typed signatures make things
   *worse* on validation, and the model overfits (train L1 at epoch 60:
   v2 0.14069 → typed-condition 0.13399, yet valid 0.1967 vs 0.1842).
2. **Typed is not better than coarse — it is worse.**  More context
   granularity = more capacity = more overfitting on this 10k-train task;
   the OOV handling (UNK row is never trained — it is a fixed init code
   since it never appears in train) also leaves 3.9% of ring nodes on a
   generic vector.
3. **Conditioning vs concat:** conditioning (valid 0.1967) beats concat
   (0.2012) on validation, and concat's *test* (0.1378) is much closer to
   baseline than its valid suggests — a large valid-test gap that makes
   concat an unstable reading.  Neither beats v2.
4. **Not a capacity artifact?**  The delta magnitude was measured during
   real training: mean ‖delta‖/‖e_patch‖ ≈ 1.6%, max ≈ 13% — the module
   modulates gently; the degradation comes from the added
   representation-capacity, not from pathological scaling.
5. **Train/val gap:** v2 0.1407→0.1842; typed-condition 0.1340→0.1967 —
   the gap widens (worse generalization), the opposite of what the
   hypothesis needs on a single seed.
6. **Verdict: NO-GO for this first version of context conditioning.**  The
   hypothesis ("the same local pattern should mean something different under
   different structural context") is not refuted at the conceptual level;
   what was tested here is the *first minimal implementation*: a
   context-embedding + rank-8 multiplicative gate on the exact embedding, to
   be contrasted against v2.  Per the pre-registration, an improvement <
   0.003 is "weak/no signal"; we measured a regression of 0.0125 on valid
   and 0.0247 on test (single seed).  Conditional-dictionary work
   (`D(context) = D0 + ΔD(context)`) is **not** the next step until a
   mechanism that modifies the representation *without* adding flat capacity
   (e.g. bounded/shared delta, context-decorated dictionary basis) is shown
   to hold validation on further experiments.

## Next options (not started)

- multi-seed replication of the coarse variants (cheap: +320 params) to see
  whether the small positive/negative differences are seed noise;
- a *shared, low-capacity* context modifier (e.g. delta computed by a few
  shared basis vectors per context — the "conditional dictionary" family,
  deferred by design);
- re-check the exact embedding as the conditioning site: conditioning the
  *patch state* (48D after the encoder) rather than the 16D embedding is a
  different mechanism and a cheap second iteration if the hypothesis is
  retained.

---

## Addendum (2026-09-08): v3 NO-GO 结论的重新解释（不推翻）

v3（coarse-ring 0.186304 / typed-ring 0.196666 / typed-concat 0.201215 vs v2 0.184158）
的 NO-GO 结论**维持**，但因果解释更新：v3 条件化的是**局部** ring context
（patch 半径内的环信息），而信息缺口审计证明 v2 残差的主要机制是 **target 定义中的
全局 long-cycle penalty**（GVAE `max(nx.cycle_basis)−6`，顺序依赖；见
`notes/zinc_long_cycle_audit.md`）。二者是两个不同量：局部环上下文无法表达 graph 全局
环尺度（v3 实验与审计结论自洽）。明确不建议下一步做新的 patch/ring embedding；
唯一推荐：全局不变环尺度通道。
