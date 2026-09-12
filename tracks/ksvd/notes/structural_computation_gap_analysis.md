# Same-Budget Structural Computation Gap Analysis

> **问题。** 在同参数量级（~100K）下，为什么一个 higher-order reference（CIN-small）
> 能用类似预算的参数学得比 optimized compact-v4 更有效？把问题从
> “R 后面漏了哪个统计量” 转成 **“compact-v4 在生成 R 之前采用了什么
> structural computation family？”**
>
> **结论。** **Decision Case A — ONE REPRESENTATION HYPOTHESIS AUTHORIZED**
> （本轮只设计，不运行）。Top-1 = **incidence-preserving structural
> persistence of explicit higher-order objects**：reference 把 cycle/ring
> 实例化成带 persistent learned state 的显式 2-cell 对象，并在若干层里沿显式
> incidence 更新；compact-v4 把 cycle 信息压成一个图级向量，并在一次
> encoder pass 后把 relation 对象 identity 丢进 moment。带一个
> method-philosophy 子条件（minimal 架构需要少量固定轮数的 rank-aware
> 传播，须由 seed0 区分）。
>
> **本轮不做：** 不训练任何新模型；不跑 official test；不重训 compact-v4；
> 不重跑 endpoint / covariance / triad；不实现 CIN；不实现 message passing；
> 不加新 feature；不产出 compact-v7。静态分析 + 参数/计算 profiling + 单变量
> seed0 设计。
>
> Code/artifacts: `tracks/ksvd/results/structural_computation_gap_analysis/`
> （14 个规定文件 + `figures/` + `answers_q1_q20.json`）。
> Tests: `tracks/ksvd/tests/test_structural_computation_gap_analysis.py`.
> Reference primary sources: CIN paper (NeurIPS 2021, arXiv:2106.12575) +
> official code `twitter-research/cwn`.

---

## 1. Motivation

Every local representation route in the track terminated **NO-GO**
(pair endpoint association, centre-incidence covariance, triadic binding,
function-basis accessibility, conditional readout sufficiency, graph-head
family), and the optimized-manifold broad frozen-state screen was
**INCONCLUSIVE — DO NOT BUILD OPTIMIZED OOF**
(`notes/optimized_manifold_broad_state_screen.md`). Meanwhile the optimized
compact-v4 baseline is real and strong (valid 0.146420 seed0; frozen test
0.125179 ± 0.003374) and the global topology channel is a **surviving
architecture GO** (optimized Δtopo +0.019846, `notes/optimized_baseline_critical_revalidation.md`).

The meta-question is no longer “is one more statistic missing?”. It is:

> Which **structural computation family** does compact-v4 use before R, and
> what does a same-scale higher-order reference (CIN-small) spend its
> parameters on instead?

## 2. Why same parameter count is not sufficient

We verified, from the official CIN code, that the official CIN-small config
instantiates **130,945 trainable parameters**, not “100K”. Our compact-v4 has
**99,613** (selection phase). That is *same parameter scale* (~1.31×), **not**
same parameter count; and it is definitely not same **compute** or
**preprocessing** (§9, §21). The paper states only “in the order of 100k”
(no exact number) — so the audit treats the ~100K framing as an
order-of-magnitude match.

## 3. Benchmark comparability

`results/structural_computation_gap_analysis/benchmark_comparability.json` →
**PARTIALLY VERIFIED**.

Verified identical (from official code + paper, source `S1`,`S2`):
PyG `ZINC(subset=True)` (ZINC-12K, official code asserts 10000/1000/1000),
official split, same regression target, MAE, no external data, batch 128.

Not verified / different: seed count (10 vs 1–4), horizon (≤1000 vs 240),
LR schedule (ReduceLROnPlateau vs none), checkpoint rule, weight decay,
hardware, and parameter count (130,945 vs 99,613).

Consequence (hard rule honoured): the performance gap is treated **only** as
architecture headroom evidence. No claim of the form “0.094 vs 0.125 proves
mechanism X” is made anywhere in this audit.

## 4. Optimized compact-v4 computation graph

Reconstructed from the real code
(`experiments/luyin16/zinc_patch_path_pooling.py`, `PatchPathModel`), not from
old summaries. Full serialisation:
`results/structural_computation_gap_analysis/compact_v4_computation_graph.json`.

* **Patch.** One radius-2 ego-net per centre atom (mean **23.08** patches per
  valid molecule). Token = the historical aliased *rooted typed incidence
  certificate* (`pynauty.certificate`); continuous side channel = 146D shell
  descriptor. Patches overlap; an atom belongs to its own patch and every
  patch whose ego-net contains it.
* **Patch state.** `h_i = patch_encoder([146D shell; 0D ctx; 16D typed embed;
  8D parent embed])` = MLP `170→64→48` (14,192 params).
* **Pair state.** *Complete* unordered pair set, mean **264.78** pairs per
  valid molecule. `q_ij = pair_encoder([u_i+u_j; |u_i−u_j|; u_i·u_j·gate;
  23D relation])` = MLP `64→64→16`, computed **once** (5,328 params;
  `pair_projection` 768, `relation_encoder` 1,360, `distance_gate` 80).
* **Centre context.** Each unordered pair row is duplicated to **both**
  endpoints and compressed per `(centre, distance bucket)` into
  `[mean 16 ; population std 16 ; log1p(count) 1] × 5 = 165D`.
* **Centre update.** `h_i ← h_i + MLP([h_i;165D])`, MLP `213→60→48`,
  zero-init residual (15,888), applied **once**.
* **Graph representation.** `R = [unary moments 97 ; pair moments 165 ;
  global 32 ; topology 8] = 302D`.
* **Head.** `302→64→32→1` (21,633), L1.

Key structural facts: **single pass**, **no message passing**, **no explicit
higher-order object**, **no per-cycle object**. Cycle information exists only
as the graph-level 25D topology vector (`zinc_topology_features.py`,
`mode="hinge"`: longest cycle, cycle spectrum 3..10/>10, MCB summary, cycle
rank, `L`, `L²`, `ReLU(L−3..L−10)`) mapped by a 552-param MLP to 8D.

## 5. Compact-v4 structural object lifecycle

`results/structural_computation_gap_analysis/structural_object_lifecycle_v4.json`.

| object | created | persists | identity lost at | survives to R |
|---|---|---|---|---|
| patch `i` | per centre atom | yes (`h_i`) | unary moment pooling | no |
| pair `(i,j)` `q_ij` | per unordered pair | yes within one pass | `_pool_pairs_to_centres` + `_pool_pairs` (immediately) | no |
| centre incidence of `q_ij` | duplicated to both endpoints | no | per-centre moment aggregation | no |
| centre context 165D | from incident pairs | no (statistic) | consumed once by `center_update` | no |
| cycle/ring | — | **does not exist** | already a graph-level scalar | no per-cycle identity |

Object lifecycle in one line:

```
patch i  --(identity preserved)-->  pair (i,j)  --(identity preserved, ONE pass)-->
centre incidence  --(AGgregated to mean/std/log-count)-->  individual relation identity LOST -->
graph moments  --(permutation-invariant pooling)-->  R
```

## 6. Compact-v4 parameter allocation

`results/structural_computation_gap_analysis/parameter_allocation_v4.csv`
(sum == 99,613 == checkpoint total, Test 1).

| component | params | % | structural role |
|---|---:|---:|---|
| typed_token_embedding | 36,420 | 36.56 | per-exact-patch **identity lookup table** (not an operator) |
| graph_head | 21,633 | 21.72 | sole graph readout |
| center_context | 15,888 | 15.95 | one-shot centre update |
| patch_encoder | 14,192 | 14.25 | per-patch operator |
| pair_encoder | 5,328 | 5.35 | per-pair operator (single pass) |
| global_encoder | 3,136 | 3.15 | graph-level chemical descriptor |
| relation_encoder | 1,360 | 1.37 | per-pair relation |
| pair_projection | 768 | 0.77 | patch → pair projection |
| topology_encoder | 552 | 0.55 | graph-level cycle vector MLP |
| parent_token_embedding | 256 | 0.26 | parent token lookup |
| distance_gate | 80 | 0.08 | bucket gate |

Observation: the largest block is an **identity lookup table**, and all
structural operators are applied in a **single** feed-forward pass.

## 7. Reference architecture (CIN-small)

Primary source, not memory. `results/.../reference_computation_graph.json`.

* Paper: *Weisfeiler and Lehman Go Cellular: CW Networks* (Bodnar, Frasca,
  Otter, Wang, Liò, Montúfar, Bronstein), **NeurIPS 2021**, arXiv:2106.12575v3.
* Official code: `https://github.com/twitter-research/cwn`, config
  `exp/scripts/cwn-zinc-small.sh` (unchanged since 2021-07-29 `15e135a4c1`;
  reproduced at HEAD `c4ddd24e2`).
* Config: `ZINC`, `max_dim 2`, `num_layers 2`, `emb_dim(hidden) 48`,
  `max_ring_size 18`, `use_edge_features`, `use_coboundaries True`,
  `init_method sum`, `readout sum`, `final_readout sum`, `readout_dims (0,1,2)`,
  `graph_norm bn`, no dropout, batch 128, Adam lr 1e-3,
  `ReduceLROnPlateau(factor 0.5, patience 20)`, early stop lr<1e-5,
  `epochs 1000`, seeds 0..9, MAE at best-validation epoch.
* **Parameter count.** Paper: “in the order of 100k” (no exact number).
  Our instantiation of the official config: **130,945** trainable
  (`use_coboundaries=True`); without coboundaries **103,009**.
* Reported ZINC MAE (Table 3): **0.094 ± 0.004** (with edge features);
  0.139 ± 0.008 (no edge features). Table 9 No-Rings ablation:
  **CIN No-Rings small 0.174 ± 0.006**, CIN No-Rings 0.159 ± 0.007,
  GIN-E Custom 0.196 ± 0.007; CIN-small 0.094, CIN 0.079.
* Structural preprocessing: induced-cycle ring lifting up to k=18
  (graph-tool, one-off); 2-cell feature = sum of atom embeddings; 1-cell =
  bond embedding.

## 8. Reference structural object lifecycle

`results/.../structural_object_lifecycle_reference.json` (same fields as v4).

| object | created | persists | updated | aggregated | survives to readout |
|---|---|---|---|---|---|
| 0-cell (atom) | input, learnable embedding | yes | every layer (upper-adjacency) | rank-0 sum readout | yes (until readout) |
| 1-cell (bond) | input, learnable embedding | yes | every layer (boundary + upper) | rank-1 sum readout | yes |
| **2-cell (ring)** | **induced-cycle lifting, sum of atom embeddings** | **yes** | **every layer (boundary + upper)** | rank-2 sum readout | **yes** |
| incidence (boundary/coboundary/upper) | lifting index tables | yes | defines messages every layer | — | yes |

## 9. Reference parameter / computation allocation

`results/.../parameter_allocation_reference.csv` (sum == 130,945).

| component | params | % |
|---|---:|---:|
| message passing (rank-0) | 38,400 | 29.33 |
| message passing (rank-1) | 38,400 | 29.33 |
| message passing (rank-2) | 38,400 | 29.33 |
| cell readout `MLP_R` | 14,112 | 10.78 |
| 0-cell embedding | 1,344 | 1.03 |
| 1-cell embedding | 192 | 0.15 |
| final dense | 97 | 0.07 |

**88% of CIN-small's parameters are rank-specific message-passing operators**
(`MLP_M`, `MLP_up`, `MLP_B`, `MLP_U` per rank per layer), i.e. learned
operators applied repeatedly to structural objects — not an identity table.
Compare v4, whose largest block is a lookup table.

## 10. Higher-order structure: exact meaning

CIN-small, decomposed (do not collapse into “uses higher-order structure”):

* **A. Higher-order input object** — YES: explicit 2-cells (rings).
* **B. Higher-order connectivity** — YES: boundary, coboundary, upper-adjacency.
* **C. Higher-order state update** — YES: each ring carries a hidden state.
* **D. Multi-rank interaction** — YES: 0↔1↔2 exchange messages (Eq. 4).
* **E. Repeated computation depth** — YES: 2 stacked layers.

Update (paper Eq. 4), matched to `mp/layers.py::SparseCINCochainConv`:

```
h_sigma^{t+1} = MLP_U( MLP_B( (1+eps_B) h_sigma + Σ_{tau∈B(sigma)} h_tau )
              ||  MLP_up( (1+eps_up) h_sigma
                        + Σ_{tau∈N_up(sigma), delta∈C(sigma,tau)} MLP_M( h_tau || h_delta ) ) )
```

cofaces and down-adjacent cells neglected (`include_down_adj=False`,
consistent with the paper's Theorem 7). `use_coboundaries=True` is the
`MLP_M` term.

## 11. Structural persistence

**Analysis terminology introduced in this audit** (not a literature term):
*Structural Persistence* = the degree to which an explicit object keeps an
independent identity and keeps participating in learned transformations.

* compact-v4: patch identity survives the pair front-end and the one-shot
  centre update, then is pooled. Pair identity survives one encoder pass only.
* CIN-small: every 0/1/2-cell keeps a distinct state and is updated every layer.

## 12. Incidence persistence

**Analysis terminology introduced in this audit**: *Incidence Persistence* =
whether “which lower-order object belongs to which higher-order object” stays
explicitly accessible to later learned computation.

* compact-v4: the relation→centre incidence is consumed by mean/std/log-count
  aggregation *before* `center_update`; afterwards no specific `q_ij`↔centre
  membership is queryable. There is no higher-order incidence at all.
* CIN-small: boundary/coboundary/upper-adjacency tables are available at
  every layer.

**Why this is not “add one incidence statistic”.** The distinction is:
after statistical aggregation, can the original structural object still be
queried and take part in learned computation? compact-v4 answers *no*;
CIN-small answers *yes* for every rank.

## 13. Learned interaction depth

Not receptive field, and not “sees the whole graph”:

* compact-v4: one learned rank transition (patch→pair), one pair→centre
  injection, one readout. **Effective learned depth on the pair path = 1.**
* CIN-small: 2 layers, each with boundary + upper-adjacency + combine at
  every rank. Information can go atom→bond→ring→bond→atom in 2 layers.

Information-path examples: `results/.../` (see §4/§8 tables and
`answers_q1_q20.json` Q12/Q13).

## 14. Global topology comparison

* compact-v4 has an **explicit global structural prior** (25D cycle
  spectrum + MCB + hinges → 8D). This is a *positive* architecture component
  and must be preserved; on this axis compact-v4 is ahead, not behind.
* CIN-small has no separate global prior; global cycle information enters
  through the **learned states of the 2-cells** and their sum readout.

Therefore the Top-1 hypothesis must be **complementary** to the global
topology channel, not a replacement.

## 15. Existing negative evidence from our repo

* endpoint association → **NO-GO** (0/5 folds, worse than R-only).
* centre-incidence covariance → **NO-GO** (worse than matched marginal control).
* triadic relation binding → **NO-GO** (worse than unbound control, 0/5).
* frozen readout sufficiency → **NO-GO**; optimized broad state screen →
  **INCONCLUSIVE** (combined Δ_state +0.00030).
* larger radius, local ring conditioning, embedding sharing, richer readout,
  FM family, head-only late adaptation → all closed.

Interpretation: these exclude “compact-v4 is discarding a *recoverable
statistic of its already-computed states*”. They do **not** exclude a
different object class with a persistent learned state.

## 16. Gap evidence matrix

`results/.../gap_evidence_matrix.csv`. Rows G1–G8 with for/against evidence,
whether already tested here, confounds, priority. Summary:

* **G1 object vocabulary** (rings) — HIGH.
* **G2 structural persistence** — MEDIUM-HIGH.
* **G3 incidence persistence** — MEDIUM.
* **G4 multi-rank interaction** — MEDIUM.
* **G5 repeated refinement** — LOW-MEDIUM (would be Case B).
* **G6 operator parameterization** — LOW.
* **G7 global prior** — NEGATIVE (our advantage).
* **G8 training/compute budget** — HIGH as a **confound**, not a mechanism.

Central evidence: CIN-small ablation No-Rings small **0.174 → 0.094**; the
counter-evidence that depth alone is insufficient (No-Rings, 2 layers, no
rings, is weaker than our single-pass model) indicates the gap is the
**explicit higher-order object**, with depth secondary and confounded.

## 17. Candidate hypotheses rejected

* “compact-v4 lacks a recoverable statistic of its pair states” — rejected by
  the endpoint / covariance / triad / broad-state NO-GOs.
* “more capacity / bigger head / FM head” — rejected (head-family audit
  INCONCLUSIVE; capacity is not the mechanism).
* “repeated message passing is the whole story” — rejected as the *primary*
  gap because CIN No-Rings (2-layer MP, no rings) is weaker than compact-v4;
  retained only as a flagged secondary/confounded factor.
* “global cycle prior is missing” — rejected: we already have it and it is a
  GO.

## 18. Top-1 hypothesis

`results/.../top1_hypothesis.json`.

> **At the same parameter scale, CIN-small's advantage comes primarily from
> explicit higher-order structural objects (induced-cycle 2-cells) that
> (a) are instantiated as identity-bearing objects, (b) carry a learned hidden
> state updated along explicit incidence, and (c) remain available for learned
> computation at readout; compact-v4 instead reduces cycle information to a
> single graph-level pooled topology vector and discards pair-object identity
> into moments after a single encoder pass.**

Primary label: **incidence-preserving structural persistence of explicit
higher-order objects** (G1 with G2/G3 as the enabling mechanism).

Why not another missing statistic: it changes the **computation graph** (new
object class, persistent state, explicit incidence), which no prior audit did.

## 19. Minimal falsification architecture

`results/.../minimal_falsification_plan.json` (design only, not run).

Name: `compact-v4-cell`. **One principle changed:** incidence-preserving
structural persistence.

1. Reuse the existing bounded induced-cycle enumeration (same family as the
   topology channel / `structural_context`); pick one bounded `k`
   (pre-registered, e.g. 6 or 8) — **no sweep**.
2. Ring cell state `h_c = CellInit(mean_{i∈atoms(c)} h_i)`, small MLP.
3. **Exactly 2** pre-registered rank-aware incidence rounds: patch state ←
   mean of incident cell states; cell state ← mean of boundary patch states;
   residual + rank-specific shared MLPs; permutation-invariant.
4. Pool cell states `[mean; std; log1p(count)] = 97D`, append to R
   (`302 → 399`).
5. Existing head consumes the wider R.

Budget: increment ~8–12K → **~108–112K total (≤120K)**. Unchanged: radius,
tokenizer, embeddings, pair/relation encoders, loss, protocol, topology
channel, head convention.

## 20. Method-philosophy compatibility

The hypothesis is compatible *if* implemented as explicit structural objects
with clear meaning, rank-specific structured operators, permutation
invariance, and parameter efficiency. **Honest caveat:** the reference version
is cellular message passing with 2 layers, and the minimal architecture needs
a small fixed number of rank-aware propagation rounds. This is structured,
incidence-restricted propagation — not generic all-pairs message passing and
not a Transformer — but it must be stated plainly. If seed0 shows the effect
requires many/iterative rounds, the decision flips to **Decision Case B
(METHOD-PHILOSOPHY TRADEOFF)**, and implementation is *not* authorized.

## 21. Compute budget

`compute_profile_v4.json` / `compute_profile_reference.json`.

| | compact-v4 | CIN-small |
|---|---|---|
| params | 99,613 | 130,945 (official config) |
| structural objects / valid mol | 23.08 patches, 264.78 pairs, 0 cycles | 23.1 atoms, 25.0 bonds, 2.88 rings |
| learned operator applications / mol | patch 23, pair 265, centre 23, head 1 | rank-specific cell updates ×2 layers |
| iterative rounds | 1 | 2 |
| forward | 24.9 ms / 128-mol synthetic batch (4 CPU threads); full process RSS ~695 MB | paper Table 5: 7.08 s/epoch train on TITAN Xp |
| preprocessing | radius-2 ego-nets + all-pairs shortest paths (one-off, cached) | induced-cycle listing k≤18 (one-off, graph-tool) |

Conclusion: same parameter **scale**, different **compute/preprocessing
budget**. “Same 100K therefore fair” is not valid.

## 22. What is and is not proven

**Proven (static, source-backed):**

* compact-v4 is a single-pass, moment-pooling, non-message-passing
  architecture with no persistent pair identity and no per-cycle object
  (from real code), 99,613 params, R = 302D.
* CIN-small = 2-layer cellular message passing over induced-cycle 2-cells,
  official config 130,945 params, No-Rings ablation 0.174 → 0.094.
* The dataset/split/target/metric are comparable; the protocol/budget are not.

**Not proven:**

* That the per-object mechanism *causes* the performance gap (benchmark
  immaturity + budget confound + object-vocabulary/depth confound).
* That compact-v4's moment pooling is information-theoretically insufficient.
* That the minimal architecture will help; that is the seed0 question.

## 23. Final decision

**Decision Case A — ONE REPRESENTATION HYPOTHESIS AUTHORIZED (design only)**,
with a flagged method-philosophy sub-condition.

* Top-1: incidence-preserving structural persistence of explicit higher-order
  objects.
* Next run: **one seed0** optimized-protocol training of `compact-v4-cell`
  (≤~120K), baseline optimized compact-v4 seed0 valid **0.146420**.
* Gate: `Δvalid ≥ +0.004` **and** mechanism alive (cell-shuffle sensitivity)
  **and** bulk safe; only then buy seed1; never seed3 by default.
* Control: multiset-preserving, incidence-breaking **cell-shuffle** (+
  parameter-matched global cycle-statistic control) to separate incidence
  persistence from extra statistics.
* Official test: **never loaded**.

`results/structural_computation_gap_analysis/final_decision.json`.
