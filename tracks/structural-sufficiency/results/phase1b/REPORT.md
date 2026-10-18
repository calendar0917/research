# SSRC Long-Horizon Research Assessment — Structural Relations, Predictive Performance, and Next Method Direction

**Status:** COMPLETE (one seed for the full arm ladder on two tasks; second seed for the gate-passing relation candidate)
**Protocol:** `ssrc-phase1b-v1`, frozen pre-GPU at commit `43313de`
**Control code:** commit `066ee94` (B1 rewiring, B2 matched-compute schedule)
**Hardware:** `res`, 2×A100-SXM4-40GB, torch 2.5.1+cu124, driver 550.163.01
**Run window:** 2026-10-19 00:15–02:35 (+08:00); 24 runs, all exit 0
**Decision:** **`PURSUE_RELATIONAL_SUFFICIENCY`**

> **一句话总结.** 在真实分子键上做一次性的、端点交换不变的关系组合（R1），
> 相对共享同一 trunk、但不看 root edge 的局部 patch 编码器（S0）带来可复现的
> ZINC 增益（seed 0 +0.0571，seed 1 +0.0992 MAE），且该增益在“度/边类型保持
> 但对应关系打乱”的对照下消失（broken-R1 比 true-R1 差 0.0706，甚至比 S0 更差）。第二轮 tied message passing 的表观增益
> 在匹配计算量后完全消失（M1-long 0.2495 vs M2 0.2522，Δ=−0.0027，CI 跨零）。
> 因此：**关系对应是真的，迭代不是机制**。但 ZINC 上 G_ref 仍明显更强
> （0.1653 vs 0.2495–0.2602），而 Lipo 上 patch 臂与 G_ref 持平；
> 下一步应沿“一次性关系充分性”路线，用静态的全局/深度通路去关闭 ZINC 的
> 绝对性能差距，而不是增加迭代轮数。

---

## A. Research status

### A.1 What this round asked

The track's standing question is: given an explicit local structural
representation \(E_r\) (exact rooted radius-2 patches with atom/bond attributes),
when is learned relational computation necessary — and is it adding information
that \(E_r\) lacks, or only improving learnability/generalization at fixed
information? Phase 1a answered a narrow version on Lipophilicity: with exactly
the same \(E_2\) patch multiset, cross-patch self-attention (X) did **not** beat
shared local encoding + pooling (S); the decision was
`NO_CLEAR_INTERACTION_VALUE_IN_THIS_PILOT`.

Phase 1b asked the next, sharper question, on two tasks:

1. **Does explicit local structure suffice, or do real molecular bonds add
   predictive information?** (S0 → R1)
2. **Is the value in one-shot relational composition, or in iterative
   message passing?** (R1 → M1 → M2, with a matched-compute control)
3. **How far is this family from a strong mature baseline?** (G_ref)
4. **What is the next method direction?**

### A.2 Frozen design (pre-GPU)

| Component | Frozen choice |
|---|---|
| Arms | **S0**: shared 2-layer GINE patch encoder + 2 residual token-FFN blocks + mean pool + head; **no root edges**. **R1**: same trunk + one-shot endpoint-swap-invariant relation `MLP([h_i+h_j, |h_i−h_j|, h_i*h_j, e_ij])` on true root edges; edge readout counts each undirected bond once; **no node-state update**. **M1/M2**: R1 + 1 or 2 tied GINE update rounds; parameter-identical and weight-tied. **G_ref**: 6-layer residual GINE + FFN on the raw atom graph, sum+mean readout. |
| Shared trunk | Bitwise-identical initialization for S0/R1/M1/M2 at the same seed; verified by CPU gate (output diff 0.0). |
| Data (ZINC) | Official **train only** (10,000 rows); repository-frozen internal split `rng(20260922).permutation(10000)` → Fit 8,000 / Monitor 1,000 / Reserved 1,000. Official valid/test never instantiated. |
| Data (Lipo) | Phase 1a frozen scaffold split 2,352 / 504 / 504; Fit+Monitor only; Reserved never read. Labels from the Phase 1a Fit/Monitor-only artifact. |
| Target | ZINC `logP_SA_cycle_normalized` (benchmark-normalized; no extra standardization). Lipo Phase 1a target unchanged. No penalized-logP component in any input. |
| Training | Batch 64, AdamW lr 1e-3 / wd 1e-5, cosine schedule, L1 loss, grad clip 5; ZINC 300 epochs / patience 30; Lipo 200 / 30; best-Monitor-MAE selection; deterministic seed-everything. |
| Endpoints | `Gain_relation = MAE(S0)−MAE(R1)`; `Gain_update = MAE(R1)−MAE(M1)`; `Gain_iteration = MAE(M1)−MAE(M2)`; `Gap = MAE(candidate)−MAE(G_ref)`. Positive gain = right-hand arm better. |
| Uncertainty | Paired bootstrap over Monitor scaffold clusters (Lipo) / molecules (ZINC), 5,000 resamples, 95% percentile CI. Per-comparison, not family-wise. |
| Significance bands | Lipo: negligible <0.02, meaningful ≥0.04. ZINC: negligible <0.01, meaningful ≥0.02. Between the two = weak/inconclusive. |
| Branch logic | B1 relation control (broken adjacency), B2 iteration control (matched compute), B3 all-close, B4 gap-dominates, B5 task disagreement. At most one control per branch. |

The full protocol, including the 5-question decision log and the non-claims, is
`notes/phase1b_protocol.md` (sha256 `25bfff0c…`); the machine-readable config is
`configs/phase1b.yaml` (sha256 `ccb4dccb…`).

### A.3 What was executed

| Phase | Runs | Purpose |
|---|---|---|
| P0 | prep (CPU) + 10 short GPU smokes | pipeline, CUDA kernels, step time/memory |
| A | 10 full runs: {S0,R1,M1,M2,G_ref} × {ZINC,Lipo}, seed 0 | absolute table + endpoint signs |
| B1 | 1 run: R1 with degree/type-preserving broken root adjacency (ZINC) | correspondence vs generic edge channel |
| B2 | 1 run: M1 with a 600-epoch budget (ZINC) | iteration vs extra computation |
| D | 2 runs: S0, R1 seed 1 (ZINC) | run-to-run replication of the relation candidate |

Total: 24 runs, 10 GPU jobs + 1 CPU prep job, all exit 0. GPU spend 1.601 h (run-level; 1.64 h
job occupancy), 20% of the 8 h cap. Wall clock from first GPU launch to last
completion: 2 h 18 m.

### A.4 What changed relative to Phase 1a

- A **second task** (ZINC-12k) with a different target scale, graph size, and
  split protocol; Phase 1a's Lipo pipeline is reused bitwise.
- New **R1/M1/M2** arms that expose true root edges to a shared trunk, plus a
  **strong 6-layer GINE reference** rather than only a 2-layer G.
- A **correspondence control** (broken adjacency) and a **matched-compute
  iteration control**, neither of which existed in Phase 1a.
- Phase 1a's S is reproduced as S0: same parameters (59,011) and same output
  (max abs diff 0.0 on the CPU gate); the Lipo Monitor MAE is 0.5610, matching
  Phase 1a's pooled S at n=2352 (0.5610).

### A.5 Validation and scope audit

- **CPU gates:** 23/23 PASS (`results/phase1b/cpu_gates_phase1b.json`, sha256
  `e5e826b4…`). Key checks: ZINC internal split exactly matches the
  repository-frozen TCCD-v0 permutation; Lipo split/label hashes match Phase
  1a; patch tensors bitwise-match Phase 1a; S0 bitwise-matches Phase 1a S
  (59,011 params); shared trunk init identical across arms; M1/M2
  parameter-identical (107,139); permutation invariance; batch-composition
  invariance; endpoint-swap invariance; S0 ignores root edges; R1/M are
  sensitive to root edges; relation/update gradients live; guard rejections; broken-adjacency invariants in
  the pytest.
- **Tests:** 7/7 fast CPU tests pass in 3.1 s
  (`tests/test_phase1b.py`), including the new B1 rewiring test (degree
  sequence, edge count, type multiset, no self-loops/duplicates, patches
  untouched).
- **Label vault:** each run audits `reserved_labels_retained=0` and
  `official_valid_test_labels_retained=0`; ZINC vault retains 9,000 Fit+Monitor
  labels only; Lipo vault retains 2,856.
- **No leakage:** no official ZINC valid/test; no Lipo Reserved; no ZINC
  Reserved; no penalized-logP inputs; no SSL, ensembles, or hyperparameter
  sweeps; all remote execution/transfer through `rr` on clean commits (one
  read-only `ssh ls` was used only to check file presence during monitoring;
  no result depends on it).

### A.6 What this round cannot answer

- Whether \(E_2\) is **task-information sufficient** in the Bayes-risk sense.
  All arms are functions of the same patch representation; a good S0/R1 does
  not prove \(P(Y|G)=P(Y|E_2)\).
- Whether message passing is necessary in general, at larger depth, larger
  data, or on long-range tasks. The iteration null is local to this scale and
  recipe.
- Whether the Lipo relation gain is nonzero: its CI crosses zero and it is
  below the negligible band.
- Which sub-component of R1 (edge features, endpoint pairing, extra capacity)
  carries the gain. R1 vs S0 confounds relation access with +66% parameters;
  the broken-adjacency control isolates correspondence *given the R1
  architecture*, not the capacity component.
- Anything about official ZINC test performance or leaderboard comparability.

---

## B. Absolute predictive performance

All numbers are **internal train-only Monitor** (ZINC n=1,000; Lipo n=504).
They are not official-test numbers and are not comparable to published ZINC
leaderboards.

### B.1 ZINC-12k (Fit 8,000 / Monitor 1,000), seed 0

| Arm | Params | Forward FLOPs (batch 64) | Step ms | GPU s | Peak MB | Best ep / run | Train MAE | **Monitor MAE** | Monitor RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 56,963 | 434.1 M | 12.69 | 394.9 | 58.8 | 199 / 230 | 0.2617 | 0.3173 | 0.8303 |
| R1 | 94,659 | 644.2 M | 17.13 | 337.0 | 72.4 | 160 / 191 | 0.1743 | **0.2602** | 0.6175 |
| M1 | 107,139 | 691.7 M | 19.91 | 686.7 | 73.8 | 290 / 300 | 0.1533 | 0.2756 | 0.6213 |
| M2 | 107,139 | 949.3 M | 25.10 | 705.6 | 86.9 | 164 / 195 | 0.1723 | **0.2522** | 0.6056 |
| G_ref | 439,559 | 1,185.1 M | 29.99 | 1,051.5 | 81.6 | 250 / 281 | 0.1155 | **0.1653** | 0.4482 |

Constant (training-mean) predictor: **1.5339 MAE**. Every arm beats it
substantially (S0 by 79%, R1 by 83%, G_ref by 89%).

### B.2 Lipophilicity (Fit 2,352 / Monitor 504), seed 0

| Arm | Params | Forward FLOPs (batch 64) | Step ms | GPU s | Peak MB | Best ep / run | Train MAE | **Monitor MAE** | Monitor RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 59,011 | 584.2 M | 5.06 | 90.4 | 67.2 | 125 / 156 | 0.2883 | 0.5610 | 0.7511 |
| R1 | 97,731 | 844.6 M | 10.96 | 92.1 | 84.1 | 106 / 137 | 0.2541 | **0.5521** | 0.7357 |
| M1 | 110,211 | 901.1 M | 11.28 | 97.0 | 84.3 | 107 / 138 | 0.2765 | 0.5716 | 0.7613 |
| M2 | 110,211 | 1,218.1 M | 15.46 | 199.7 | 102.1 | 182 / 200 | 0.2110 | **0.5470** | 0.7278 |
| G_ref | 447,751 | 1,460.5 M | 29.41 | 219.3 | 91.8 | 167 / 198 | 0.2121 | 0.5644 | 0.7650 |

Constant predictor: **0.9545 MAE**. S0 is 41% below it; all arms are far from
the constant.

### B.3 Seed replication (ZINC, relation candidate)

| Seed | S0 MAE | R1 MAE | Gain (S0−R1) | S0 best ep | R1 best ep |
|---|---:|---:|---:|---:|---:|
| 0 | 0.3173 | 0.2602 | **+0.0571** | 199 / 230 | 160 / 191 |
| 1 | 0.3563 | 0.2571 | **+0.0992** | 69 / 100 | 183 / 214 |
| mean | 0.3368 | 0.2587 | **+0.0782** | — | — |

Run-to-run absolute spread: S0 **0.0390**, R1 **0.0031**. R1 is remarkably
stable; S0 is the noisy arm. The gain is positive in both seeds and exceeds
S0's spread in both.

### B.4 Observations

1. **All arms are sane.** They beat the constant predictor by wide margins;
   training curves are finite and converge; best epochs are not all at the
   budget edge (except M1, discussed below).
2. **ZINC has a real absolute gap to G_ref.** G_ref 0.1653 vs M2 0.2522
   (+0.0869, 52.6% of G_ref's MAE), R1 0.2602 (+0.0949), M1 0.2756 (+0.1103),
   S0 0.3173 (+0.1520). All CIs exclude zero (§C.1). G_ref has 4.1× M2's
   parameters, 6 layers, and global sum+mean readout; the gap is confounded
   with depth/capacity/receptive field, but it is large.
3. **Lipo has no gap.** S0 (0.5610), R1 (0.5521), M2 (0.5470) and G_ref
   (0.5644) are within noise of each other; the patch arms achieve this at
   13–25% of G_ref's parameters. On this task, a 59k-parameter static patch
   model matches a 448k-parameter 6-layer GINE.
4. **M1 is undertrained at 300 epochs on ZINC.** Its best epoch is 290/300 and
   it never early-stopped; its train–Monitor gap (0.1223) is the largest of all
   ZINC arms. This is exactly why the B2 control was needed and why its result is
   informative.
5. **Train–Monitor gaps are large on Lipo** (0.27–0.35) and smaller on ZINC
   (0.05–0.12), consistent with the much smaller Lipo Fit set. G_ref fits train
   best on both tasks but does not generalize better on Lipo.

### B.5 Reference context (not a leaderboard comparison)

Published ZINC-12k numbers use the official test split and different target
normalization/evaluation; they cannot be compared numerically to these
internal Monitor values. For orientation only: Eijkelboom et al.'s Table 2
reports full-graph ZINC test MAE ≈0.192–0.263 for MP models and ≈0.527–0.614
for no-MP baselines; GSN ≈0.101 and GraphGPS ≈0.070 are stronger than our
6-layer GINE reference; ESA+ (Buterez et al. 2025) reports 0.052±0.001 on
ZINC-12k. Our G_ref (0.1653 internal Monitor) is a solid mid-strength GINE, not
a SOTA baseline. The scientific contrast here is **within our frozen harness**,
not against those numbers.

---

## C. Mechanism evidence

### C.1 Endpoint table

Positive values mean the right-hand arm has lower MAE.

| Comparison | ZINC Δ [CI95] | Lipo Δ [CI95] | Pre-registered reading |
|---|---:|---:|---|
| **Gain_relation** S0−R1 | **+0.0571** [+0.0307, +0.0863] | +0.0089 [−0.0214, +0.0400] | ZINC **meaningful** (≥0.02, CI>0); Lipo negligible (<0.02, CI crosses 0) |
| **Gain_update** R1−M1 | −0.0154 [−0.0404, +0.0088] | −0.0196 [−0.0567, +0.0166] | One update round does **not** help; same sign both tasks, CIs cross 0 |
| **Gain_iteration** M1−M2 | **+0.0234** [+0.0018, +0.0453] | +0.0246 [−0.0050, +0.0556] | ZINC meaningful at face value; Lipo weak; **collapses under B2 control** |
| Gap S0−G_ref | +0.1520 [+0.1161, +0.1916] | −0.0035 [−0.0428, +0.0356] | ZINC gap; Lipo none |
| Gap R1−G_ref | +0.0949 [+0.0697, +0.1217] | −0.0124 [−0.0537, +0.0282] | ZINC gap; Lipo none |
| Gap M1−G_ref | +0.1103 [+0.0872, +0.1344] | +0.0072 [−0.0350, +0.0497] | ZINC gap; Lipo none |
| Gap M2−G_ref | +0.0869 [+0.0641, +0.1124] | −0.0174 [−0.0588, +0.0226] | ZINC gap; Lipo none |
| **B1** broken-R1−R1 | **+0.0706** [+0.0480, +0.0948] | — | True correspondence beats broken correspondence |
| **B2** M1_long−M2 | **−0.0027** [−0.0209, +0.0152] | — | Matched compute closes the iteration gap |

Bootstrap: paired, 5,000 resamples; Lipo resamples scaffold clusters (504
molecules in 421 clusters), ZINC resamples molecules. CIs are per-comparison,
not family-wise; with 7–9 comparisons per task, a single 95% CI is descriptive,
not a multiple-testing correction.

### C.2 Relation: real bonds add information (S0 → R1)

On ZINC, R1 improves Monitor MAE by **0.0571** over S0, a **18.0%** relative
reduction, with a CI that excludes zero; seed 1 gives **0.0992** (27.8%
relative), and the two-seed mean is **0.0782**. R1's run-to-run spread
(0.0031) is an order of magnitude smaller than the effect; S0's spread (0.0390)
is comparable to the seed-0 effect but smaller than the seed-1 effect, and the
paired sign is positive in both seeds. On Lipo the same sign appears
(+0.0089) but is negligible and not distinguishable from zero.

The absolute level matters: S0 is not a weak model (0.3173 vs constant 1.5339),
so the relation gain is not "edges rescue a broken encoder." It is a genuine
increment on top of a strong local representation.

### C.3 The broken-adjacency control: it is the correspondence, not a generic edge channel

R1 with a **degree-preserving, edge-count-preserving, bond-type-multiset-
preserving** double-edge-swap rewiring of the root adjacency scores **0.3307**,
i.e. **0.0706 worse** than true R1 (CI [+0.0480, +0.0948]) and **0.0134 worse
than S0** (0.3173). The local patch cache is untouched; only the root
adjacency is rewired. The rewiring changed all 10,000 molecules (209,176 swaps
applied) and passed in-run invariants.

Interpretation:

- If the R1 gain were a generic "extra edge channel" or extra capacity, a
  same-capacity R1 with wrong edges should be at least as good as S0. It is
  **worse** than S0: the relation module, given wrong correspondence, fits
  noise and hurts.
- Therefore the gain depends on the **match between patch content and root
  adjacency** — the relational correspondence — not merely on the presence of
  an edge feature or on parameter count.
- This is a strong but not surgical null: the rewiring changes global topology
  (while preserving degree/type/count), so it also removes any long-range
  topological signal that true edges carried. That is the intended
  "broken correspondence" contrast, but it means the control does not separate
  "local correspondence" from "global topology" as the active ingredient.

The rewired slots keep their original bond features, so a rewired edge's
feature no longer describes its new endpoint pair. This is a conservative
choice: it can only make broken-R1 worse, and it means the control tests the
joint consistency of adjacency and edge semantics, not adjacency alone. One
rewiring realization was bought; the effect is large and the CI is far from
zero, so a second realization was not purchased (see decision log P2).

### C.4 One update round does not help (R1 → M1)

`Gain_update` is **negative** on both tasks (−0.0154 ZINC, −0.0196 Lipo): the
one-round model is slightly *worse* than the one-shot relation, with CIs
crossing zero. A single tied GINE round on top of the relation readout does not
add predictive value at this scale. This is already a warning that "more
relational computation" is not the active ingredient.

### C.5 The iteration gain is computation, not mechanism (B2)

Phase A shows M2 beating M1 by +0.0234 on ZINC (CI [+0.0018, +0.0453]) and
+0.0246 on Lipo (CI crosses zero). The pre-registered B2 control trains **M1**
(the one-round model) with a 600-epoch budget, matching M2's GPU time while
using fewer relation applications than M2:

| Quantity | M2 (2 rounds) | M1_long (1 round) |
|---|---:|---:|
| Epochs run / best | 195 / 164 | 285 / 254 (early stop) |
| Relation applications | 390 | 285 |
| GPU seconds | 705.6 | 713.8 |
| Monitor MAE | 0.2522 | **0.2495** |
| Δ (M1_long − M2) | — | **−0.0027** [−0.0209, +0.0152] |

M1_long **matches (slightly beats) M2** at matched GPU time (713.8 vs 705.6 s) and with *fewer*
relation applications (285 one-round applications vs M2's 390 two-round
applications). The Phase A iteration advantage is therefore a
schedule/compute effect: M1 at 300 epochs was undertrained (best epoch 290,
train–Monitor gap 0.1223), and giving the one-round model a longer cosine
schedule recovers the apparent M2 gain. Under the pre-registered rule, this is
the outcome "iteration is not separated from extra computation."

This does **not** say iteration can never help at larger depth, larger data, or
on long-range tasks. It says that at this scale, on these two tasks, with this
recipe, the second tied round has no mechanism value beyond optimization
budget.

### C.6 What is established, and what remains confounded

**Established (within the frozen harness):**

1. A one-shot endpoint-swap-invariant relation on true root edges improves
   over a strong local-patch encoder on ZINC (+0.057/+0.099 across two seeds;
   CI excludes zero at seed 0).
2. The improvement is correspondence-specific: a degree/type-preserving broken
   adjacency is worse than true adjacency by +0.0706 and worse than S0.
3. A second tied message-passing round adds nothing beyond a longer schedule
   at matched compute (Δ=−0.0027, CI crosses zero).
4. On Lipo, static patches alone match a 6-layer GINE at 13% of its
   parameters; on ZINC they do not.
5. The ZINC gap to G_ref (0.087–0.152) is real and consistent, but
   task-dependent (absent on Lipo) and confounded with depth/capacity/global
   receptive field.

**Not established / confounded:**

- **Capacity vs relation in R1−S0.** R1 has 66% more parameters than S0. The
  broken control shows correspondence matters *given R1's capacity*, but does
  not decompose how much of the S0→R1 gain is capacity vs information access.
  A capacity-matched S0 (same parameter count, no root edges) was not bought;
  the protocol forbids naive capacity scaling without a matched control.
- **Which part of the relation module matters.** Edge features, endpoint
  pairing, the swap-invariant symmetric form, and the extra MLP capacity are
  bundled in R1. No component ablation was pre-registered or bought.
- **Global topology vs local correspondence.** The broken rewiring preserves
  degrees and type counts but scrambles global topology; both are removed
  together.
- **Task dependence.** Two tasks, two splits. The Lipo relation gain is not
  significant; the ZINC gain is. A third task/split is needed before a general
  claim.
- **The ZINC gap's cause.** Depth, width, global readout, receptive field, and
  optimization budget are confounded in G_ref. No matched-capacity G_ref
  control was bought (the protocol's Phase C was not triggered because no
  single evidence-backed fix was identified).

### C.7 Relation to the reference literature

- **GSN (Bouritsas et al. 2022)** showed substructure counts add expressive
  power. Here the patches are exact attributed rooted neighborhoods, and the
  question is computational: a one-shot relation over patch identities captures
  the gain, while an extra MP round does not.
- **GraphGPS (Rampášek et al. 2022)** combines local MP with global attention;
  our G_ref is GINE-only, not GPS. The ZINC gap is consistent with the known
  value of global/long-range access, but we did not test a GPS-style arm.
- **GRIT (Ma et al. 2023)** uses global attention with random-walk positional
  encodings; not tested here. The ZINC gap motivates a static global pathway,
  not necessarily GRIT's full mechanism.
- **CSA / ESA+ (Buterez et al. 2025)** are compute-efficient global
  self-attention methods; ESA+ reports ZINC-12k 0.052±0.001 on the official
  test. Our internal Monitor numbers are not comparable, but the direction
  (global access matters on ZINC) matches.
- **Edge Transformer (Bergen et al. 2021)**, **MoSE**, **SPSE**, and
  **Eijkelboom et al.** all put computation at the relation/edge level rather
  than only at nodes. Our result is consistent with that line — explicit
  relation representation reduces the need for iterative MP — and adds a
  controlled correspondence test plus a matched-compute iteration null.
- **Eijkelboom et al.** report no-MP ZINC test MAE ≈0.527–0.614 vs MP
  ≈0.192–0.263. Our S0 (0.317 internal Monitor) vs G_ref (0.165) shows the same
  qualitative direction on a different split/target; it is not a numerical
  replication.

The novel contribution of this round is not "edges help" or "MP helps on
ZINC." It is the **joint controlled decomposition**: (i) the relation gain
survives a capacity-preserving broken-correspondence control, and (ii) the
iteration gain does not survive a matched-compute control. That combination
separates *information access* from *computation mechanism* on the same frozen
representation.

---

## D. Resource spend

### D.1 Ledger

| Phase | Runs | GPU-hours (run-level) | Job occupancy | Cap | Used / cap |
|---|---:|---:|---:|---:|---:|
| P0 smoke | 10 short | 0.002 | 0.015 | reserve 0.5 | <1% |
| A matrix | 10 | 1.076 | 1.094 | 2.0 | 54% |
| B controls | 2 | 0.298 | 0.305 | 2.0 | 15% |
| C improvement | 0 | 0.000 | 0.000 | 2.0 | 0% |
| D seed | 2 | 0.224 | 0.228 | 1.5 | 15% |
| **Total** | **24** | **1.601** | **1.642** | **8.0** | **20%** |

Components are rounded; exact run-level total is 1.6005 h.

- All 10 GPU jobs exited 0; no failed-job GPU cost.
- The largest single job was the 4-run ZINC Phase A loop (2,808 s ≈ 0.78 h
  occupancy; 2,780.8 s run-level).
- Wall clock from first GPU launch (00:17:07) to last completion (02:35:12)
  = 2 h 18 m. Local CPU work (implementation, gates, analysis, report) is not
  GPU time.
- Under-spend is deliberate: Phase C was not triggered because no single
  evidence-backed fix was identified, and no purchase was made that could not
  change the route decision.

### D.2 Budget interpretation

The 8 h cap is not a target. This round bought the minimum evidence needed to
separate the pre-registered hypotheses: 10 matrix runs, 2 controls, 2 seed
runs. The remaining ~6.4 GPU-h is available for the next round's matched-
capacity/global-pathway tests, which are the natural follow-up to §E.

### D.3 Run provenance

`results/phase1b/run_manifest.tsv` (sha256 `bda11c2d…`) lists every run: phase,
experiment, rr run_id, dataset, model, seed, label, GPU seconds, job seconds,
best epoch, Monitor MAE, parameters, step time, commit, output sha256, path.
All formal runs used clean commits (`43313de` for Phase A; `066ee94` for
controls and Phase D) with no `--allow-stale`/`--allow-dirty`.

---

## E. Route decision

### E.1 Decision

## **`PURSUE_RELATIONAL_SUFFICIENCY`**

The next method direction is to build on the **one-shot, endpoint-swap-
invariant relation on true root edges** as the computational core, and to close
the ZINC absolute-performance gap with a **static global/depth pathway** — not
with additional iterative message-passing rounds.

### E.2 Why this route, and not the alternatives

| Candidate route | Verdict | Reason |
|---|---|---|
| **PURSUE_RELATIONAL_SUFFICIENCY** | **Chosen** | The one-shot relation is the only intervention with a reproducible, correspondence-specific gain (two seeds, broken-adjacency control). It is cheap (R1: 94.7k params, ≈0.09 h ZINC training, 337 GPU-s) and its mechanism is identified. |
| PURSUE_ITERATIVE_RELATIONS | Rejected | The B2 matched-compute control closes the M2−M1 gap (Δ=−0.0027, CI crosses 0). The second tied round has no mechanism value beyond optimization budget at this scale. |
| PURSUE_EFFICIENT_STATIC_STRUCTURE | Rejected as the single route | Supported on Lipo (S0 0.5610 ≈ G_ref 0.5644 at 13% params) but contradicted on ZINC (S0 0.3173 vs G_ref 0.1653; S0 is the weakest arm). Static structure is efficient, not sufficient in general. |
| PERFORMANCE_GAP_DOMINATES | Not chosen | The ZINC gap is real (0.087–0.152), but it is task-dependent (absent on Lipo) and confounded with depth/capacity/global access rather than traced to a failure of the relational mechanism. It is the **engineering target** of the chosen route, not a reason to stop mechanism work. If the next round's capacity/global controls fail to close it, this label becomes correct. |
| REFRAME_RESEARCH_QUESTION | Not yet | The task-dependence (local suffices on Lipo; global/depth needed on ZINC) is an important finding, but the mechanism result is strong enough to pursue one more round before reframing. A reframe would be justified if the ZINC gap survives matched-capacity/global controls. |
| INSUFFICIENT_EVIDENCE | Rejected | The relation effect replicates across two seeds with a CI excluding zero and survives a strong control; the iteration null is confirmed by a matched-compute control. The evidence is sufficient for a directional decision. |

The decisive asymmetry: **the relation effect is robust to the control that
should kill it if it were an artifact; the iteration effect is not.** That is
what makes relational sufficiency, not iteration, the productive route.

### E.3 Next method direction (concrete, pre-registerable)

**Working title:** *One-shot relational sufficiency with a static global
pathway* (working name `R1-G`).

1. **Keep** the frozen S0 trunk and the R1 one-shot relation readout.
2. **Add** a static global/depth pathway: either (a) 2–4 learned global tokens
   attended once by the patch tokens, or (b) a deeper residual trunk (4–6
   blocks) with the same one-shot relation. Both are static (no iterative
   node-state updates).
3. **Match capacity to G_ref** (≈440k params) and report a capacity-matched
   S0 control (same trunk/head, no root edges) to decompose the S0→R1 gain
   into capacity vs relation access.
4. **Pre-register three controls:** capacity-matched S0; broken-adjacency R1
   (one realization, same invariants); G_ref at matched capacity. If the
   global/depth pathway closes the ZINC gap **without** the relation, the route
   shifts to `PERFORMANCE_GAP_DOMINATES` / static structure.
5. **Do not add iterative MP rounds.** If a round is tested, use the
   M1_long-style matched-compute protocol; a single 300-epoch M1 is not a fair
   iteration test (it is undertrained).
6. **Evaluate on ZINC + a third task/split** before any paper-level claim. The
   third task should have a different graph-size/long-range regime to test the
   task-dependence directly.

### E.4 Falsification criteria for the next round

- **Capacity explanation:** if capacity-matched S0 ≈ R1 on ZINC, the relation
  gain is capacity → downgrade to `PERFORMANCE_GAP_DOMINATES` / static
  structure.
- **Global-pathway explanation:** if a static global/depth pathway closes the
  ZINC gap without the relation, relational sufficiency is not the active
  ingredient → `PERFORMANCE_GAP_DOMINATES`.
- **Task-conditioned reversal:** if a third task/split reverses the relation
  sign, downgrade to a task-conditioned claim or `INSUFFICIENT_EVIDENCE`.
- **Persistent gap:** if the ZINC gap survives matched capacity and global
  access, `REFRAME_RESEARCH_QUESTION` (the question becomes local vs global
  information, not one-shot vs iterative computation).

### E.5 What we will not do

- No iterative MP expansion on the basis of the Phase A M2 number (B2 rules it
  out at this scale).
- No hyperparameter sweep, SSL objective search, ensembles, or architecture
  zoo.
- No official ZINC valid/test and no Lipo/ZINC Reserved labels.
- No claim of SOTA or leaderboard comparability from internal Monitor numbers.
- No post-hoc change of the frozen bands or branch logic.

---

## F. Delivery checks

### F.1 Files and hashes

| Artifact | Path | sha256 |
|---|---|---|
| Frozen protocol | `tracks/structural-sufficiency/notes/phase1b_protocol.md` | `25bfff0c…` |
| Frozen config | `tracks/structural-sufficiency/configs/phase1b.yaml` | `ccb4dccb…` |
| ZINC internal split | `results/phase1b/split_zinc_phase1b.json` | `fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49` |
| ZINC Fit+Monitor labels | `results/phase1b/zinc_labels_fit_monitor.json` | `631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78` |
| CPU gates | `results/phase1b/cpu_gates_phase1b.json` | `e5e826b4…` |
| Analysis | `results/phase1b/analysis_phase1b.json` | `ae5de61b…` |
| Run manifest | `results/phase1b/run_manifest.tsv` | `bda11c2d…` |
| Decision log | `results/phase1b/decision_log.md` | (see commit) |
| This report | `results/phase1b/REPORT.md` | (see commit) |

Code: `code/phase1b/{ssrc_data,ssrc_models,ssrc_train,ssrc_rewire,run_phase1b,phase1b_gates,analyze_phase1b,make_run_manifest}.py`;
tests: `code/phase1b/tests/test_phase1b.py`.

### F.2 Tests and gates

- `uv run pytest tracks/structural-sufficiency/code/phase1b/tests/test_phase1b.py -q`
  → **7 passed** (3.1 s), including the B1 rewiring invariants (degree
  sequence, edge count, type multiset, no self-loops/duplicates, patches
  untouched).
- `uv run python tracks/structural-sufficiency/code/phase1b/phase1b_gates.py`
  → **23/23 PASS**; S0-vs-Phase-1a output diff 0.0; split exact match; reserved
  labels requested/blocked; endpoint-swap invariance 0.0; M1/M2 parameter
  identity.
- Every run JSON contains a `label_vault` audit with
  `reserved_labels_retained=0` and
  `official_valid_test_labels_retained=0`.

### F.3 Reproduce

```bash
# CPU gates + tests
uv run python tracks/structural-sufficiency/code/phase1b/phase1b_gates.py
uv run pytest tracks/structural-sufficiency/code/phase1b/tests/test_phase1b.py -q

# prep (CPU; builds caches, split, labels)
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py prep --dataset zinc
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py prep --dataset lipo

# one training trajectory (example)
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py train \
  --dataset zinc --model R1 --seed 0 --device cuda

# controls
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py train \
  --dataset zinc --model R1 --seed 0 --device cuda \
  --edge-rewire-seed 20261019 --label rewire
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py train \
  --dataset zinc --model M1 --seed 0 --device cuda --epochs 600 --label long

# analysis + manifest
uv run python tracks/structural-sufficiency/code/phase1b/analyze_phase1b.py
uv run python tracks/structural-sufficiency/code/phase1b/make_run_manifest.py
```

All formal remote runs used `rr run res ... --pool res-gpu{0,1}` on clean
commits; see `results/phase1b/run_manifest.tsv` for the exact run IDs.

### F.4 Non-claims

- This is **not** a proof that \(E_2\) is task-information sufficient, and not
  a Bayes-risk equality claim.
- This is **not** a general claim that message passing is unnecessary; it is a
  matched-compute null for one extra tied round at this scale/task/recipe.
- This is **not** a SOTA or leaderboard claim; all numbers are internal
  train-only Monitor, not official test.
- The broken-adjacency control is **not** a surgical null: it preserves
  degree/type/count and leaves patches untouched, but scrambles global topology
  and removes any long-range signal the true edges carried.
- The Lipo relation gain is **not** established: its CI crosses zero and it is
  below the negligible band.
- The capacity-vs-relation decomposition is **not** complete: R1 has 66% more
  parameters than S0; the broken control isolates correspondence at fixed R1
  architecture, not the capacity component.

### F.5 Open questions (ranked for the next round)

1. Does a capacity-matched S0 close the S0→R1 gap on ZINC? (capacity vs
   relation)
2. Does a static global/depth pathway close the ZINC gap to G_ref without
   iterative MP? (the chosen next method)
3. Is the relation gain present on a third task/split with a different
   graph-size regime? (task dependence)
4. Which sub-component of R1 carries the gain: edge features, endpoint
   pairing, or the symmetric swap-invariant form? (component ablation, only
   after 1–2)
5. Why is Lipo insensitive to the relation, while ZINC is sensitive? (label
   noise, target scale, graph size, or scaffold split)

### F.6 Compliance matrix

| Requirement | Status |
|---|---|
| ≤8 h wall clock | ✅ ~3.5–4 h elapsed, 2 h 18 m GPU window |
| ≤8 A100 GPU-h | ✅ 1.601 h run-level, 1.642 h occupancy |
| Phase caps A/B/C/D/reserve | ✅ 1.076 / 0.298 / 0.000 / 0.224 / 0.002 |
| S0/R1/M1/M2 + strong G_ref | ✅ all five arms on both tasks |
| Gain_relation / Gain_iteration / Gap | ✅ paired bootstrap CIs, both tasks |
| Information access vs computation mechanism | ✅ broken-adjacency and matched-compute controls |
| ZINC official train only; internal 8k/1k/1k | ✅ official valid/test never instantiated |
| Lipo frozen 2352/504/504; Fit/Monitor only | ✅ split/label hashes match Phase 1a |
| No penalized-logP leakage | ✅ target/inputs audited |
| CPU audits before GPU; frozen protocol | ✅ 23 gates + 7 tests; protocol frozen at `43313de` |
| Decision gates + 5-question log | ✅ `results/phase1b/decision_log.md` |
| Branch logic B1–B5 | ✅ B1/B2 controls executed; B3 not triggered (ZINC has meaningful differences); B4 only on ZINC (not Lipo); B5 task-dependence checked and discussed in §C.6/§E |
| Run manifests + provenance | ✅ `run_manifest.tsv`, per-run JSON with commit/env |
| No SSL/ensembles/sweeps | ✅ none |
| Stop-early / under-spend | ✅ Phase C not bought; 20% of budget |
| Single route decision | ✅ `PURSUE_RELATIONAL_SUFFICIENCY` |
| Report sections A–F | ✅ this document |

---

### Bottom line

The local-structure question now has a clean, two-task answer. **Real bonds
add predictive value through correspondence, one-shot composition captures it,
and the extra MP round does not.** The second message-passing round is not a mechanism; it is a longer
optimization schedule in disguise. On small molecules, static patches are
already competitive with a much larger GINE; on ZINC, the remaining gap is
global/depth, not iteration. The next method should therefore keep the
validated one-shot relation and spend its capacity on a static global pathway,
with the capacity-matched controls specified in §E.3–E.4. If that fails to close
the ZINC gap, the correct next label is `PERFORMANCE_GAP_DOMINATES`, not more
iterative relations.
