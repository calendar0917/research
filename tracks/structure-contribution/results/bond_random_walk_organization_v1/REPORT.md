# Bond random-walk structural organization v1 — REPORT

- protocol `cscl-bond-random-walk-organization-v1` · decision **RW_STRUCTURE_USED_AND_GAIN**
- commit `10496f29243c3a55ca7cb123d00c8322f1c55e4f` · Part A decision `PART_A_PASS_GPU_PURCHASE`
- execution regime: host `res` (a100-2), driver 550.163.01, torch 2.5.1+cu124, pool `res-gpu1` (`CUDA_VISIBLE_DEVICES=1`)
- GPU wall: **0.2461 h** of the 1.5 h cap; R 446.0 s + U 440.1 s
- Monitor read exactly once after both prediction files were frozen (lock
  `2026-10-14T11:48:13Z`, `y_sha256=bbcf6faf…`, prediction manifest
  `078703e4…`); Locked never read; official valid/test never loaded.

## 1. Question and arms

Without a learned Junction encoder: does a fixed multi-scale random-walk relation
on the real physical-bond line graph, feeding a shared Bond–Bond Pair MLP, add
anything over the same machinery with uniform global pair weights?

| arm | pair weights K | params | init | SSL encoder | batch order | recipe |
|---|---|---:|---|---|---|---|
| R | `Ks = sym((P+P^2+P^3)/3)` on the real bond line graph | 349,985 | `90bf6322…` | `71bb65877…` | 22680 updates | R3 |
| U | `K_ef = 1/N_bond` | 349,985 | `90bf6322…` | `71bb65877…` | matched hash | R3 |

Both arms: one shared scratch init, Fit-only masked-SSL atom/bond encoders, raw-y
L1, Adam 1e-3 cosine→1e-5, wd 1e-5, clip 5, batch 128, shuffle seed 101,
360 epochs / 22,680 updates, last-5 epoch-state soup. The only arm difference is
the unordered pair-weight list handed to the identical model
(`φ(b_e,b_f)=MLP([b_e+b_f, |b_e−b_f|, b_e⊙b_f])`, `r_e=Σ_f K_ef φ`), and
`φ` is exactly symmetric so using `K` is identical to using `Ks` (unit-tested).

## 2. Part A — label-blind structure audit (first 2000 split-D Fit rows)

- lazy walk `P=(I+D⁻¹A_L)/2`, zero-degree rows one-hot; every row sums to 1
  exactly (max err 2.2e-16); `P¹..P⁴` exact float64 products; triplet
  reconstruction of the double sum err ≤ 1.9e-7.
- mixing: mean TV(P³,U)=**0.688** (gate ≥0.03), mean TV(P⁴,U)=0.651,
  normalised entropy 0.371→0.577 (k=1→k=3), reachable fraction 0.158→0.442.
- **scale distinctness** mean row-TV(P¹,P³)=**0.250** (gate ≥0.05).
  Amendment-1 (config, pre-execution, no data): the originally written
  `||P³−P¹||₁/(2N²)` is bounded by 1/N (mean bound 0.042), so the gate was
  unsatisfiable; the corrected standard row-Total-Variation definition was
  frozen before any audit run, threshold unchanged. Raw value recorded
  (0.0104) as `scale_distinctness_raw_p1p3`.
- K mean TV to uniform **0.744** (gate ≥0.05); K effective rank 8.97.
- invariance: atom-relabel max K diff 0.0; bond-permutation conjugation
  1.1e-16; statistics 8.9e-15.
- controlled organisation pairs: 208 pairs of real Fit molecules with
  **identical** raw marginals (n_bonds, bond-type multiset, line-graph degree
  multiset, per-bond junction-count multiset) but a different K spectrum
  (mean sorted-entry max-abs diff 0.023, max 0.050). The abstract
  junction-moment counterexample (4 bonds / 4 junctions, same marginals,
  line graph C4 vs 2K2, K diff 0.25) is explicitly **not a legal molecule**.
- communities on the same line graph: self-contained deterministic Walktrap
  (amendment-2: networkx 3.4.2 has no `walktrap_communities`; Pons–Latapy
  mean-row distances on P⁴, modularity-max along the deterministic merge path)
  mean 3.63 communities/molecule vs TASU 2ECC/bridge 10.16 and greedy
  modularity 4.33; ARI(Walktrap, TASU 2ECC/bridge)=0.506,
  ARI(Walktrap, greedy)=0.629; unique bond membership 100%; all partitions
  cover all bonds; 93.1% of communities connected; singleton-community share
  3.4%; ring-like share 10.6%; relabel ARI min −0.024 / mean 0.994 / median
  1.000 / 98.0% ≥0.9 (the minimum is a small number of near-tied modularity
  cases, disclosed as a limitation of the deterministic tie handling).
- boundary statement: bond communities assign every bond to exactly one
  community, but they do **not** guarantee unique atom membership; among atoms
  with >1 incident bond, 17.5% have incident bonds in more than one community.
  Nothing here is a chemistry claim.

## 3. Monitor results (split D, n=1000, seed 0, historically exposed)

| arm | MAE | RMSE | ordinary y>−3 (n=929) | tail y≤−3 (n=71) |
|---|---:|---:|---:|---:|
| R | **0.107520** | 0.319330 | **0.090779** | **0.326569** |
| U | 0.117248 | **0.306442** | 0.100714 | 0.333583 |

- **Primary R − U = −0.009728**, paired molecule bootstrap CI95
  **[−0.015539, −0.004015]** (both bounds negative), fraction R better 0.540.
- ordinary Δ **−0.009936**; tail Δ **−0.007014**; top-10-removed Δ
  **−0.011044** (more negative than the full-set Δ, so the MAE gain is not
  driven by a few molecules); max single-molecule MAE-gain share 1.5%.
- RMSE moves the other way: R 0.3193 vs U 0.3064 (+0.0129 for R). Since MAE,
  ordinary, tail and top-10-removed all favour R, this is a small number of
  large-error rows where R's error is heavier; disclosed as a real tail risk of
  the structured arm. It is exactly the opposite pattern to the historical
  A2-vs-A0 RMSE observation, which was also single-molecule-dominated.
- Fit/Monitor gap disclosure: soup Fit MAE is 0.001882 (R) / 0.002979 (U)
  while Monitor MAE is ~0.11 — both arms fit the 8000 Fit rows almost perfectly
  and overfit strongly. The R−U contrast is measured under *matched*
  overfitting (identical recipe, init, encoder, order), so it isolates the
  relation weights, but the absolute Fit numbers must not be read as
  generalisation.
- gradient clipping: 9,592 (R) / 9,699 (U) clipped updates of 22,680, max
  clipped norm 20.6 — matched between arms.

Read-only prior-round references (published, different trajectories and
parameter budgets; **descriptive only, no attribution**):

| reference | Monitor MAE | R − ref (descriptive) |
|---|---:|---:|
| A0 no-junction-content (prior round) | 0.121514 | **−0.013994** |
| A1 (prior round) | 0.106239 | +0.001281 |
| A2 / P3 BJOC-Null (prior round) | 0.103960 | +0.003560 |

R clearly beats the historically weakest A0 reference and is close to, but
descriptively behind, the stronger A1/A2 references whose models have a
different budget (413,505 params) and mechanism; no architecture attribution is
made from these gaps.

## 4. Mechanism evidence (Part C, label-blind)

- **C1 — the structure is used.** R vs U Monitor predictions differ by mean
  |Δ| 0.0760 (max 0.971, corr 0.9984). The trained Pair-MLP gradients are live
  (per-batch norms 0.65/0.08/0.24/0.20 and 0.73/0.05/0.29/0.22). Controlled
  synthetic relation task (target `Σ K_ef(b_e·b_f) + 0.1Σb_e`): real-K model
  test MAE **0.717** vs the same model evaluated with uniform K 23.79 vs the
  uniform-K-trained model 4.45 vs the linear-summary model 7.64. The
  junction-moment counterexample shows identical raw marginals with K differing
  by 0.25 (abstract, non-molecular).
- **C2 — not reducible to a linear weighted sum.** With `b` and `φ` frozen,
  reassigning only the relation weights changes predictions by mean |Δ| 0.271
  (uniform), 0.157 (random row-stochastic), 0.153 (bond-slot permutation),
  0.220 (zero) — the same bond marginal set gives different outputs. The
  trained Pair MLP uses the joint input blocks (zeroing the product block
  changes φ by 76% relative, the difference block by 220%, the sum block by
  55%), and the exact linear-reweighting surrogate
  `r_e ~ W(Kb)_e + c·rowsum_e` leaves **52.3%** relative residual.
- **C3 — interpretation boundary.** The random-walk relation is not a faithful
  contribution decomposition and none of this is a chemistry-causality claim.
  The round shows only that the actual topological organisation contributes
  predictive information beyond an unstructured global pair summary, through a
  non-linear pair channel.

Acceptance (frozen thresholds): `k_effect_pass`, `pair_grad_live_pass`,
`product_block_used_pass`, `linear_summary_residual_pass` all true.

## 5. Part D — the five questions (final answers)

1. **多尺度信息是否真实存在、k=2/3 是否没有退化成全局均值？**
   是。Part A 的标签盲审计里，P³ 到均匀分布的 TV 距离均值 0.688（门槛 0.03），
   K 的 TV 距离 0.744（门槛 0.05），P¹→P³ 的行 TV 尺度差异 0.250（门槛 0.05），
   归一化熵从 0.371 升到 0.577，可达比例从 0.158 升到 0.442；所有门通过。
   208 对真实分子具有完全相同的原始边际（键数、键型多重集、线图度多重集、
   每键 junction 计数多重集）却有不同 K；抽象反例同样显示边际相同而 K 差 0.25
   （明确标注非真实分子）。因此 k=2/3 提供的不是全局均值的重新包装。
2. **Walktrap 社区是否与 TASU 2ECC/Bridge 有实质差异、是否稳定，社区与原子
   归属边界如何？**
   Walktrap 平均 3.63 个社区/分子，TASU 2ECC/Bridge 10.16、贪心模块度 4.33；
   ARI(Walktrap, TASU)=0.506，ARI(Walktrap, greedy)=0.629，说明二者有共享骨架
   但并非同一划分。稳定性：重标号 ARI 均值 0.994、中位数 1.000、98.0% ≥0.9，
   最小值 −0.024（少数模块度近似并列的确定性合并路径，已披露为限制）。
   每个键恰好属于一个社区，但社区不保证原子唯一归属：在有多个入射键的原子中
   17.5% 的入射键跨越多个社区；因此不能把键社区直接当作原子/化学单元划分，
   也不作化学因果声明。
3. **真实连通性是否打败了无结构的全局 pair 对照？**
   是。冻结的 R−U 主对比：MAE 0.107520 vs 0.117248，Δ = −0.009728，
   配对 bootstrap CI95 [−0.015539, −0.004015] 完全为负；ordinary（y>−3，
   n=929）Δ −0.009936，tail（y≤−3，n=71）Δ −0.007014，去掉最差 10 个分子后
   Δ 反而更负（−0.011044），单分子最大 MAE 增益占比 1.5%。代价是 RMSE 反向
   （R 0.3193 vs U 0.3064），来自少数大误差行，已如实披露。
4. **多尺度组织是否值得作为完整结构模型继续开发？**
   值得继续，但只以有限形式：本轮证明真实线图组织→共享 Pair MLP 的非线性通道
   确实带来跨分子可迁移增益（C1/C2 全过，C2 线性代理残差 52.3%，K 重分配使
   预测平均变动 0.15–0.27）。但绝对水平仍描述性落后于参数更多的 A1/A2/P3
   参考（+0.0013/+0.0036），且两臂都严重过拟合（Fit MAE ~0.002 对 Monitor
   MAE ~0.11），因此下一轮若继续需先解决泛化/容量-正则平衡，而不是加大关系
   通道；本轮按预注册停止，不追加 seed、不搜索游走长度/宽度、不进入下一轮。
5. **支持证据与停止信号是什么？**
   支持：Part A 全门通过（mixing 0.03/0.05、尺度 0.05、不变性 ≤1.1e-16）、
   R−U CI 全负、机制验收四项全真（K 效应、Pair 梯度活跃、乘积块被使用、
   线性代理残差 0.52）、合成关系任务真实 K 0.717 远优于 uniform 4.45/线性 7.64。
   停止信号/限制：单 seed（冻结规则：无明确机制证据才有第二 seed，但本轮已明确，
   故不加）；RMSE 与 tail 的极端行风险；社区重标号最小值 −0.024；全部结果基于
   历史暴露的 split-D Monitor，只能作探索性评价；Locked 从未读取，official
   valid/test 从未在本轮读取；不进行化学因果归因。

## 6. Discipline

- no GNN/attention/iteration; no learned Junction encoder; no walk-length,
  width or community-parameter search; no second seed; Locked and official
  splits untouched; the two arms differ only in the pair weights handed to the
  same model.
- pre-execution amendments (config `amendments`): amendment-1 corrected the
  scale-distinctness normalisation (mathematically impossible gate, threshold
  kept); amendment-2 documented the self-contained Walktrap implementation
  (networkx 3.4.2 API absence). No audit number or label existed before them.
