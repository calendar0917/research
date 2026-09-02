# KSVD 从零路线：当前证据、边界与下一决策

> 日期：2026-07-31
>
> 当前状态：**表示与单初始化 KSVD reconstruction 成立；marginal、pair 和 statistics-conditioned residual routes 均未建立稳定 downstream added value。普通无监督 task branch 已停止。**

## 1. 这条路线现在已经排除了哪些混淆

| 混淆 | 当前处理 | 结果 |
|---|---|---|
| 依赖很多随机初始化 | 每个独立数据集只用一次 deterministic maximin INIT | 不依赖 restart；U0-D 5/5 通过 |
| 人工 planted motif / `D_true` | 图由 degree-preserving global rewiring 生成，不给 atom vocabulary | KSVD 自己决定 12 个 latent basis |
| 强制 atom 可命名 | 不做 atom-to-motif gate；只审计使用、坍缩、重建和下游 | 允许不可解释 atom |
| canonical 保证同一性就等于线性坐标合理 | 单独做 U0-R geometry audit | canonical identity 正确但局部几何失败；改用 WALK-order signal |
| 输入根本没有任务信号 | KSVD 前先做 U0-P | WALK raw mean/std mean BA = 0.815，明确有信号 |
| FINAL 好是 initializer 已经好 | 全程 paired INIT vs FINAL | 重建增益明确；下游增益不够一致 |
| classifier 伪信号 | label shuffle + graph-code shuffle | means 0.491 / 0.475 |

## 2. 当前三层结论

### 层 A：representation 可用——通过

first-discovery-order adjacency 的坐标含义固定：`(i,j)` 表示第 i、j 个首次发现节点之间是否有 induced edge。它不是唯一 unlabeled-subgraph canonical form，但适合作为 rooted walk-induced patch signal。

U0-P：

- WALK mean/std：`0.815 ± 0.027`；
- edge-count histogram：`0.826 ± 0.012`；
- label shuffle：`0.501 ± 0.014`。

### 层 B：无监督 KSVD optimizer 可用——通过

U0-D：

- test relative reconstruction error 平均下降 `34.55%`；
- 5/5 replicate 下降至少 10%；
- FINAL 的 12 个 atoms 在每个 replicate 都不是 dead；
- effective atom count `8.56–9.51`；
- coherence `0.556–0.731`。

因此可以说：

> 单次初始化 KSVD 能从没有人工 atom 词表的图 patches 中学习非坍缩、可泛化的 sparse reconstruction basis。

### 层 C：KSVD update 对下游不可替代——未通过

U1A mean test BA：

| 方法 | mean |
|---|---:|
| simple graph stats | `0.935` |
| edge histogram | `0.826` |
| raw WALK mean/std | `0.815` |
| PCA-12 | `0.809` |
| FINAL KSVD | `0.750` |
| INIT | `0.713` |
| fixed Gaussian | `0.709` |

FINAL−INIT：`[-0.005, +0.065, +0.035, -0.010, +0.100]`，平均 `+0.037`，但只有 `3/5` 为正。因此未通过预注册的 4/5 一致性 gate。

这说明：

1. KSVD 的重建优化是真实且稳定的；
2. FINAL codes 也确实含预测信号；
3. 但“更好重建”没有稳定转化为“比同一 INIT 更好分类”；
4. raw/PCA/edge-count 更强，说明当前 task 主要依赖 KSVD sparse bottleneck 未完整保留的低阶统计。

## 3. 对“自发现 atom”的准确回答

现在不应问“12 个 atom 是否每个都像 triangle/star”。更准确的问题是：

- 它们是否被数据使用？——是；
- 是否严重重复或坍缩？——否；
- 是否跨数据 replicate 形成相似 basis？——FINAL matched-atom cosine mean `0.862`，有描述性稳定性；
- 是否改善未知 patches 的 sparse reconstruction？——是；
- 是否稳定改善当前下游任务？——尚未证明。

所以“自发现”已经在**统计 basis discovery**层得到支持，但还没有在**task-useful discovery**层得到完整支持。

## 4. 现在不要做什么

1. 不增加随机 restart 后挑最好；
2. 不因为平均 FINAL−INIT 为正就忽略 3/5 一致性失败；
3. 不继续在当前 test seeds 上调 `K/T/iterations/readout`；
4. 不把 continuous atom threshold 成图后凭视觉宣布 motif；
5. 不直接跳回 MolHIV/CIN；
6. 不同时修改 sampler、target、objective 和 readout，否则无法归因。

## 5. 下一步需要先做的科学决策

### 方向 A：把 KSVD 定位为无监督 sparse graph-patch compressor

如果研究目标是“能否从图局部信号自发现一个健康、稳定、可重建的 sparse basis”，当前 U0-D 已经给出清晰正证据。下一步应做 post-hoc atom usage/top-activating-patch 分析和更大规模计算审计，而不是强求每个 atom 服务于 LOW/HIGH label。

### 方向 B：继续要求 KSVD update 有稳定 downstream added value

则当前路线尚未通过。下一 protocol 必须使用新 seeds，并且一次只改变一层：

1. **target**：换成不直接等同于 rewiring level 的动力学/鲁棒性 target，并比较 `stats`、`stats+INIT`、`stats+FINAL`；
2. **readout**：检查当前 frequency/mean-abs/RMS 是否丢失 code co-occurrence 或 patch distribution 信息；
3. **objective**：若无监督 reconstruction 与 target 长期错位，再讨论 discriminative/task-aware dictionary，而不是仍叫普通 KSVD 自发现；
4. **sampler**：只有证据表明 walk patches 未覆盖目标尺度时才修改。

最保守的下一步不是立即运行更多实验，而是先冻结一个 **U1B incremental-utility protocol**：选择单一、预先定义的 graph property，使用 `simple stats` 作为强基线，只问 `stats+FINAL` 是否稳定超过 `stats+INIT`。

## 6. 当前一句话结论

> **KSVD 作为无人工 motif 词表的图 patch 稀疏重建器是可行的；作为能稳定产生额外下游价值的“结构原子发现器”，当前证据仍不足。**

## 11. IMDB-BINARY raw 迁移（R0-P）

完整 raw 1000 图的 exact-isomorphism audit 得到 537 个结构组，其中 44 个组含标签冲突、涉及 318 图；cleaned 的 493 图对应 label-consistent 去重后的任务，因此只作为 sensitivity，不替代 raw benchmark。

6-node WALK 直接迁移的 clique fraction 为 0.442、canonical effective signature count 为 9.76，原 gate 形式失败。只改变 patch size 到 7 后，clique fraction 降至 0.336、effective count 升至 18.93，WALK mean/std raw-stratified BA 为 0.619，label shuffle 为 0.504，R0-P 通过。graph stats 仍更强（0.700），且 stats+WALK 没有增量，因此下一步只执行冻结 `s=7,d=21,K=12,T=2` 的 paired INIT-vs-FINAL R0-D reconstruction audit。

详细协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_RAW_TRANSFER_ROUTE_20260731.md`。


## 12. IMDB-BINARY raw：R0-D dictionary audit

冻结的 `s=7,d=21,K=12,T=2,updates=25` 已在完整 raw IMDB-BINARY 上完成。每个 fold 只有一次 deterministic maximin INIT，没有 restart，也没有 K/T/iteration scan。

Split 审计：

- raw stratified：5 个 200-graph folds，类别均为 100/100；作为 benchmark view，允许 exact duplicate 跨 fold；
- raw exact-isomorphism-grouped：5 个 200-graph folds，类别均为 100/100，537 个 exact structure groups 完整分配，group leakage 为 0。

主结果：

| view | positive folds | mean INIT→FINAL graph-balanced reduction | nondead | max usage share |
|---|---:|---:|---:|---:|
| stratified | 5/5 | 47.87% | 每 fold 12/12 | 0.287 |
| grouped | 5/5 | 47.89% | 每 fold 12/12 | 0.270 |

描述性 controls 的 mean graph-balanced test error：

| view | Gaussian | INIT | medoid | FINAL | PCA-12 |
|---|---:|---:|---:|---:|---:|
| stratified | 0.8584 | 0.7025 | 0.4252 | 0.3657 | 0.1843 |
| grouped | 0.8584 | 0.6808 | 0.4017 | 0.3551 | 0.1843 |

R0-D 分类：**PASS_R0D_REAL_DICTIONARY_OPTIMIZATION**。

这把真实数据上的结论推进到：

> 单次初始化、无人工 atom vocabulary 的 KSVD，确实能在 raw IMDB 局部结构 patches 上自发现一个非坍缩、held-out reconstruction 更好的 sparse basis；而且该结论在去除 exact-isomorphism leakage 后仍成立。

但它仍没有回答 downstream added value。由于 R0-P 中 graph stats 已强于 WALK summaries，下一阶段只能做 R0-A 的条件归因：`STATS+FINAL` 对比 `STATS+INIT`，而不是只报 FINAL standalone accuracy。

详细结果：`tracks/ksvd/results/from_scratch/IMDB_BINARY_R0D_DICTIONARY_AUDIT_20260731.md`。


## 13. IMDB-BINARY raw：R0-A downstream incremental attribution

R0-A 已使用与 R0-P/R0-D 不同的新 outer split seed `731401` 完成。两个 raw views 各 5 folds；每 fold 单 deterministic INIT、0 restart，固定 36-D frequency/mean-abs/RMS graph-code readout，并使用 inner validation 只选择 L2 logistic regularization。

主结果：

| view | STATS | STATS+INIT | STATS+FINAL | update direction | mean FINAL−INIT | mean FINAL−STATS |
|---|---:|---:|---:|---:|---:|---:|
| stratified | 0.704 | 0.680 | 0.678 | 2/5 positive | -0.002 | -0.026 |
| exact-isomorphism-grouped | 0.623 | 0.624 | 0.630 | 3/5 nonnegative | +0.006 | +0.007 |

stratified primary gate 未通过。其正确对齐 `STATS+FINAL` 也没有优于 `STATS+SHUFFLED_FINAL`（0.678 vs 0.685）；label shuffle 为 0.477。grouped view 有很弱的正方向并按预注册 mechanism gate 通过，但不足以挽救 primary failure。注册总分类为 `FAIL_R0A_CONTROL_INTEGRITY`；post-hoc permutation audit 已确认 shuffle 真实执行，因此它表示 alignment control 的 substantive failure，而非实现没有打乱。

当前最准确的一句话结论更新为：

> **KSVD 作为无人工 motif 词表的真实图 patch sparse reconstruction basis learner 可行；但其 reconstruction update 通过当前最简单 bag-of-codes graph readout，并未在 raw IMDB-BINARY 上产生稳定、超出 INIT 与 graph statistics 的分类价值。**

这把失败位置缩小到了 `patch basis -> graph readout/task` 的桥梁。若继续，不能增加 restart 或扫描已看过的 outer tests；只能另开新 seed 的单轴 readout/relations protocol。

详细协议与结果：

- `tracks/ksvd/docs/KSVD_IMDB_BINARY_R0A_PROTOCOL_20260731.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0A_DOWNSTREAM_ATTRIBUTION_20260731.md`

## 14. IMDB direct n-hop sampler diagnosis

R0-A 后检查了“是否主要因为 WALK sampler”。不训练 KSVD，只比较固定 root budget 下的 direct capped n-hop selectors 与 exact rooted canonical output。

结果：radius-2 在 IMDB 上对所有 roots 都已覆盖整图；radius-1 大小高度可变，固定到 7 nodes 时 74%–86% cutoff 存在结构排序 ties。三种 n-hop selector 的 clique mass 为 0.559–0.642，effective topology count 仅 4.70–6.76，均明显差于 WALK s=7 的 0.336 与 18.93。n-hop standalone signal 最高 BA 0.647，但 `STATS+n-hop` 仍低于 STATS。

正式分类：`FAIL_DIRECT_CAPPED_NHOP_SAMPLER`。

所以当前瓶颈不能简化为“把随机游走换成 n-hop 即可”。更准确的是：IMDB 没有适合固定小 adjacency patch 的自然 n-hop 局部尺度，且无监督 reconstruction basis 与 graph labels、graph readout 之间存在目标错位。

完整讨论：`tracks/ksvd/docs/KSVD_ROUTE_BOTTLENECK_DISCUSSION_20260801.md`。


## 14. IMDB-BINARY raw：R0-B atom-pair readout audit

R0-B 使用新 outer seed `731501`，只在 36-D marginal readout 后加入 66-D within-patch atom-pair co-activation；dictionary、sampler、K/T/updates、单 deterministic INIT 和 classifier protocol 均未改变。

| view | marginal FINAL | marginal+pair FINAL | pair update mean | pair update direction | pair added over marginal |
|---|---:|---:|---:|---:|---:|
| stratified | 0.698 | 0.680 | +0.015 | 3/5 positive | -0.018 |
| exact-isomorphism-grouped | 0.669 | 0.648 | +0.001 | 2/5 nonnegative | -0.021 |

两个 view gates 都失败，注册分类为 `FAIL_R0B_PAIR_READOUT_UTILITY`。correct alignment 和 label shuffle controls 均正常，所以可以直接解释为：同 patch 内 atom support co-activation 没有补充 marginal readout，平均反而降低分类泛化。

当前一句话结论进一步更新为：

> **无人工 atom vocabulary、单初始化 KSVD 的真实图 patch basis discovery 与 sparse reconstruction 是成立的；但 reconstruction gain 尚未通过 marginal usage 或最小 atom-pair composition 转化为 raw IMDB 的稳定 downstream added value。**

若继续 task 路线，下一步不能再堆 bag statistics；需要在真正跨-patch relation 与 task-aware objective 之间做明确选择。前者仍保持无监督字典，但系统复杂度明显上升；后者直接改变研究命题，不再是普通 KSVD 的 task-optimal self-discovery。

详细结果：`tracks/ksvd/results/from_scratch/IMDB_BINARY_R0B_PAIR_READOUT_AUDIT_20260801.md`。


## 15. IMDB-BINARY raw：R0-X objective-alignment diagnosis

R0-X 直接复用 R0-D frozen dictionaries，不重新训练、不调参。结果显示：

- `STATS -> reconstruction_gain` held-out explained fraction：stratified `0.737`，grouped `0.672`；
- `STATS -> FINAL code` explained fraction 比 INIT 增加：stratified `0.350 -> 0.382`，grouped `0.359 -> 0.405`；
- 控制 STATS 后 FINAL residual label projection：stratified `-0.033`、方向 `2/5`；grouped `+0.080`、方向 `2/5`；
- reconstruction gain 与 edge-count bin mass 高相关：`0.894/0.850`，但 top-3 bins 的 gain share 约 `0.626/0.582`，暂不足以单独判为 frequency-reweighting route。

因此 R0-X 路由为 `PRIORITIZE_STATS_CONDITIONAL_OBJECTIVE`：当前 KSVD update 主要增强了 statistics-coupled reconstruction variation，而不是稳定的 residual label direction。下一步如果继续 task route，应在新 outer seed 下设计 statistics-conditioned/residual dictionary objective；不能继续堆 readout，也不能把这个诊断当成新的分类结果。

详细结果：`tracks/ksvd/results/from_scratch/IMDB_BINARY_R0X_ALIGNMENT_DIAGNOSIS_20260801.md`。


## 16. IMDB-BINARY raw：R0-C statistics-conditioned residual KSVD

R0-C 只修改 dictionary target，不使用 labels：先由 outer-train graph statistics 预测每图 patch mean，再对 residual patches 学 KSVD。

结果分层明确：

- residual held-out reconstruction：stratified/grouped 均 `5/5` 改善，mean reduction `0.362/0.372`；每 fold `12/12` atoms non-dead；
- downstream residual FINAL−INIT：stratified `-0.016`（2/5 positive），grouped `+0.036`（3/5 nonnegative）；
- residual FINAL−STATS：`-0.023/+0.011`；
- residual FINAL−standard FINAL：`+0.011/-0.015`；
- controls 正常。

注册分类：`FAIL_R0C_STATS_CONDITIONAL_UTILITY`。

这说明问题不是 residual basis 学不出来，而是即使明确去除 STATS-predictable patch mean，reconstruction improvement 仍不稳定对应 classification improvement。按照预注册停止规则，普通无监督 reconstruction-KSVD 的 raw IMDB task-utility branch 到此结束。

当前最终边界：

> **KSVD 能自发现健康、可泛化重建的图 patch sparse basis；但在 raw IMDB 上，没有证据表明 standard、pair-readout 或 statistics-conditioned residual reconstruction updates 会自动产生稳定的分类增量。**

后续只能选择 compressor/basis-discovery 定位，或明确进入 label-conditioned task-aware dictionary learning。

## 17. R1-A/R1-B：basis discovery 分支

R1-A 复用 R0-D dictionaries 做 label-free atom characterization。两个 raw views 均显示：FINAL `12/12` nondead、cross-fold matched cosine `0.753/0.746`、比 Gaussian nearest-real cosine 高 `0.250/0.244`，top-5 exemplars 也覆盖多个 WALK vectors 与 edge-density regimes。

R1-A 正式结果仍是 `FAIL_R1A_BASIS_CHARACTERIZATION`，因为预注册的 `FINAL nearest-real >= INIT nearest-real` 条件结构性失配：INIT 本身就是归一化真实 training patch，nearest cosine 恒为 1。不能事后把该 gate 删除，因此 R1-A 整体记为 `protocol-misspecified / inconclusive`，不是 basis evidence 全部失败。

R1-B 随后只做无 gate 的 grouped consensus atlas：12 个 latent components，reference-matched cosine mean `0.717`，平均每 atom 25 个 held-out exemplars 中有 `9.17` 个 unique canonical signatures，dominant canonical mass 平均 `0.327`。这支持“连续 latent basis component”这一表述，但不支持“12 个都对应清晰、稳定、可命名 motifs”。其中一个 atom 的 reference-fold match 最低仅 `0.102`，稳定性存在明显个体差异。

当前最准确的 compressor 结论：

> **KSVD 在 raw IMDB 上学习到了健康、可重建、整体具有跨 fold 对齐性的连续 patch basis；但这些 basis components 不是一组全部稳定且可人工命名的 graphlet vocabulary。**

这条分支不再回到 classification gate。后续若继续，只能另开 multi-view/attribute compressor，或者明确进入 label-conditioned task-aware dictionary learning，并改变研究命题。

## 18. IMDB-BINARY G0：per-graph dictionary descriptor

为区分导师式“每图 factorization statistics”与当前 shared dictionary vocabulary，G0 在新 outer seed `732101` 上固定同一套 WALK `s=7` patches，只把 dictionary scope 改为 per-graph。每图使用 `K=8,T=2,updates=10`，同时构造：

- permutation-invariant atom-set readout（主）；
- 导师式 ordered `D/X/Gram` readout（历史敏感性）；
- 同图 INIT、FINAL、rank-8 PCA；
- 强条件基线 `STATS+RAW`。

机制层：

- invariant readout 在同步 atom permutation 后 max difference `1.24e-14`；
- mean per-graph reconstruction error `0.1370 -> 0.0935`，5/5 fold means 改善；
- 但只在 `643/1000` graphs 上 FINAL 严格优于 INIT；
- PCA reconstruction 更低，为 `0.0663`。

任务层：

| view | STATS | STATS+RAW | +INIT | +FINAL | +PCA | update direction | mean FINAL−INIT |
|---|---:|---:|---:|---:|---:|---:|---:|
| stratified | 0.702 | 0.691 | 0.693 | 0.681 | 0.689 | 2/5 | -0.012 |
| grouped | 0.651 | 0.647 | 0.675 | 0.667 | 0.627 | 1/5 positive | -0.008 |

stratified aligned FINAL 也没有优于 graph-row-shuffled FINAL（mean difference `-0.005`）；label shuffle 为 `0.478`。legacy ordered readout 对随机 atom order 的 mean sensitivity 为 `0.019/0.031`，验证了原导师式拼接确实依赖没有严格对齐的 per-graph atom index。

注册分类：**RECON_ONLY_PERGRAPH_DICTIONARY**。

因此导师式 per-graph factorization 是一个数学上成立的 graph descriptor，但在当前 raw IMDB WALK substrate 上，KSVD updates 没有产生超出同一 INIT、RAW、STATS 的稳定分类增量。grouped view 的 descriptor 增量主要已存在于 INIT，而不是由 FINAL updates 创造。按照冻结协议，不继续在已看过 folds 上扫描 `K/T/updates/readout`，也不进入 dynamic ego 复刻。

协议与结果：

- `tracks/ksvd/docs/KSVD_IMDB_PERGRAPH_DICTIONARY_PROTOCOL_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_G0_PERGRAPH_DICTIONARY_AUDIT_20260801.md`
