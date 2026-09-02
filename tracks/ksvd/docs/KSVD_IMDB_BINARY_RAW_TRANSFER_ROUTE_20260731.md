# KSVD 从零路线：IMDB-BINARY raw 真实结构迁移协议

> 日期：2026-07-31  
> 状态：R0-P/R0-D reconstruction 通过；R0-A/R0-B/R0-C task utility 均未通过。普通无监督 reconstruction-KSVD 的 raw IMDB task branch 按协议停止。  
> 目标：检查无人工 atom 词表的 WALK-order sparse basis 能否迁移到真实、变长、纯结构图；不把 cleaned 分数冒充公开 raw benchmark。

## 1. 数据集角色必须分开

### 1.1 Raw benchmark view

- 数据：完整 IMDB-BINARY raw，1000 图，类别 500/500。
- 用途：和常见 IMDB-BINARY 数据版本保持一致，提供外部性能参考。
- 局限：常规随机划分允许相同结构的副本跨 train/test；因此不能单独承担机制结论。

### 1.2 Raw exact-isomorphism-grouped view

- 数据仍是完整 raw 1000 图，不删除样本。
- 同一个 exact-isomorphism group 的所有样本必须进入同一个 outer fold。
- 用途：判断模型能否泛化到未在训练中出现过的整体图结构。
- 含冲突标签的同构组也必须整体放入同一 fold，不能按标签拆开。

### 1.3 Cleaned sensitivity view

- 数据：TU/PyG cleaned IMDB-BINARY，493 图。
- 用途：检查去重且排除结构相同、标签冲突样本后的敏感性。
- 禁止：将 cleaned accuracy 与使用 raw 1000 图的论文结果直接比较。

## 2. R0-I：raw 数据同构与标签审计

本轮使用 WL/degrees 作为候选过滤，再在候选桶内运行 exact graph isomorphism；WL hash 本身不是最终同一性判据。

Raw 结果：

- 1000 图；
- 537 个 exact structure groups；
- 116 个 duplicate groups；
- 579 张图位于 duplicate groups；
- 44 个 structure groups 含相互冲突的 graph labels；
- 冲突组涉及 318 张图；
- 删除冲突组并在每个 label-consistent group 留一个代表后，恰有 493 个结构，类别代表数 261/232，与 cleaned 规模一致；
- 对 exact isomorphism 严格不变的纯结构确定性分类器，在 raw 样本上的组内多数标签经验 accuracy ceiling 为 `0.886`，至少 114 个样本无法同时判对。

因此 raw 与 cleaned 不是单纯“1000 样本 vs 少一些样本”：cleaned 同时改变了重复权重和标签噪声结构。

## 3. R0-P0：冻结的 6-node WALK 直接迁移

配置：

- raw 1000 图；
- first-discovery-order induced adjacency；
- patch size `s=6`，signal dimension `d=15`；
- 每图 `min(n,24)` 个不同根；
- sampling seed `20260731`；
- 不训练字典。

结果：

- 17,526 patches；
- clique fraction `0.4422`；
- missing-at-most-one-edge fraction `0.4501`；
- rooted-canonical unique signatures `128`；
- canonical effective signature count `9.7626`；
- dominant signature mass `0.4422`，即 6-clique；
- within-graph canonical unique fraction median `0.3846`；
- mapped-trajectory permutation mismatches `0/1746`。

Raw stratified 5-fold × 3 split-seed mean BA：

| Feature | Mean BA |
|---|---:|
| graph stats | 0.7003 |
| WALK mean/std | 0.6057 |
| canonical mean/std | 0.6300 |
| edge histogram | 0.6377 |
| stats + WALK | 0.6927 |
| WALK label shuffle | 0.5039 |

原冻结 gate 的 effective signature count 要求 `>=10`，实际为 `9.7626`，因此形式上失败。不能把 9.7626 事后改成通过；其连续解释是：**不是完全坍缩，但 6-node substrate 仍明显受 clique 主导，且有效结构容量与计划的 K=12 非常接近。**

## 4. R0-P1：只改变 patch size 的 7-node 修复探针

R0-P0 后只改变一项：

- `s: 6 -> 7`；
- `d: 15 -> 21`；
- sampler、根策略、patch budget、数据和分类器协议不变；
- 该探针是顺序探索，不与 P0 合并成 confirmatory evidence。

结果：

- 17,526 patches；
- clique fraction `0.3359`；
- dominant signature mass `0.3359`；
- canonical effective signature count `18.93`；
- WALK mean/std BA `0.6187`；
- canonical mean/std BA `0.6297`；
- edge histogram BA `0.6480`；
- graph stats BA `0.7003`；
- stats + WALK BA `0.6957`；
- WALK label shuffle BA `0.5044`。

解释：

1. `s=7` 明显降低 clique domination，并将有效 topology 数从约 9.76 提升到约 18.93；
2. patch summaries 的 standalone label signal 仍存在；
3. patch summaries 尚未提供超出 graph stats 的线性增量；
4. 第 3 点不能提前判定 KSVD 失败，但意味着 R0-D/R0-A 必须保留 `STATS` 条件基线；
5. 基于 substrate 而不是 KSVD 结果，现冻结 `s=7` 作为下一阶段唯一主表示，不再扫描 `s=8/9/...`。

## 5. R0-D：下一步唯一允许执行的 dictionary audit

### 5.1 冻结参数

- representation：7-node WALK first-discovery-order induced adjacency；
- `d=21`；
- `K=12`；
- `T=2`；
- KSVD updates `25`；
- 每图 `min(n,24)` patches；
- train-coordinate centering；
- 每个 outer fold 一个 deterministic maximin INIT；
- restart `0`；
- 不查看 atom 是否像人工 motif，不以可命名性作为 gate。

### 5.2 数据隔离

在每个 outer fold 中：

1. patch sampling 可以预先按冻结 seed 完成，因为它不使用标签和总体拟合统计；
2. centering 只在 outer-train patches 上拟合；
3. INIT 只由 outer-train patches 构造；
4. KSVD 只在 outer-train patches 上更新；
5. outer-test 只用于 held-out reconstruction/health evaluation；
6. PCA、medoid 等所有数据依赖控制也只能在同一 outer-train 上拟合。

### 5.3 两种 reconstruction view

先运行：

1. `raw/stratified/5-fold`：普通 raw feasibility；
2. `raw/exact-isomorphism-grouped/5-fold`：结构泛化审计。

Cleaned 不在 R0-D 第一轮执行，避免一次展开三条路径。只有 raw 两种 view 的实现和结果清楚后再补。

### 5.4 主比较

必须 paired 比较同一个 fold 的：

- `INIT` sparse reconstruction；
- `FINAL` sparse reconstruction。

同时保留：

- PCA-12 reconstruction；
- fixed Gaussian dictionary；
- medoid dictionary；
- train/test graph-balanced error；
- train/test patch-weighted error。

主归因量：

\[
\Delta_{recon}
=\frac{E_{INIT,test}-E_{FINAL,test}}{E_{INIT,test}}.
\]

### 5.5 健康指标

- non-dead atom count；
- atom usage distribution；
- effective atom count；
- maximum coherence；
- maximum single-atom usage share；
- train-to-test reconstruction gap；
- fold 间 matched atom cosine，只作为描述性稳定性，不作为挑选 fold/restart 的依据。

### 5.6 冻结 gate

R0-D 通过要求：

1. graph-balanced held-out reconstruction relative reduction 至少在 `4/5` folds 为正；
2. 5-fold mean relative reduction `>=10%`；
3. 每 fold 至少 `10/12` non-dead atoms；
4. 无 fold 的单 atom usage share `>0.60`；
5. grouped view 不得出现与 stratified view 方向相反的系统性失败。

如果失败：停止，不增加 restart，不扫描 K/T 来修复本轮结果。

## 6. R0-A：已冻结的下游增量归因

R0-A 已在运行结果不可见前单独冻结，完整协议见：

- `tracks/ksvd/docs/KSVD_IMDB_BINARY_R0A_PROTOCOL_20260731.md`

最小设计为：

- raw/stratified 与 raw/exact-isomorphism-grouped 各 5 outer folds；
- 新 outer split seed `731401`，不扫描额外 seeds；
- 每 fold 仍只有一个 deterministic INIT、0 restart；
- 固定 36-D graph-code readout：每 atom 的 activation frequency、mean-absolute coefficient、RMS coefficient；
- 主归因 `BA(STATS+FINAL)-BA(STATS+INIT)`；
- 同时要求 FINAL 不低于 STATS，并通过 conditional graph-code shuffle 与 label shuffle；
- grouped inner validation split 继续保持 exact-isomorphism groups 完整；
- 第一轮不运行 cleaned，不增加 attention、MIL、co-occurrence 或 patch relations。

R0-A 是机制归因而不是 leaderboard protocol；只有通过后才另行冻结公开方法比较方案。

## 7. R0-D 执行结果

R0-D 已按冻结协议完成。两个 raw view 都使用一次 deterministic maximin INIT、零 restart，并在 outer train 内完成 centering、初始化、KSVD、PCA 与 medoid 拟合。

### 7.1 Split 审计

- stratified：每 fold 200 图、类别 100/100；它作为普通 benchmark view，不约束 exact-isomorphism leakage，五折累计有 300 个 fold-group leakage events；
- exact-isomorphism-grouped：每 fold 200 图、类别 100/100、结构组数 104–109；全部同构组保持完整，group leakage 为 0；
- 44 个 label-conflict groups 也没有被拆开。

### 7.2 主结果

| view | positive folds | mean graph-balanced reduction | min nondead | max usage share | gate |
|---|---:|---:|---:|---:|---|
| raw/stratified | 5/5 | 0.4787 | 12/12 | 0.2871 | PASS |
| raw/exact-isomorphism-grouped | 5/5 | 0.4789 | 12/12 | 0.2697 | PASS |

FINAL 的 mean graph-balanced test error：

- stratified：INIT `0.7025`，medoid `0.4252`，FINAL `0.3657`，PCA-12 `0.1843`；
- grouped：INIT `0.6808`，medoid `0.4017`，FINAL `0.3551`，PCA-12 `0.1843`。

因此 R0-D 正式分类为：

> **PASS_R0D_REAL_DICTIONARY_OPTIMIZATION**

它支持的结论是：即使不要求 atom 可命名、不给人工 motif 词表、也不依赖多次随机初始化，KSVD update 仍能在真实 raw IMDB patches 上学到健康且对未见 patches 泛化的 sparse basis。grouped view 与 stratified view 几乎同方向，说明这个 reconstruction gain 不是简单依赖 exact duplicate 泄漏。

但控制也给出两个重要边界：

1. PCA-12 明显低于 T=2 sparse methods，这是稀疏约束与非稀疏 rank-12 reconstruction 的容量差异，不能解释为 KSVD 应击败 PCA；
2. medoid 已明显强于 maximin INIT，FINAL 又进一步优于 medoid，但这些 reconstruction 排名仍不等价于 downstream usefulness。

## 8. 当前结论与下一步

现在可以说：

> 完整 raw IMDB-BINARY 上，s=7 WALK substrate、单初始化 KSVD optimization 与 dictionary health 均已通过；下一步可以进入 R0-A，但主问题必须是 `STATS+FINAL` 是否稳定超过 `STATS+INIT`。

现在还不能说：

- KSVD 在 IMDB 分类上优于其他方法；
- KSVD 提供了超出 graph statistics 的信息；
- cleaned 结果可以替代 raw benchmark；
- learned atoms 是可命名 graph motifs；
- reconstruction gain 必然转化为 classification gain。

## 9. 对应产物

代码：

- `tracks/ksvd/code/imdb_walk_substrate.py`
- `tracks/ksvd/code/run_imdb_binary_r0p_audit.py`
- `tracks/ksvd/code/test_imdb_walk_substrate.py`
- `tracks/ksvd/code/imdb_walk_dictionary.py`
- `tracks/ksvd/code/run_imdb_binary_r0d_dictionary_audit.py`
- `tracks/ksvd/code/test_imdb_walk_dictionary.py`

结果：

- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0P_WALK_AUDIT_20260731.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0P_WALK_S7_PROBE_20260731.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0D_DICTIONARY_AUDIT_20260731.md`
- 对应 JSON 文件。


## 10. R0-A 执行结果：简单 graph-code readout 未建立稳定下游增量

R0-A 使用新 outer split seed `731401`，两个 raw views 各 5 folds；每个 fold 仍只有一个 deterministic INIT、0 restart。dictionary、centering、PCA、medoid 均只用 outer train，inner validation 只选择 L2 logistic regularization。graph code 固定为 36 维：每 atom 的 activation frequency、mean absolute coefficient 与 RMS。

### 10.1 Primary attribution

| view | STATS mean BA | STATS+INIT | STATS+FINAL | positive/nonnegative update folds | mean FINAL−INIT | mean FINAL−STATS |
|---|---:|---:|---:|---:|---:|---:|
| raw/stratified | 0.704 | 0.680 | 0.678 | 2/5 positive | -0.002 | -0.026 |
| raw/exact-isomorphism-grouped | 0.623 | 0.624 | 0.630 | 3/5 nonnegative | +0.006 | +0.007 |

stratified reference view 明确失败：FINAL update 不稳定超过 INIT，也没有补充 STATS。grouped view 按预先冻结的较弱 mechanism gate 形式通过，但增量仅 `+0.006`，不能推翻 primary view 的失败，也不能单独支持路线成功。

### 10.2 Controls

- stratified `STATS+SHUFFLED_FINAL` mean BA `0.685`，高于正确对齐 FINAL 的 `0.678`；
- grouped 正确对齐 FINAL 比 shuffle 高 `0.021`；
- label shuffle mean BA：stratified `0.477`，grouped `0.485`；
- permutation post-hoc implementation audit 表明 shuffle 是 split-local、multiset-preserving 的真实 permutation，fixed points 仅 `0..4`，不是 identity。

注册 classification 为：

> **FAIL_R0A_CONTROL_INTEGRITY**

这个名称表示 registered alignment control 未通过，不表示 permutation 代码无效。更重要的是，即使忽略 shuffle control，stratified 的主 attribution 仍因 `2/5`、mean update `-0.002`、beyond-STATS `-0.026` 而失败。

### 10.3 当前路线结论

现在证据清楚地分成两层：

1. **通过**：KSVD 是健康的真实图 patch sparse reconstruction basis learner；FINAL 在两个 raw views 都稳定改善 held-out reconstruction；
2. **未通过**：固定的最简单 bag-of-codes readout 下，KSVD updates 没有稳定转化为超出 INIT/STATS 的 IMDB classification value。

因此不能继续宣称“reconstruction 更好，所以发现了对下游最有用的 atoms”。但也不能把结果解释成 KSVD 完全不可行：失败点已被定位在 **patch objective 到 graph-level task readout 的桥梁**，而不是 dictionary optimization 或人工 atom vocabulary。

下一步不得在相同 outer tests 上增加 restarts、扫描 K/T/iterations 或挑 readout。若继续，只能新开协议、使用新 outer split seed，并一次只修一个轴。基于 `luyin11` 关于 patch 间内在关联尚未解决的提醒，最合理的单轴候选是 readout/relations；但在执行前必须先说明它试图恢复哪一种被 frequency/mean-abs/RMS 丢失的信息，并保留 `STATS+INIT` vs `STATS+FINAL` 条件归因。

详细结果：`tracks/ksvd/results/from_scratch/IMDB_BINARY_R0A_DOWNSTREAM_ATTRIBUTION_20260731.md`。

## 11. 2026-08-01 direct n-hop sampler audit

为检查 R0-A 失败是否主要来自随机游走，执行了不训练 KSVD 的 matched n-hop feasibility audit。结论为：

> **FAIL_DIRECT_CAPPED_NHOP_SAMPLER**

关键事实：

- radius-1 ego 只有 13.81% roots 恰为 7 nodes；54.64% 大于 7，必须截断；
- radius-2 ego 对 100% roots 已等于整张图，不再是 local patch；
- 三种 capped n-hop selectors 的 cutoff tie fraction 为 0.740–0.859；多种排序不能严格保证 selection invariance；
- local-signature ordering + exact rooted canonicalization 的输出 relabel match 达 0.995，但 selected abstract set match 只有 0.344；
- n-hop clique mass 为 0.559–0.642，显著高于 WALK 的 0.336；
- n-hop effective canonical count 只有 4.70–6.76，低于 WALK 的 18.93；
- n-hop standalone BA 最高 0.647，比 WALK canonical 0.630 只高 0.017；加入 STATS 后仍低于 STATS-only。

因此不能把当前失败简单归因于“随机游走采样不好”。在 IMDB 上，WALK 反而帮助离开直接 clique neighborhood；最直接的 fixed-7 n-hop 会更严重地坍缩到 clique。

协议与结果：

- `tracks/ksvd/docs/KSVD_IMDB_NHOP_SAMPLER_AUDIT_PROTOCOL_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_NHOP_SAMPLER_AUDIT_20260801.md`

路线瓶颈讨论：`tracks/ksvd/docs/KSVD_ROUTE_BOTTLENECK_DISCUSSION_20260801.md`。


## 11. R0-B 执行结果：within-patch atom-pair readout 仍未建立稳定桥梁

R0-B 只改变 graph readout：在 R0-A 的 36-D marginal code 后增加 66-D atom-pair co-activation。pair 由 `T=2` sparse support 自然定义，不人工指定 motif pair；其余 `s=7,d=21,K=12,T=2,updates=25`、单 deterministic INIT、0 restart 全部不变。正式运行使用新 outer seed `731501`。

### 11.1 Primary attribution

| view | STATS | marginal INIT | marginal FINAL | marginal+pair INIT | marginal+pair FINAL | pair update direction | mean pair update | pair added over marginal FINAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw/stratified | 0.707 | 0.686 | 0.698 | 0.665 | 0.680 | 3/5 positive | +0.015 | -0.018 |
| raw/exact-isomorphism-grouped | 0.631 | 0.663 | 0.669 | 0.647 | 0.648 | 2/5 nonnegative | +0.001 | -0.021 |

两个 registered view gates 都失败，总分类：

> **FAIL_R0B_PAIR_READOUT_UTILITY**

controls 正常：correctly aligned FINAL pair representation 比 whole-code shuffle 分别高 `+0.023/+0.050`，label shuffle 为 `0.529/0.495`。因此这里没有 R0-A classification 名称带来的实现歧义；失败来自 pair update 的 fold 一致性不足，而且 pair features 平均降低 marginal FINAL 的泛化表现。

Post-hoc failure-shape audit 也显示更像过拟合而不是缺少容量：stratified FINAL marginal 的 mean train/test BA 为 `0.742/0.698`，加入 pair 后变为 `0.773/0.680`；grouped 从 `0.708/0.669` 变为 `0.722/0.648`。所以继续增加 bag-level quantiles、signed pairs 或更高阶组合，没有当前证据支持。

### 11.2 对路线的进一步收缩

R0-B 排除的是一个具体且最小的解释：

> R0-A 失败主要只是因为 marginal pooling 忘记了同一个 patch 内哪两个 atoms 共同激活。

当前结果不支持这个解释。它不排除真正的跨-patch overlap、距离或顺序关系，但意味着不能继续在相同 outer tests 上堆 quantile、pair sign、pair threshold 或更多 bag statistics。

现在路线的层级证据为：

```text
WALK substrate                         PASS
single-init KSVD optimization          PASS
held-out sparse patch reconstruction   PASS
marginal graph-code task bridge        FAIL
within-patch pair task bridge           FAIL
```

如果目标仍是普通无监督 KSVD 的可行性，当前最稳妥的定位是 **sparse graph-patch compressor / basis discovery**。如果目标必须包含 classification added value，下一步已不再是一个“小 readout 修补”；必须在新协议中明确选择：

1. 表达真正的跨-patch relation（需要保留 patch roots/trajectories/overlap，复杂度显著上升）；或
2. 承认 reconstruction objective 与 task 可能错位，研究 task-aware/conditional objective，此时不能再把结果称为普通无监督 KSVD 自动发现 task-optimal atoms。

详细协议与结果：

- `tracks/ksvd/docs/KSVD_IMDB_BINARY_R0B_PAIR_READOUT_PROTOCOL_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0B_PAIR_READOUT_AUDIT_20260801.md`


## 12. R0-X 执行结果：重建收益主要与 statistics-coupled variation 一起变化

R0-X 不重新训练字典，直接复用 R0-D 保存的 INIT/FINAL dictionaries、outer folds 和 train coordinate means。它检查 `STATS -> code/gain` 的 held-out explained fraction、控制 STATS 后的 label-effect consistency，以及 patch-frequency allocation。

### 12.1 关键结果

| view | INIT code R2 | FINAL code R2 | reconstruction gain R2 | residual FINAL−INIT projection | residual direction | gain Cohen d | top3 mass/gain share |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw/stratified | 0.350 | 0.382 | 0.737 | -0.033 | 2/5 | 0.132 | 0.656/0.626 |
| raw/exact-isomorphism-grouped | 0.359 | 0.405 | 0.672 | +0.080 | 2/5 | 0.199 | 0.654/0.582 |

`STATS -> reconstruction_gain` 的 explained fraction 很高，说明 KSVD update 取得的 reconstruction benefit 大量跟随 graph statistics/global nuisance variation。FINAL code 对 STATS 的 explained fraction 也比 INIT 更高，但 residual label projection 没有通过预注册的一致性条件。

Patch edge-count bin mass 与 positive reconstruction gain 的相关系数为 stratified `0.894`、grouped `0.850`；top-3 bins 约占 `0.65` patches，贡献约 `0.58–0.63` 的 positive gain。这是 frequency coupling 的证据，但未达到预注册 frequency-reweighting route 的 threshold，因此暂列为次要现象。

### 12.2 路线收缩

R0-X 路由为：

> **PRIORITIZE_STATS_CONDITIONAL_OBJECTIVE**

下一步不应再增加 marginal/pair/quantile readout。更有信息量的单轴修复是：让 dictionary objective 不再优先重建已经能由 graph statistics 解释的 variation，例如在 outer-train 内拟合 statistics-conditioned patch expectation，再对 residual patch signal 学 dictionary。这个方向需要新 outer seed 和全新的 protocol；不能用当前 R0-X 的诊断量宣称 conditional KSVD 已经有效。

如果 conditional objective 仍没有 residual task utility，下一分叉才是：

1. **停止 task claim**，把 KSVD 定位为 sparse graph-patch compressor；或
2. **进入 task-aware dictionary learning**，明确使用 labels，且不再把它称为普通无监督 KSVD 自发现 task-optimal atoms。

详细协议与结果：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0X_OBJECTIVE_ALIGNMENT_PROTOCOL_20260801.md`。


## 13. R0-C 执行结果：conditional residual reconstruction 成功，但 task utility 仍失败

R0-C 使用新 outer seed `731601`，只改变 dictionary target：outer-train graph-balanced least squares 用 12-D graph statistics 预测每图 patch-coordinate mean，KSVD 对 residual patches 学习。STANDARD 与 CONDITIONAL branches 使用相同 `K=12,T=2,updates=25`、单 deterministic INIT、0 restart 和 36-D marginal readout。

### 13.1 Dictionary 层

| view | residual recon positive folds | mean held-out reduction | min nondead | max usage share |
|---|---:|---:|---:|---:|
| raw/stratified | 5/5 | 0.362 | 12/12 | 0.282 |
| raw/exact-isomorphism-grouped | 5/5 | 0.372 | 12/12 | 0.280 |

Residual dictionary optimization 明确通过。因此 conditional target 是可学习的，且不导致 atom collapse。

### 13.2 Task 层

| view | STATS | standard FINAL | residual INIT | residual FINAL | FINAL−INIT | FINAL−STATS | residual−standard |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw/stratified | 0.715 | 0.681 | 0.708 | 0.692 | -0.016 | -0.023 | +0.011 |
| raw/exact-isomorphism-grouped | 0.630 | 0.656 | 0.605 | 0.641 | +0.036 | +0.011 | -0.015 |

Stratified primary gate 失败：只有 `2/5` positive updates，平均 update 和 beyond-STATS 都为负。Grouped 虽有 `3/5` 非负和 `+0.036` mean update，但平均低于同 fold standard FINAL，且正均值明显受 fold 0 的 `+0.225` 影响，因此 grouped gate 也失败。

Full-patch FINAL reconstruction 中，standard 与 residual branches 几乎相同，conditional 略差：stratified `0.1732 vs 0.1761`，grouped `0.1740 vs 0.1767`。Residualization 改善了 INIT 起点，但没有让 FINAL 获得更优的 full reconstruction 或稳定 task residual。

总分类：

> **FAIL_R0C_STATS_CONDITIONAL_UTILITY**

### 13.3 路线终点

到这里已经顺序检查：

```text
substrate signal                         PASS
single-init standard reconstruction      PASS
marginal readout task bridge             FAIL
within-patch pair readout                FAIL
objective-alignment diagnosis            STATS-coupled
statistics-conditioned reconstruction    PASS
statistics-conditioned task bridge       FAIL
```

因此 raw IMDB 上继续修改普通无监督 reconstruction KSVD 的投入回报已经很低。不能再通过扫描 residualizer、frequency weights、K/T/restarts 或更多 readout 来挽救已看过的结果。

现在有两个诚实的研究定位：

1. **保留普通 KSVD**：定位为 label-free sparse graph-patch compressor / basis discovery，研究稳定性、覆盖、top-activating real patches、扰动鲁棒性和属性多视图重建；
2. **分类必须成功**：另开 task-aware dictionary learning，明确 labels 进入 objective，并将命题改为 task-conditioned discovery，不再声称普通 reconstruction KSVD 会自动发现 task-optimal atoms。

详细结果：`tracks/ksvd/results/from_scratch/IMDB_BINARY_R0C_STATS_CONDITIONAL_AUDIT_20260801.md`。

## 14. R1-A 执行结果：basis characterization protocol 发现结构性 gate 失配

R1-A 复用 R0-D 保存的 dictionaries，不重新训练、不使用 labels、不重新打开 task branch。两个 views 的其余 evidence 都一致：

| view | FINAL nondead | INIT cross-fold cosine | FINAL cross-fold cosine | FINAL vs Gaussian nearest cosine margin | top-5 unique WALK | edge-regime dispersion |
|---|---:|---:|---:|---:|---:|---:|
| raw/stratified | 12/12 | 0.536 | 0.753 | +0.250 | 2.50 | 1.86 |
| raw/grouped | 12/12 | 0.561 | 0.746 | +0.244 | 2.45 | 2.12 |

正式 R1-A classification 为 `FAIL_R1A_BASIS_CHARACTERIZATION`，但 post-hoc audit 发现唯一失败条件是 protocol bug：INIT 是 deterministic maximin 从真实 outer-train centered patch 中选出的、再归一化的 atom，因此 INIT nearest-real cosine 恒为 `1.0`；要求 FINAL 不低于 INIT 不适合比较 continuous learned basis。

这个结果不能事后改判为 PASS，也不能解释成 basis characterization 全部失败。准确表述是：

> R1-A overall characterization **protocol-misspecified / inconclusive**；但 FINAL 的 cross-fold stability、Gaussian proximity、top-patch diversity 与 density-regime coverage 均有正面 descriptive evidence。

## 15. R1-B：grouped consensus basis atlas

为避免继续制造事后 gate，R1-B 只做 descriptive atlas，无 PASS/FAIL。固定 grouped fold 0 为 reference，用 exact maximum absolute cosine assignment 对齐其余 folds；每个 fold 每个 atom 取 held-out top-5 absolute activation，合并为每个 consensus atom 的 25 个 exemplars。

结果摘要：

- 12 个 latent basis components；
- reference-matched cosine mean `0.717`，最弱 atom/fold 为 `0.102`；
- 每 atom 平均 `9.17` 个 unique canonical signatures / 25 exemplars；
- 平均 canonical effective count `6.66`；
- 平均 dominant canonical mass `0.327`；
- atom exemplar edge-count mean 范围 `9.96..14.16`。

这说明 learned atoms 不是每个都对应单一合法 graphlet；更准确地说，它们是连续 latent basis components，各自覆盖若干相近但不完全相同的 local structure families。Atom 7 的 cross-fold match 明显较弱，说明不能把 12 个 atom 都当作同等稳定的“发现对象”。

因此当前 compressor 分支的最稳妥结论是：KSVD 学到了健康且有一定跨 fold 对齐性的连续 patch basis，但不是 12 个清晰、稳定、可命名的人工 motif vocabulary。详细 atlas：

- `tracks/ksvd/docs/KSVD_IMDB_BINARY_R1B_CONSENSUS_BASIS_ATLAS_PROTOCOL_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R1B_CONSENSUS_BASIS_ATLAS_20260801.md`
