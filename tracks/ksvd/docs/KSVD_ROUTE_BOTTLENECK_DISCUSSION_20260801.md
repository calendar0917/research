# KSVD 从零路线：当前瓶颈与路线决策

> 日期：2026-08-01  
> 依据：U0-R/U0-D/U1-A、IMDB R0-P/R0-D/R0-A 与 direct n-hop sampler audit。  
> 目的：区分“某个实验没调好”与“研究命题本身走到分叉口”。

## 1. 现在确实到了瓶颈，但不是 KSVD 优化器瓶颈

当前证据已经排除了几个早期疑问：

- 不需要大量随机 restart：单 deterministic INIT 足以完成真实数据 R0-D；
- 不需要人工指定 atom vocabulary：FINAL basis 可以自发现并保持非坍缩；
- atom 不必都是合法、可命名图：连续 basis 仍能稳定改善 held-out sparse reconstruction；
- KSVD update 本身不是没有工作：IMDB 两个 raw views 的 INIT→FINAL reconstruction reduction 都约 48%。

真正没有建立的是：

> reconstruction-optimal patch basis 是否也是 task-useful graph representation。

R0-A 的 stratified `STATS+FINAL - STATS+INIT` 平均为 `-0.002`，`STATS+FINAL - STATS` 为 `-0.026`。这不是多跑几次初始化就能自然解决的问题。

## 2. n-hop audit 排除了“只因为随机游走”这一简单解释

在 raw IMDB-BINARY：

- radius-1 ego size median 8，54.64% roots 大于 7，31.56% 小于 7，只有 13.81% 恰为 7；
- radius-2 ego 对 100% roots 已经等于整张图；
- 因而 `2-hop` 在该数据上不是更宽的 local patch，而是 whole-graph representation；
- capped radius-1 必须大量截断，并在 74%–86% roots 的 cutoff 处遇到 structural ranking ties。

更重要的是，三种 capped n-hop selector 都比 WALK 更 clique-dominated：

| sampler | clique/dominant mass | effective canonical count | within-graph unique median |
|---|---:|---:|---:|
| WALK s=7 | 0.3359 | 18.9290 | 0.5000 |
| n-hop ID | 0.5589 | 6.7553 | 0.2143 |
| n-hop degree | 0.6221 | 4.8278 | 0.1667 |
| n-hop signature | 0.6418 | 4.7002 | 0.1538 |

这说明 WALK 在 IMDB 上并非只带来噪声。它还有一个实际作用：沿轨迹离开 root 的直接 clique neighborhood，从而获得比“root 加六个近邻”更丰富的 patch substrate。

n-hop standalone BA 有小幅提高（最高从 WALK canonical 的 0.6297 到 0.6467），但仍未补充 STATS；最佳 conditional gain 仍为负。直接 capped n-hop 因此没有解决 R0-A 的核心问题。

## 3. 多种排序不能从根本上解决邻接不变性

实际重编号审计：

| selector | selected set match | ranked vector match | rooted canonical output match |
|---|---:|---:|---:|
| ID | 0.1741 | 0.6394 | 0.8396 |
| degree | 0.3175 | 0.9611 | 0.9698 |
| local signature | 0.3440 | 0.9903 | 0.9952 |

结论分三层：

1. richer structural ordering 可以明显减少 coordinate noise；
2. exact rooted canonicalization 可以进一步使输出接近 invariant；
3. 但 selector 的抽象节点集合仍只有约 34% 完全一致，因为 cutoff ties 没有消失。

所以“BFS、degree、几种排序拼起来”只能是 robustness feature，不能作为严格保证。严格保证需要：

- 不截断完整 ego，并接受 variable/high dimension；或
- 在 cutoff tie 上包含完整 orbit/全部等价选择；或
- 使用真正的 full-ego canonical labeling 后再定义 invariant quotient；或
- 放弃 adjacency slots，改用 permutation-invariant structural statistics。

这些选择都会改变当前 `21-D adjacency + vanilla KSVD` 的方法定义。

## 4. 当前瓶颈有四层

### 4.1 数据局部尺度不匹配

IMDB social graphs 高聚集、直径几乎都不超过 2：

- 1-hop 太 clique-dominated；
- 2-hop 直接变成整图；
- 固定 7-node local patch 没有自然的 n-hop scale。

这不是 sampler 小修能完全解决的。

### 4.2 无监督目标与分类目标错位

KSVD 最小化平均 patch reconstruction。高频 clique/dense patterns 对 reconstruction 很重要，但未必区分类别。自发现的含义只是“从输入分布自发现 basis”，不保证“自动发现对任意下游最有用的 basis”。

若要保证 task relevance，需要 supervised/discriminative objective；但那将不再是普通无监督 KSVD 的原命题。

### 4.3 patch 到 graph 的信息桥梁不足

当前 frequency/mean-abs/RMS readout 只保留 atom marginals，不保留：

- 同一 patch 中的 atom pair composition；
- 不同 roots 的 occurrence relations；
- atom occurrence 在原图中的距离、重叠和连接方式。

这与 `luyin11` 的“patch 是片段，片段间关联尚未解决”完全一致。

### 4.4 方法身份开始稀释

如果继续同时加入：

- canonical full ego；
- multiscale sampler；
- relation graph；
- attention/MIL；
- task-aware dictionary；

最终性能即使提高，也很难判断核心贡献还是不是 KSVD。路线会从一个清晰可证伪的命题，变成不断叠加模块的模型搜索。

## 5. 现在有三条诚实的路线

### 路线 A：把 KSVD 定位为无监督 sparse patch compressor

主张缩小为：

> 无人工 motif vocabulary 的 KSVD 能从真实图局部信号中学习健康、稳定、对 unseen patches reconstruction 更好的 sparse basis。

这条路线已有正证据。后续应评价：

- compression/reconstruction trade-off；
- atom usage、top-activating real patches；
- 跨 fold/data stability；
- out-of-distribution reconstruction；
- 计算成本和字典覆盖率。

优点是结论清晰；缺点是不能把它包装成分类模型突破。

### 路线 B：再做一次 terminal graph-readout falsification

保持 WALK、K/T、INIT、KSVD objective 全部不变，只修改 graph readout。最小候选是：

1. 同一 patch 内 `T=2` atom-pair co-activation（66 维）；
2. 若仍失败，再不继续扩大 readout 搜索，结束普通 KSVD downstream claim。

必须使用新 outer split seed，并继续比较：

```text
STATS+PAIR_INIT vs STATS+PAIR_FINAL
```

这只能回答 marginal pooling 是否丢失了 atom composition，尚不能完整解决 patch-to-patch graph relations。

### 路线 C：承认需要 relational/task-aware extension

将研究对象改为：

> sparse patch basis + occurrence relation model

或：

> discriminative graph dictionary learning

此时要重新定义方法和 baseline，普通 KSVD 只作为初始化器/regularizer/compressor，不再把所有下游收益归功于 KSVD atom discovery。

## 6. 建议

当前最稳妥的选择是：

1. 将路线 A 作为已经成立的核心结论；
2. 若仍希望验证 task utility，只允许路线 B 的 **一次 terminal pair-readout experiment**；
3. 若 terminal experiment 仍失败，停止在 IMDB 上继续调 sampler/readout；
4. 是否进入路线 C，应作为一个新的研究问题，而不是当前路线的补丁。

因此“瓶颈”不是坏事。到这里我们已经知道：

> KSVD 能学 basis；困难在于局部 reconstruction basis 与 graph task 之间不存在自动等价关系。

这个边界本身比继续堆实验更有研究价值。
