# KSVD 从零路线：IMDB-BINARY n-hop sampler feasibility audit

> 日期：2026-08-01  
> 状态：在 n-hop signal/KSVD 结果不可见前冻结  
> 性质：R0-A 失败后的 sampler 诊断；本轮不训练 KSVD，也不选择“最好排序”进入正式测试。

## 1. 要拆开的两个问题

当前 WALK patch 同时包含两项设计：

1. 随机游走决定 7 个节点；
2. first-discovery order 决定邻接向量坐标。

直接将结果差异归因于“随机游走不好”是不充分的，因为换成 n-hop 后仍要解决：

- n-hop ego 大小可变；
- 大于 7 个节点时如何截断；
- 小于 7 个节点时如何扩展或 padding；
- cutoff shell 中结构等价节点如何选择；
- 选定节点后如何形成 permutation-invariant adjacency vector。

本轮先检查这些基础条件，不直接重新训练字典。

## 2. 为什么“多种排序”不能保证不变性

BFS、degree 和 local-signature 排序都可能出现 ties。若最终用 node ID 打破 tie，整体节点重编号后可能选择不同节点或改变坐标。因此：

> 多种排序只能做 sensitivity audit；把几种排序拼接起来也不构成严格 permutation invariance。

本轮对每一种 selector 都同时报告：

1. selected abstract node set 在整体重编号后是否一致；
2. raw ranked-order adjacency 是否一致；
3. 对选定 7-node patch 做 **exact rooted canonicalization** 后是否一致；
4. cutoff 是否落在 invariant ranking tie 中。

只有第 3 项验证输出表示；第 1、4 项用于判断 sampler 本身是否依赖任意 tie-break。

## 3. 数据与 patch budget

```text
dataset = raw IMDB-BINARY, 1000 graphs
patch size = 7
roots/graph = min(n_nodes, 24)
root sampling seed = 20260731
relabel audit = first 100 graphs, 3 permutations/graph
relabel seed = 20260801
```

根集合沿用 WALK 路线的冻结 root budget。根本身在重编号审计中同步映射，不把独立随机 root realization 混入 selector 审计。

## 4. Direct n-hop feasibility

对 raw 数据的全部节点统计 radius-1 与 radius-2 ego size：

- min/median/mean/quantiles/max；
- `size < 7`、`size = 7`、`size > 7` 比例；
- radius-2 ego 是否已经等于整张图。

如果 radius-2 在大部分图上等于全图，则它不是“更宽一点的 local patch”，而是 graph-level representation；不能无说明地替换 WALK patch。

## 5. 三种固定 7-node n-hop selectors

所有 selector 先按 root shortest-path distance 扩展，取 root 加排名最前的 6 个节点，因此 patch 连通。最终 node ID 只作为确定性 tie-break；该依赖必须由 relabel audit 暴露，不能被描述成 invariant。

### N-ID

```text
key = (distance, node_id)
```

等价于最直接的 capped BFS shell。它是故意保留的 negative ordering control。

### N-DEG

```text
key = (distance, -global_degree, node_id)
```

测试简单结构排序能否减少 tie dependence。

### N-SIG

```text
key = (
  distance,
  -common_neighbors_with_root,
  -global_degree,
  -sum_neighbor_degrees,
  node_id
)
```

前三项（不含 node ID）是可观测、重编号不变的 local signature。若 cutoff 仍 tie，则该 root 没有被这些统计唯一决定。

## 6. 邻接 vectorization

每种 selector 保留两种 21-D adjacency：

1. `ranked-order`：root 为 slot 0，其余为 selector rank；
2. `rooted-canonical`：固定 root，枚举其余 `6!` 排列，取 lexicographic minimum。

Signal exposure 只使用 rooted-canonical 版本，避免把 node-ID coordinate noise 当成 sampler signal。ranked-order 只用于不变性诊断。

## 7. Substrate 与 signal checks

每种 selector 报告：

- clique fraction；
- dominant canonical signature mass；
- canonical effective signature count；
- within-graph canonical unique fraction；
- cutoff tie fraction；
- selected-set、ranked-order vector、rooted-canonical vector relabel exact-match rate；
- canonical mean/std standalone raw-stratified BA；
- `STATS+canonical mean/std` BA。

Signal classifier 与 R0-P 完全相同，使用 split seeds：

```text
731201, 731202, 731203
```

这些 seeds 已用于 R0-P，因此本轮只是与 WALK 的 matched exploratory comparison，不是新的 confirmatory test。

## 8. 解释规则

n-hop 只被视为值得进入新的正式 dictionary protocol，当且仅当至少存在一个预定义 selector 同时满足：

1. rooted-canonical output relabel exact-match rate `>= 0.99`；
2. dominant canonical mass 不高于 WALK s=7 的 `0.3359`；
3. canonical effective signature count 不低于 WALK 的 `18.929`；
4. standalone BA 至少比 WALK canonical mean/std `0.6297` 高 `0.02`；
5. `STATS+n-hop` 不低于 STATS-only matched score。

这是一个严格的“值得继续”条件，不是论文显著性结论。若没有 selector 同时满足，不训练 n-hop KSVD；先承认 direct capped n-hop 没有解决当前瓶颈。

无论结果如何，本轮都禁止：

- 从三种排序中只报告最好者；
- 将 node-ID tie-break 称为严格 invariant；
- 增加 KSVD restart；
- 扫描 radius、patch size、K/T；
- 用 cleaned 替代 raw。
