# IMDB-BINARY R0-P：完整 raw 数据上的 WALK substrate 审计

> 日期：2026-07-31
>
> 探索性结论：**FAIL_R0P_RAW_WALK_SUBSTRATE**

## 1. 本轮边界

- 主数据是完整 raw IMDB-BINARY（1000 图），不是 cleaned 替代 benchmark。
- cleaned 只用于解释重复图和冲突标签，不用于替代 raw 分数。
- 本轮不训练 KSVD；只审计真实 patch substrate、置换性质和非 KSVD signal exposure。
- 普通 raw stratified CV 是 benchmark view；exact-isomorphism grouped CV 留到下一阶段。

## 2. Raw 与 cleaned 的精确同构结构

| 指标 | raw | cleaned |
|---|---:|---:|
| 图数 | 1000 | 493 |
| exact structure groups | 537 | 493 |
| duplicate groups | 116 | 0 |
| graphs in duplicate groups | 579 | 0 |
| label-conflict groups | 44 | 0 |
| graphs in label-conflict groups | 318 | 0 |
| pure-structure empirical ceiling | 0.8860 | 1.0000 |

Raw 有 537 个不同结构；其中 44 个结构组包含互相冲突的 graph labels，涉及 318 张图。去掉冲突组后剩 493 个 label-consistent structure groups；其类别代表数为 {'0': 261, '1': 232}。
因此任何对精确同构严格不变的纯结构确定性分类器，即使在这 1000 张图上记住每个结构并按组内多数标签预测，经验 accuracy 上限也只有 0.8860；至少 114 个样本不可同时判对。

这解释了为什么 cleaned 适合敏感性分析，但也说明它改变了任务：它不仅去重，还排除了 raw 中无法由纯结构唯一判定的冲突样本。

## 3. Raw WALK patch substrate

- 图数：1000；patch 数：17526。
- 每图 patch 数 min/median/max：12/17.0/24。
- 根覆盖率 min/median/mean：0.1765/1.0000/0.9465；全根覆盖图数：795。
- patch edge mean：12.1883。
- tree fraction：0.0001。
- clique fraction：0.4422。
- missing-at-most-one-edge fraction：0.4501。
- WALK unique vectors：565。
- rooted-canonical unique signatures：128。
- canonical effective signature count：9.7626。
- dominant canonical signature mass：0.4422。
- within-graph canonical unique fraction min/median/mean：0.0417/0.3846/0.3772。

Patch edge-count histogram：

```text
{"10": 307, "11": 3420, "12": 730, "13": 335, "14": 139, "15": 7750, "5": 1, "6": 70, "7": 520, "8": 739, "9": 3515}
```

## 4. 节点重编号审计

对 100 张图、1746 条已采样 walk trajectory 进行整体节点重编号后映射：mismatch=0，maximum L1=0.0000。

这里验证的是严格命题：给定同一条抽象 walk trajectory，first-discovery-order induced adjacency 不随原始节点 ID 改变。独立随机采样的逐样本结果不要求完全相同。

## 5. Raw stratified signal exposure（尚未训练 KSVD）

| 特征 | 3 split-seed mean BA | split-seed std | per-seed means |
|---|---:|---:|---|
| graph_stats | 0.7003 | 0.0054 | [0.7020, 0.7060, 0.6930] |
| walk_mean_std | 0.6057 | 0.0135 | [0.5920, 0.6240, 0.6010] |
| canonical_mean_std | 0.6300 | 0.0045 | [0.6290, 0.6250, 0.6360] |
| edge_count_histogram | 0.6377 | 0.0048 | [0.6420, 0.6310, 0.6400] |
| graph_stats_plus_walk_mean_std | 0.6927 | 0.0065 | [0.7010, 0.6920, 0.6850] |
| graph_stats_plus_canonical_mean_std | 0.6940 | 0.0073 | [0.7040, 0.6870, 0.6910] |
| graph_stats_plus_edge_count_histogram | 0.7003 | 0.0017 | [0.7020, 0.7010, 0.6980] |
| walk_label_shuffle | 0.5039 | 0.0184 | [0.5258, 0.4807, 0.5053] |

这些分数只证明 raw WALK summaries 是否暴露标签信号。KSVD 的 added value 必须在下一阶段通过同一 fold 内的 `STATS+INIT` vs `STATS+FINAL` 单独归因。

## 6. Gate

### Substrate checks

- [x] `dominant_canonical_signature_le_0_50`
- [ ] `canonical_effective_signature_count_ge_10`
- [x] `median_within_graph_canonical_unique_fraction_ge_0_20`
- [x] `mapped_trajectory_exact_invariance`

### Signal checks

- [x] `walk_mean_ba_ge_0_60`
- [x] `walk_exceeds_shuffle_by_0_05`
- [x] `shuffle_mean_in_0_45_0_55`

下一步：Stop before dictionary learning and revise only the patch substrate/representation.

## 7. 下一阶段必须保留的三种视角

1. `raw/stratified`：与常见 IMDB-BINARY benchmark 设置接近，用于外部参考。
2. `raw/exact-isomorphism-grouped`：保留 1000 图，但同构组不跨折，用于结构泛化判断。
3. `cleaned/stratified`：只做去重与冲突样本移除后的敏感性分析，不替代 raw benchmark。
