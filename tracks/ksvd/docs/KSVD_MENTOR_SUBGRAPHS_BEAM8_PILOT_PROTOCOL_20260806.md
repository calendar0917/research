# 导师 50-node 真实子图：Stable-ID Beam8 pilot 协议

> 日期：2026-08-06  
> 状态：结果前冻结  
> 数据：`data/subgraphs_50_20_10000_batch_0.pkl`  
> SHA-256：`936134e7876740d5efc478f6bdf9663e31440ab506ecee507b00f8e32a7f05cb`  
> 前置：`KSVD_GLOBAL_STABLE_NODE_ID_PROTOCOL_20260805.md`、`KSVD_BEAM8_COVERAGE_OPERATING_POINT_PROTOCOL_20260805.md`

## 1. 目的

本 pilot 不使用 graph labels，也不重新打开普通 KSVD motif-vocabulary 主张。它只裁决：

1. 导师真实子图能否无损接入当前二值、无向、连通图管线；
2. `GLOBAL_WL stable preorder → Beam8 → rooted-canonical local slots` 是否从合成 50-node bank 迁移到真实子图；
3. `s8/o2`、`s10/o3`、`s12/o4` 的 BASE/FAIR95 coverage frontier 在真实密度分布上是否可行；
4. 哪些 branches 可以进入 grouped-split、train-only KSVD reconstruction follow-up。

本轮不训练 Transformer，不把 source node ID 输入 KSVD，也不声称 patch chain 是优于直接 adjacency 的 codec。

## 2. 已知数据事实与边界

结果前只允许使用格式体检中已经确认的事实：

- 顶层为 10000 个 `networkx.Graph`；每图恰好 50 节点；
- 全部无向、无自环、连通；无 node/graph labels；边权恒为 1；
- 10000 子图共享 2805 个 source node IDs，且共同出现节点对的邻接关系未发现冲突；
- 平均度范围显著宽于导师口头描述的 15–25，因此必须按密度分层报告；
- 第一个 insertion-order 节点是否为 sampling root 尚未由生成方确认，只称 `root_candidate`。

Source IDs 只允许用于：数据一致性校验、root/group bookkeeping、train/test overlap audit、patch membership/transition/stitching 映射。45D patch vector 只来自 rooted-canonical local adjacency，不包含 source ID 数值或字符串。

## 3. 紧凑缓存契约

原始 pickle 只读保留。转换缓存必须使用 `np.load(..., allow_pickle=False)` 可读的非 object arrays，至少包含：

```text
adjacencies:       uint8 [G,50,50]
global_node_ids:   int32 [G,50]
source_node_vocab: unicode [N_global]
root_global_ids:   int32 [G]
edge_counts:       int32 [G]
average_degrees:   float64 [G]
metadata_json:     unicode scalar/array
```

转换 gate：

1. source checksum 严格匹配；
2. 10000/10000 图通过 simple-undirected-binary-connected 契约；
3. source-node vocabulary、edge ID endpoint 与共同出现 pair adjacency 无冲突；
4. cache round-trip 后逐图 node/edge count 与邻接矩阵严格一致。

任何失败都保留在报告中，不静默丢图或二值化。

## 4. Pilot 抽样

正式 pilot 固定 500 图，selection seed=`20260806`。

密度层按 average degree 固定为：

```text
lt5 / d5_10 / d10_15 / d15_25 / ge25
```

每层目标 100 图；若某层不足，再按剩余容量确定性补齐。层内按 `root_candidate` round-robin，优先增加 root 多样性，再由固定 seed 决定同 root 内顺序。Density/root 只用于抽样与条件报告，不作为模型标签或 checkpoint 选择信号。

另从 pilot 中固定选择最多 50 图作为 relabel stability 子集：每个 density stratum 均匀抽取，permutation seeds=`960201/960202`。

## 5. Frozen preprocessing 与 sampler

```text
input adjacency in source insertion order
→ compute GLOBAL_WL stable structural classes/IDs
→ reorder adjacency by GLOBAL_WL preorder
→ marginal candidate Beam8/R1
```

设置：

```text
geometries = s8/o2, s10/o3, s12/o4
retained_beam = 8
candidate_restarts = 1
maximum_patches = 60
base multiplier = 1.5
cover seed = 970201
```

每个 graph × geometry 只生成一条最长 60 patch chain。所有 checkpoint 取同一条链前缀：`BASE / EDGE90 / EDGE95 / FAIR95 / EDGE99 / EDGE100`。采样失败与 checkpoint unreachable 必须显式保留。

## 6. 必报指标

### 6.1 数据与身份

- node/edge/average-degree 分布；density strata 与 root-candidate 数量；
- source node vocabulary、union edge、共同出现 nonedge 与冲突数；
- GLOBAL_WL singleton-node fraction、fully-singleton graph fraction、largest class；
- stable-ID 计算 wall-clock。

### 6.2 Coverage

对每个 geometry × checkpoint，整体和 density strata 分别报告：

- reach fraction、patch count、raw pair slots、unique observed pairs；
- node/pair/true-edge coverage；
- residual edge count、RAW zero-fill RMSE；
- node incident-edge recall mean/p10/min；
- fully-covered-node fraction、nodes-with-residual fraction；
- marginal new edges、edge multiplicity；
- low/high-degree incident、zero-common-neighbor 与 bridge edge recall；
- dictionary/code scalars 与 optimistic hybrid bits（只作相对成本代理）。

真实数据没有 block ground truth，因此不报告 synthetic `cross_block` 主指标；若后续加入自动社区划分，必须标为 exploratory proxy。

### 6.3 Relabel stability

对 stability 子集的原图和独立 numeric relabel 图：

- stable-class exact match；
- singleton unique-ID match；
- reordered adjacency match；
- Beam8 abstract chain match / patch Jaccard；
- rooted-canonical vector-row match；
- transition-map match；
- edge/pair coverage 与 RAW RMSE delta；
- fully-singleton 与 ambiguous graphs 分层。

Automorphism class 内的具体 node identity 不可识别；只允许主张 equivalence-class stability。

## 7. Pilot gates

一个 geometry 可进入真实 KSVD follow-up 必须满足：

1. cover invariant fraction `=1.0`；
2. sampler success fraction `>=0.99`；
3. BASE reach fraction `>=0.99`；
4. EDGE95 reach fraction `>=0.95`；
5. FAIR95 reach fraction `>=0.90`；
6. 不得有任何样本数不少于 20 的 density stratum，其 sampler success `<0.95`。

Stable preprocessing gate：

1. stable-class match `=1.0`；
2. singleton unique-ID match `>=0.999`（若没有 singleton trial 则单独记 null，不伪造 1.0）；
3. reordered adjacency match `>=0.99`；
4. rooted-vector row match `>=0.99`；
5. transition-map match `>=0.99`；
6. edge/pair coverage mean absolute delta `<=0.01`。

分类：

- 数据 gate、stable gate 均通过且至少一个 geometry 通过：`READY_FOR_GROUPED_KSVD_RECONSTRUCTION_FOLLOWUP`；
- coverage geometry 通过但 stable gate 不通过：`COVERAGE_READY_STABILITY_REQUIRES_REPAIR`；
- stable gate 通过但无 geometry 通过：`STABLE_PREORDER_READY_BEAM8_GEOMETRY_NO_GO`；
- 数据契约失败：`FAIL_MENTOR_SUBGRAPH_DATA_CONTRACT`。

本 pilot 不用 coverage-only 数字强行选单一 geometry。通过 branches 进入下一轮 train-only KSVD 后，再按 reconstruction、dictionary scalars 与 per-graph code scalars做 Pareto裁决。

## 8. 后续顺序

只有本 pilot 达到 `READY_FOR_GROUPED_KSVD_RECONSTRUCTION_FOLLOWUP` 才执行：

1. 建立 random-reference 与 root/source-overlap-aware grouped splits；
2. 比较 RAW / PCA-3 / INIT / FINAL-KSVD / random dictionary；
3. BASE 与 FAIR95 均报告 uncorrected 和 exact-residual-corrected reconstruction；
4. FINAL 只有在 grouped folds 稳定优于 INIT，并形成相对 PCA/random dictionary 的稀疏率—误差 Pareto 时，才进入 relation Transformer 主表。

否则不得通过扫描 `K/T/iterations/restarts` 修复 pilot 失败。
