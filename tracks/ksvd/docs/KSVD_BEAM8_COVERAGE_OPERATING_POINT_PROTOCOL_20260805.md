# Stable-ID Beam8 coverage operating-point 审计协议

> 日期：2026-08-05
> 状态：结果前冻结
> 前置：`KSVD_GLOBAL_STABLE_NODE_ID_PROTOCOL_20260805.md` 与 `KSVD_BEAM8_EDGE_RESIDUAL_PROTOCOL_20260805.md`
> 范围：完整已知、无向、无自环二值图；不使用 graph labels；本轮选择 coverage operating point，不重新主张 labeled-adjacency codec。

## 1. 问题

当前 `s10/o3/m1.5/B8_R1` 的约 0.875 edge coverage 不能直接冻结，因为：

1. `s/o/m` Pareto 来自旧 target-edge sampler，而非 Beam8；
2. `m=1.5` 是容量 multiplier，不对应固定 coverage 或节点级公平性；
3. 总 edge coverage 可能掩盖少数节点或特殊边系统性漏失；
4. residual 只补解码时的未观察真实边，不会让该边进入 patch/KSVD 表示。

本轮固定已经通过审计的 identity/slot 机制，只裁决：

- `s8/o2`、`s10/o3`、`s12/o4` 在同一 stable-ID Beam8 下的 prefix coverage frontier；
- BASE、EDGE90、EDGE95、FAIR95、EDGE99、EDGE100 的成本和漏边结构；
- 哪些 operating points 值得进入 rooted-canonical KSVD matched follow-up。

## 2. 冻结预处理和 sampler

```text
input adjacency
→ GLOBAL_WL stable-ID preorder
→ marginal candidate Beam8/R1
```

设置：

```text
retained_beam = 8
candidate_restarts = 1
maximum_patches = 60
base multiplier = 1.5
cover seeds = 970101 / 970102 / 970103
geometries = s8/o2, s10/o3, s12/o4
```

本轮不改变 candidate score、不搜索第一 patch、不放宽 retained-overlap connected 条件。这样 geometry/stop 比较不会与 sampler 改法混杂。若三种 geometry 均失败，再另立 candidate-generation follow-up。

## 3. 图集

### 3.1 主审计

沿用固定 bank：

```text
graph_bank_seed = 810001
families = regular / small_world / block
n = 50
degrees = 15 / 20 / 25
8 graphs per cell = 72 graphs
```

### 3.2 稀疏 transfer probe

检验 node-bound 主导区间，不把主 bank 结论外推为一般规律：

```text
graph_bank_seed = 810002
families = regular / small_world / block
n = 50
degrees = 3 / 4
4 graphs per cell = 24 graphs
```

稀疏 probe 必须报告 sampler failure；失败图不能从均值中静默删除。

## 4. Prefix checkpoints

每个 graph × geometry × seed 只生成一条最长 60 patch 的链，所有 checkpoint 取同一条链前缀：

- `BASE`：旧 `patch_budget(..., multiplier=1.5)`；
- `EDGE90/95/99/100`：首次达到对应 global true-edge coverage；
- `FAIR95`：首次同时满足：
  - global edge coverage `>=0.95`；
  - 节点 incident-edge recall 的 graph-level p10 `>=0.90`；
  - node coverage `=1.0`。

未在60 patches内达到即标记 unreachable，不用最后前缀代替。

## 5. 必报 coverage 指标

### 5.1 全局

- patch count、raw pair slots、unique observed pairs；
- node/pair/true-edge coverage；
- residual edge count、RAW zero-fill coverage RMSE；
- marginal new edges at checkpoint 与 mean new edges/patch；
- covered edges / unique observed pairs；
- edge observation multiplicity mean/CV。

### 5.2 节点公平性

对每个非孤立节点：

```text
incident recall(v) = covered incident true edges / degree(v)
```

报告 mean、p10、minimum、fully-covered-node fraction、nodes-with-residual fraction，以及 residual incident count 的 maximum。

### 5.3 漏边结构

对每类真实边报告 category size、category recall、residual share：

- 至少一个 endpoint degree 位于 graph q25 以下的 `low-degree incident`；
- 至少一个 endpoint degree 位于 graph q75 以上的 `high-degree incident`；
- 两端没有共同邻居的 `zero-common-neighbor`；
- graph bridge（若存在）；
- block family 中跨前25/后25节点的 `cross-block`。

空 category 记为 null，不用 vacuous 1.0 混入均值。

## 6. 成本指标

沿用 edge-residual 协议的显式代理，并按 geometry 重算：

- canonical/ordered identity bits；
- prefix framing；
- K24/T3/q8 KSVD code proxy；
- residual subset bits；
- canonical hybrid proxy bits；
- direct bitset/enumerative controls；
- dictionary scalars `C(s,2)×24`；
- code scalars `patch_count×3`。

这些成本只用于比较 operating points，不恢复已停止的 global codec 主张。

## 7. Coverage 判定

### 7.1 几何可行性

一个 geometry 进入 KSVD follow-up 必须在主审计满足：

1. cover invariants 全通过；
2. `EDGE95` reach fraction `>=0.99`；
3. `FAIR95` reach fraction `>=0.95`；
4. FAIR95 的 mean global edge coverage `>=0.95`；
5. FAIR95 的 mean node p10 incident recall `>=0.90`；
6. 三个 cover seeds 分别满足 1–5。

### 7.2 Coverage 候选

对所有通过 geometry，保留：

- 它的 `BASE` control；
- 它的 `FAIR95` candidate。

不在 coverage-only 阶段强行选单一赢家。用以下 axes 标记 nondominated candidates：

```text
minimize: patch count, unique observed pairs, code scalars, dictionary scalars
maximize: edge coverage, node p10 recall, low-degree-edge recall
```

只要某 candidate 在所有 axes 不劣且至少一项严格更好，才称支配。

### 7.3 稀疏 transfer

单独给出：

- 60-patch sampler success；
- BASE/FAIR95 reach；
- 各 coverage/fairness 指标。

transfer 失败会限制外推，但不反向删除主 bank 的可行 candidate。

## 8. Rooted-canonical KSVD follow-up

只对主审计通过的 `geometry × {BASE, FAIR95}` branches 运行：

```text
rooted-canonical local slots
3-fold graph isolation
K=24, T=3, minimum sparsity=1, iterations=25
cover seeds = 970101 / 970102 / 970103
```

必须报告：

- patch relative error；
- observed-pair RMSE/F1；
- full RMSE/edge recall/F1；
- residual-corrected full RMSE；
- coverage RMSE 与 compression RMSE 分解；
- code/dictionary scalars。

Residual-corrected full RMSE 只把未观察真实边精确补为1；observed-pair KSVD 错误不修正。

`FAIR95` 相对同 geometry BASE 的 adoption gate：

1. uncorrected full RMSE 相对下降 `>=2%`；
2. residual-corrected full RMSE 不恶化超过 `1%`；
3. observed-pair RMSE 不恶化超过 `2%`；
4. full edge recall 不下降；
5. 三 cover seeds 至少2个通过。

若无 FAIR95 分支通过，冻结 `BASE + explicit residual second channel`，而不是宣称更高 patch coverage 有表示收益。若有多个通过，再按 full RMSE、corrected RMSE、code scalars 和 dictionary scalars做 Pareto选择。

## 9. 边界

- 本协议只检验 numeric relabel 已被 stable-ID preprocessing 消除后的 operating point；不重新审计 stable ID。
- `FAIR95` 是预注册研究阈值，不宣称95%具有普适语义。
- residual 作为解码 sidecar与作为下游 token 是两种方案；本轮 reconstruction follow-up只检验前者。
- 若 sparse transfer 与主 bank选择不同，必须报告 domain dependence，不用主 bank 单一参数覆盖所有图域。
