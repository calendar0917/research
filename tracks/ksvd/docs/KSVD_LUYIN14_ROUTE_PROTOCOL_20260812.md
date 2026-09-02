# luyin14 路线闭环协议：补边、分类、关系与特征融合

> 日期：2026-08-12  
> 状态：结果前冻结  
> 来源：`docs/luyin/luyin14.txt` 中陈昱部分（24:20–28:44）

## 1. 研究问题

本轮只回答四个顺序固定的问题：

1. 当前高覆盖连续 patch 链之后，把剩余真实边显式放回表示，需要多少额外 patch / code？
2. `FAIR95`、`EDGE100` 与 `FAIR95 + residual-edge sidecar` 哪一种更适合图分类？
3. patch 内容与真实重叠关系的绑定，是否稳定优于打乱绑定？
4. 在带节点特征的数据上，结构表征是否提供 feature-only 之外的稳定增量？

不以本轮结果声称 KSVD atom 是 motif、graphlet 或 cell；不恢复已经否定的 labeled-adjacency codec 主张。

## 2. 数据集

首轮四个小型、可完整闭环的数据集：

| 类型 | 数据集 | 用法 |
|---|---|---|
| 纯结构 | IMDB-BINARY | 二分类，检查强结构统计 baseline |
| 纯结构 | IMDB-MULTI | 三分类，检查跨任务稳定性 |
| 带节点特征 | MUTAG | 小分子图，节点标签 one-hot |
| 带节点特征 | PTC_MR | 小分子图，节点标签 one-hot |

REDDIT-BINARY 留作规模外推；PROTEINS 留作多分量/较大图压力测试。它们不参与首轮 go/no-go。

## 3. Patch 与补边

- 最大 patch size：`s=8`；小于 8 个节点的图使用全图 patch 并 zero-pad 到 8。
- 连续链目标重叠：`o=2`（小图退化时只用于单块全图，不构造非法转移）。
- 节点预序：GLOBAL-WL stable order；块内：rooted-canonical slots。
- sampler：Beam4/R1 marginal cover；基础预算沿用 `patch_budget(..., multiplier=1.5)`。
- `FAIR95`：先沿连续 Beam 链寻找首次同时达到 edge coverage≥0.95、非孤立节点 incident recall p10≥0.90、node coverage=1 的前缀；若连续链提前停止，则逐次加入 edge-targeted connected completion patch，直到达到该门槛。completion patch 是显式新 segment，不伪装成满足 `o=2` 的连续转移。
- `EDGE100`：从 FAIR95 继续加入 edge-targeted completion patch，直到所有真实边被至少一个 patch 观察。
- `RESIDUAL`：不增加 patch，显式记录 FAIR95 后未覆盖真实边的低容量、不变 sidecar 统计。

必须区分 edge coverage=100% 与 all-pair coverage=100%；本轮只补真实边。

## 4. KSVD 与 readout

- 字典：每个 outer-train fold 单独拟合；test patch 不进入均值、初始化或字典。
- `K=24, T=3, T_min=1, iterations=5`。
- 初始化：train-centered patch 上 deterministic maximin；FINAL 从同一 INIT 更新。
- 训练 patch 超过 3000 时，按冻结 seed 无放回采样。
- graph code：每个 atom 的 activation frequency、mean absolute coefficient、RMS coefficient，合计 `3K` 维。
- relation binding：对连续相邻关系和全局 overlap 两个通道，计算低容量加权 code-pair 相似度；TRUE 与 SHUFFLED 只改变 code 到 patch position 的绑定。

## 5. 分类与融合

### 5.1 Outer evaluation

- split seeds：`0, 1, 2`；每个 seed 3-fold stratified CV。
- 主指标：balanced accuracy；同时报告 accuracy。
- 线性 head：train-only StandardScaler + `LogisticRegression(C=1, max_iter=5000)`。
- 所有 feature sets 共用完全相同的 folds。

### 5.2 必报 feature sets

- `STATS`
- `RAW_PATCH`
- `INIT_CONTENT`
- `FAIR95_CONTENT`
- `FAIR95_RELATION_GRAPH`
- `FAIR95_TRUE_RELATION`
- `FAIR95_SHUFFLED_RELATION`
- `EDGE100_CONTENT`
- `FAIR95_RESIDUAL`

带节点特征数据额外报告：

- `FEATURE_ONLY`
- `FEATURE_STATS`
- `FEATURE_FAIR95_CONTENT`
- `FEATURE_FAIR95_TRUE_RELATION`
- `FEATURE_FAIR95_RESIDUAL`
- `FEATURE_STRUCTURE_GATE`

gate 是固定宽度 32 的低容量双分支门控 MLP；只在 outer-train 内划 validation，test 不参与 epoch 选择。

## 6. 冻结判定

### 6.1 补边

- 若 EDGE100 的额外 patch/code 明显高于 residual sidecar，且分类没有稳定提升，默认采用 residual。
- 只有 EDGE100 在至少 3/4 数据集上相对 FAIR95 平均提升 ≥1 balanced-accuracy point，才把“继续加 patch 补边”保留为默认路线。

### 6.2 关系

- `TRUE_RELATION - SHUFFLED_RELATION` 必须在至少 3/4 数据集为正，且跨全部 9 folds 的平均增益 ≥1 point，才认为当前关系绑定有效。

### 6.3 节点特征融合

- concat 或 gate 至少在两个带特征数据集之一达到 ≥1 point，并且另一个数据集不低于 feature-only 超过 1 point，才认为结构通道具有可迁移互补性。

### 6.4 总体

- 三项都失败：KSVD 保留为 compressor / diagnostic baseline，停止扩大模型。
- 仅补边通过：继续研究 representation completeness，不进入复杂融合。
- 关系或融合通过：只沿通过的单一轴做下一轮，不同时扫描 sampler、K/T 与网络深度。
