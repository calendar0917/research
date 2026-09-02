# Beam8 small attributed-TU patch-graph prescreen

> 日期：2026-08-14  
> 状态：BZR/COX2/DHFR 分类结果不可见前冻结。

## 1. 研究问题

此前分类接口将 Beam8 patches 汇总为 graph vector 或 node residual。本轮不训练 patch-GNN，
先用固定无参数 relation operator 检查完整 patch graph 是否在新的小型真实任务上同时满足：

1. 正确 patch-token / patch-position binding 可检测；
2. relation 表示超过同 token multiset 的 BAG；
3. 收益不被简单全图统计完全替代。

只有低容量预筛通过的数据集，才允许在新的未见 split 上训练一层 patch-GNN。

## 2. 数据

- `BZR`：405 graphs；
- `COX2`：467 graphs；
- `DHFR`：756 graphs。

全部为真实 TU 分子图、二分类、规模远小于 OGB。原始3维连续属性是中心化空间坐标；为避免
旋转/坐标轴依赖，本轮预筛只使用官方离散 node labels 与无类型边。连续坐标不进入 sampler、
tokens、统计 baseline 或分类器。

## 3. Beam8 patch graph

- patch size `s=8`，target overlap `o=2`；
- retained beam `8`，candidate restarts `1`；
- edge-capacity multiplier `1.5`；
- full graph discrete node-label colors 用于 component order、patch canonicalization 与
  automorphism legality；
- structural token 为8个 canonical slots 的离散 node-label one-hot，加28维 adjacency upper
  triangle；不拟合 KSVD dictionary；
- patch graph relations：`PREVIOUS/NEXT`、任意 patch overlap Jaccard、canonical-slot
  persistence。

## 4. 固定低容量 readout

所有 token 仅以 outer-train patch rows 做逐维标准化。图级分支：

- `GLOBAL_STATS`：node-label mean/max/sum + 图大小、边数、degree/cycle 等基础统计；
- `RAW_BAG`：token mean/std/max + 相同 position/relation-topology/residual sidecars；
- `RAW_PATCH_GRAPH_TRUE`：在 BAG common features 上，对 PREVIOUS、OVERLAP、SLOT 三个通道
  追加固定10维 pair statistics；
- `RAW_PATCH_GRAPH_TOKEN_SHUFFLED`：patch graph、关系权重、token multiset 与 sidecars 不变，
  只打乱 token 到 patch positions 的 binding。

分类器统一为 `StandardScaler + class-weight-balanced LogisticRegression(C=1)`。不扫描 C、
patch geometry、relation channels、特征统计或分类器。

## 5. 重标号审计

每个数据集固定前32 graphs × 2 permutations，要求：

- structural token rows = 100%；
- relation matrices = 100%；
- TRUE patch-graph feature = 100%；
- TOKEN_SHUFFLED feature = 100%。

mapped patch-set chain exact 仅作 diagnostic。任一 required check 失败则该数据集结果作废。

## 6. Evaluation

- split seeds `0/1/2`；
- 每个 split 3-fold stratified CV；
- 每个数据集共9个 outer units；
- 主指标 balanced accuracy；所有 variants 共用 folds。

## 7. 数据集晋级 gate

某一数据集必须全部满足：

1. `TRUE−TOKEN_SHUFFLED >= +1pt`，至少6/9正；
2. `TRUE−BAG >= +0.5pt`，至少6/9正；
3. `TRUE−GLOBAL_STATS >= 0pt`，至少5/9正；
4. binding 与 relation increment 的 split0/1/2 means 全正；
5. 固定重标号审计全部100%通过。

通过则判定：

`<DATASET>_BEAM8_PATCH_GRAPH_ADVANCES_TO_ONE_LAYER_GNN`

失败则判定：

`<DATASET>_BEAM8_PATCH_GRAPH_PRESCREEN_NOT_ESTABLISHED`

本轮是数据集/机制预筛，不报告为独立确认性结果。只有通过者才能在 split seeds 3/4 上运行
一层 patch-GNN，并必须保留 BAG、TOKEN_SHUFFLED 与随机/matched cover controls。

