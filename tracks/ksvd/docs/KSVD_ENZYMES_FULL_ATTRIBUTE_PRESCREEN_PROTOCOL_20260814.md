# TU ENZYMES 完整属性 Beam8 前置筛选协议

> 日期：2026-08-14  
> 状态：任何 ENZYMES Beam8 分类结果可见前冻结。

## 1. 研究问题

Mutagenicity/NCI1 上的 Beam8 分类收益被宏观图统计吸收。本轮不立即训练 Beam8，而先判断
ENZYMES 是否提供更有区分度的真实小型 TUData 环境：

> 读取 ENZYMES 全部 18 维连续 node attributes 与 3 维离散 node labels 后，GIN 是否能稳定
> 超过只读取节点属性 readout 和宏观拓扑的 GLOBAL_STATS？

若不能，ENZYMES 不进入 Beam8；若能，才说明任务中可能存在全图统计未覆盖的局部组织信息。

## 2. 数据与特征边界

- 数据：TU ENZYMES，600 graphs、6 classes；不构造数据集；
- GIN 输入：18 维连续属性 + 3 维离散 node-label one-hot；
- 连续属性按训练节点拟合 mean/std，离散 one-hot 不标准化；
- 后续 Beam8 canonicalization 只允许使用 3 维离散 labels；连续属性只能进入 patch content；
- GLOBAL_STATS：完整节点特征 mean/max/sum，加图大小、度统计、连通分量、三角形、
  transitivity 与 cycle rank；所有 scaler 仅在 outer-train 拟合。

## 3. 验证矩阵

- split seeds：0/1/2；每个 split 3-fold stratified CV；
- model seeds：0/1/2；共 27 outer-test units；
- 每个 outer fold 内再划分 inner-train/validation；
- inner validation 选择 GIN 与 frozen residual epoch，outer-test 只评估一次；
- 模型：3-layer GIN、hidden 64、sum readout、dropout 0.5；
- 比较：
  - `GLOBAL_STATS_LINEAR`；
  - `GIN_LABEL_ONLY`；
  - `GIN_FULL_ATTRIBUTES`；
  - `GIN_FULL_PLUS_GLOBAL`：冻结 full-attribute GIN 后训练 rank-16 global-stat residual。

## 4. 晋级 gate

全部满足才判定 `ENZYMES_ADVANCE_TO_BEAM8_ATTRIBUTED_SCREEN`：

1. `GIN_FULL_ATTRIBUTES - GLOBAL_STATS_LINEAR >= +3pt`，至少 18/27 units 为正；
2. `GIN_FULL_ATTRIBUTES - GIN_LABEL_ONLY >= +3pt`，至少 18/27 units 为正；
3. FULL−GLOBAL 的三个 split means 全部为正；
4. FULL−GLOBAL 的三个 model-seed means 至少两个为正；
5. `GIN_FULL_PLUS_GLOBAL - GLOBAL_STATS_LINEAR >= +3pt`；
6. 数据维度精确为 18 continuous + 3 discrete，且至少 50% graphs 超过 16 nodes。

失败时不为 ENZYMES 实现 Beam8 融合。通过时只授权下一阶段低容量
`TRUE/BAG/SHUFFLED/HIST/RANDOM` attributed Beam8 screen，不授权 cross-attention、监督字典或
普通 KSVD 参数扫描。
