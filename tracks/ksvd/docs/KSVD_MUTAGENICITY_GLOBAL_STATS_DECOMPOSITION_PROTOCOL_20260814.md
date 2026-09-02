# Mutagenicity 全图统计校准来源分解

> 日期：2026-08-14  
> 状态：Beam8 条件增量失败后冻结的解释性诊断。

## 1. 目的

`GLOBAL_STATS` 在 18/18 units 超过 `BEAM8_FULL`，且 Beam8 在冻结 `GLOBAL_STATS` 后没有条件
增量。本轮不再寻找新的 Beam8 模型，而是定位统计校准来自哪里，以指导后续 TUData 选择：

- 节点属性组成（mean）；
- 某类节点是否出现（max）；
- 各类节点的绝对计数（sum）；
- 上述全部属性统计；
- 图大小、度分布、连通分量、三角形、transitivity、cycle rank 等纯拓扑统计；
- 属性与拓扑的完整组合。

## 2. 公平性

- 数据、split3/4 × model0/1/2 × 3 folds、GINE checkpoint 与 inner/full 训练流程保持不变；
- 每个变体都 padding 到 Beam8 FULL 的相同输入维度，使用同一 rank-16 residual；
- 归一化只在 outer-train nodes 上拟合；
- GINE 与 `GLOBAL_FULL` 必须逐 unit 精确复现 specificity controls。

## 3. 解释规则

- 若某单一 component 与 `GLOBAL_FULL` 相差不超过 `0.5pt`，称为 near-sufficient shortcut；
- 若没有单一 component near-sufficient，而 `GLOBAL_FULL` 明显更高，则收益来自属性与拓扑组合；
- 该诊断只解释 Mutagenicity，不直接外推到其他数据集，也不授权扩大 Beam8 模型。
