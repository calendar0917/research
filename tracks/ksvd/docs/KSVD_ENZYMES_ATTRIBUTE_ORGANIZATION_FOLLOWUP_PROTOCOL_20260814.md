# ENZYMES 连续属性组织信息 matched follow-up

> 日期：2026-08-14  
> 状态：完整属性预筛结果可见后冻结；只做机制解释，不修改原 Beam8 晋级 gate。

## 1. 问题

预筛中 `GIN_FULL_PLUS_GLOBAL` 比 `GLOBAL_STATS_LINEAR` 高 7.68pt，但完整属性 GIN 单独只高
1.56pt。需要区分：

1. 连续节点属性在图结构中的组织确实提供了 GLOBAL_STATS 外信息；
2. global residual 只是对任意不稳定 GIN 做通用校准。

## 2. Matched control

保持预筛的 split/model/fold、GIN epoch、归一化、rank-16 residual 和 checkpoint 流程，新增：

- `GIN_LABEL_ONLY_PLUS_GLOBAL`：GIN 只读取 3 维离散 node labels，再加相同 GLOBAL_STATS residual；
- `GIN_FULL_PLUS_GLOBAL`：冻结原预筛的 18 continuous + 3 discrete 结果。

label-only GIN 分数必须精确复现原预筛。只有输入是否包含 18 维连续属性不同。

## 3. 解释 gate

若 FULL+GLOBAL−LABEL+GLOBAL mean `>=+3pt`、至少18/27为正、三个 split means 全正且至少
2/3 model means 为正，则判定：

`CONTINUOUS_ATTRIBUTE_ORGANIZATION_ADDS_BEYOND_GLOBAL_STATS`

否则判定：

`FULL_GLOBAL_GAIN_NOT_ATTRIBUTED_TO_CONTINUOUS_ATTRIBUTE_ORGANIZATION`

无论结果如何，原预筛的 `ENZYMES_DO_NOT_ADVANCE_TO_BEAM8` 不在本轮事后修改；若该机制通过，
只说明 ENZYMES 值得作为结构—属性融合数据集重新设计未见验证协议。
