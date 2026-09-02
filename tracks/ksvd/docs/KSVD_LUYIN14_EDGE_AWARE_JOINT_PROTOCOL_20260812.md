# luyin14：edge-aware 结构—属性共享稀疏码 strict screen

> 日期：2026-08-12  
> 状态：结果前冻结  
> 数据：MUTAG / PTC_MR；不构造数据集。

## 1. 前置诊断

两个真实分子数据集都提供 4D edge attributes，但此前 adjacency patch 和 GIN 基线都只使用
无类型边。rooted `v∪N(v)` patch 在 MUTAG/PTC_MR 仅有 4/6 种；加入 edge type 后为 13/17
种。无类型结构 token 因而可能丢掉关键化学关系。

本轮回答：

> 在公平的 edge-aware GINE 基线上，typed-structure 与 node-attribute 共享稀疏码是否仍提供
> 可归因于真实绑定和 K-SVD updates 的分类增量？

## 2. 表示

- 节点 patch：rooted `v∪N(v)`，最多 8 节点；
- canonicalization 使用 edge type；
- 结构 view：28 个上三角位置 × 4 类 edge one-hot，112D；
- 属性 view：中心节点属性 + 一跳邻居属性均值；
- outer-train-only block centering/RMS scaling；
- shared-code joint K-SVD：`K24/T3/updates5`。

## 3. 分类与 controls

共同 backbone 为 3-layer GINE，所有变体都读取原始 4D edge attributes：

- `GINE_ONLY`；
- `TYPED_JOINT_FINAL_TRUE`：FINAL code 以 zero-init FiLM 注入每层；
- `TYPED_JOINT_FINAL_SHUFFLED`：每图内打乱 attribute context 后重新拟合/编码；
- `TYPED_JOINT_INIT_TRUE`：同一 INIT、无 K-SVD updates。

严格 checkpoint、确定性设置与 joint multiview Stage A 相同。

## 4. 晋级条件

至少一个数据集同时满足，且另一数据集相对 GINE 不低于 `-0.01`：

1. FINAL−GINE `>=+0.01` 且至少 2/3 folds 正；
2. TRUE−SHUFFLED `>=+0.005` 且至少 2/3 folds 正；
3. FINAL−INIT `>0` 且至少 2/3 folds 正；
4. FINAL reconstruction 明显低于 INIT。

若失败，停止在 MUTAG/PTC_MR 上增加融合容量；若通过，才扩展 split seeds 1/2。
