# Beam8/NCI1 geometry feasibility protocol

> 日期：2026-08-13  
> 状态：结果前冻结；只使用输入图结构和节点属性，不使用 graph labels。

## 1. 目的

正式 `s10/o3` Stage A 显示 RAW 的真实关系绑定可检测，但 NCI1 上平均 chain 很短，
且不少分量在 BASE budget 前停止。本轮不看分类结果选择 geometry，只比较已经在旧
Beam8 operating-point audit 中预先登记的两个候选：

- `s10/o3/Beam8/R1/m1.5`；
- `s8/o2/Beam8/R1/m1.5`。

每张图、每个 geometry 只生成一次 BASE cover；非连通分量仍各自成为独立 segment。

## 2. 必报无标签指标

- mean patches 与 patch-count quantiles；
- single-patch graph fraction；
- graphs with chain edges fraction；
- mean directed chain edges；
- partial-component fraction；
- mean edge/node coverage；
- graph-level edge coverage p10；
- non-chain overlap 可用图比例；
- node-label histogram 在相邻 patch 间的真实/打乱 cosine margin。

最后一项只使用 node labels 作为输入属性，不使用 graph labels；TRUE/SHUFFLED 共用
相同 token multiset 和相同 chain，只改变 token-to-position binding。

## 3. 冻结 gate

只有 `s8/o2` 同时满足以下条件，才允许进入分类：

1. single-patch graph fraction 相对下降至少 20%；
2. graphs-with-chain fraction 不低于 `s10/o3`；
3. mean directed chain edges 相对提高至少 20%；
4. mean edge coverage 不低于 `s10/o3` 超过 1 point；
5. partial-component fraction 不高于 `s10/o3`；
6. 相邻 node-label histogram 的 TRUE−SHUFFLED cosine margin 为正。

若 gate 失败，不用分类标签救 geometry；保留 `s10/o3` 结果并转向 relation readout 或
属性对齐机制。若 gate 通过，分类协议、folds、K/T/iterations、linear head 和判定门槛
全部沿用上一轮，只把 geometry 改成 `s8/o2`。

