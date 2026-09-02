# Beam8/Mutagenicity typed feasibility protocol

> 日期：2026-08-13  
> 状态：分类结果前冻结；不使用 graph labels。

## 1. 目的

NCI1 上 `s8/o2` 能稳定暴露真实 patch binding，但关系没有提供超过 BAG/属性基线的
分类增量。Mutagenicity 提供 14 类 node labels 与 3 类真实 bond labels。本轮先判断：

1. `s8/o2` 是否仍比 `s10/o3` 形成更可用的 Beam8 chain；
2. Beam8 是否覆盖各 bond type，而不是只覆盖占绝对多数的单键；
3. 相邻 patch 的 node/bond composition 是否具有优于打乱绑定的连续性。

不使用 graph labels，不训练字典或分类器。

## 2. Geometry

只比较旧 Beam8 operating-point audit 已登记的：

- `s10/o3/Beam8/R1/m1.5`；
- `s8/o2/Beam8/R1/m1.5`。

非连通分量独立形成 segment；分量内 GLOBAL-WL stable preorder，patch 内
rooted-canonical slots。所有 cover 仅使用 binary adjacency；edge labels 只用于审计，
不影响 patch 选择。

## 3. 必报指标

- single-patch fraction、graphs-with-chain、mean chain edges；
- partial-component fraction、edge/node coverage、edge coverage p10；
- 每种 bond type 的 covered/total edges 与 graph-level mean recall；
- 相邻 patch node histogram 的 TRUE−SHUFFLED cosine margin；
- 相邻 patch newly-covered bond histogram 的 TRUE−SHUFFLED cosine margin；
- 包含稀有第三类 bond 的图中，该 bond 被 Beam8 观察到的图比例。

## 4. 冻结 gate

只有 `s8/o2` 同时满足以下条件才进入 typed classification：

1. single-patch fraction 相对 `s10/o3` 下降至少 20%；
2. graphs-with-chain 不低于 `s10/o3`；
3. mean chain edges 相对提高至少 20%；
4. mean edge coverage 不低于 `s10/o3` 超过 1 point；
5. partial-component fraction不高于 `s10/o3`；
6. 三类 bond 的 aggregate recall 均不低于 `s10/o3` 超过 2 points；
7. node 与 newly-covered-bond 两个 TRUE−SHUFFLED margin 均为正。

gate 通过后，typed classification 只能使用胜出的 geometry；不得根据分类结果切回另一
geometry。若稀有 bond graph recall 低于 0.8，分类结论必须注明第三类键证据不足。

