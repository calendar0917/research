# Attributed Beam8 node-incidence feasibility protocol

> 日期：2026-08-14  
> 状态：分类标签不可见；在设计 Beam8→GINE 融合前冻结。

## 1. 研究问题

图级 BAG/compact readout 会在融合前丢掉“哪个节点属于哪个 patch”。本轮不训练分类器，
只检查能否把 attributed `s8/o2 Beam8` patch token 通过 node–patch incidence 稳定回写到
原图节点，形成可接入 GINE 的局部结构位置编码。

## 2. 固定构造

- 数据：完整 TU Mutagenicity；不使用 graph labels；
- patch：已通过 representation invariance 的 attributed `s8/o2/Beam8/R1/m1.5 BASE`；
- patch token：canonical atom-slot + typed bond-slot vector，拼接 patch atom histogram；
- node incidence feature：对所有包含节点 `v` 的 patch token 做 mean/max；拼接 canonical slot
  distribution、center rate、patch position mean、patch relation-degree mean、incidence count/coverage；
- relation degree 包含 previous、following、all-overlap 与 slot-persistence 四个固定关系通道。
- direct incidence 另做诊断；正式候选将 direct feature 在 atom-colored/bond-typed 全图
  automorphism orbit 内求均值。GNN 不应人为区分同一 attributed orbit 内的节点。

未覆盖节点取零结构特征；后续分类中原始 node/edge attributes 仍由 GINE 主干读取，因此零值
不是删除节点。

## 3. 无标签指标

- node coverage、full-covered graph fraction；
- 每节点 incidence 数、multi-patch node fraction；
- 被 chain-active patch 覆盖的 node fraction；
- 在 atom type 相同的节点对中，incidence feature 能进一步区分的 pair fraction。

## 4. Relabel-equivariance gate

取前 512 张图，每图 3 个固定随机 node permutations。新编号节点 `i` 对应原节点
`permutation[i]`。要求：

- orbit-safe node incidence matrix mapped exact match rate = 1.0；
- graph sum/max readout match rate = 1.0。

若 node-equivariance 失败，不得直接把 incidence feature 拼接/FiLM 到 GINE；必须先对
automorphism tie 做 orbit/symmetry-safe 聚合。若通过，才冻结低容量分类协议。
