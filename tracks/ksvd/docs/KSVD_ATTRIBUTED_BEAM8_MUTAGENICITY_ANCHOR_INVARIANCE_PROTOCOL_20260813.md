# Attributed Beam8/Mutagenicity anchor invariance protocol

> 日期：2026-08-13  
> 状态：typed-anchor classification 解释前的强制实现审计。

- 取前 512 张 Mutagenicity 图，每图 3 个固定随机 node permutations；
- 原图与重编号图都使用固定全数据 `edge_dim` 构造 attributed s8/o2 BASE，再添加 typed anchor；
- 分别检查 BASE 与 ANCHOR 的 ordered token rows、token multiset、compact readout；
- concrete node sets 映射后的 exact match 只作 automorphism 诊断，不作为表示 gate；
- BASE/ANCHOR 的 token rows、multiset、compact readout match rate 必须全部等于 1.0，
  否则 attributed typed classification 只能视为 invalidated diagnostic。
