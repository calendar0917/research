# ENZYMES 离散 canonical / 连续 content Beam8 不变性审计

> 日期：2026-08-14  
> 目的：确认 split5/6 负结果不是连续属性接入破坏 node relabel invariance 所致。

- 64 graphs × 3 random node permutations；
- canonicalization 和 automorphism orbits 只读取3维离散 labels；
- 18维连续属性与离散 labels 一同随节点置换，只进入 patch content；
- required gates：token rows、token multiset、compact readout 全部不变，orbit-safe node incidence
  映回原节点后等变；
- exact chain node IDs 只作为 diagnostic，因为离散 attributed automorphism 内可有等价选择。
