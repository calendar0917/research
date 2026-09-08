# GSN（压缩精读）

> 概念：[concepts.md](concepts.md) · （旧仓完整笔记 `paper/docs/literature/notes/gsn.md` 随旧仓遗失，本文为现存权威版）

| 项 | 内容 |
|----|------|
| 标题 | Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting |
| 出处 | arXiv:2006.09252；TPAMI 2023 |
| 代码 | https://github.com/gbouritsas/graph-substructure-networks |

## 一句话

选定小子结构 → 子图同构 + **轨道计数** → 结构特征注入 MPNN → 表达力可 **>1-WL**。

## 四问（摘要）

1. **旧缺陷**：MPNN 对子结构多隐式；≤1-WL，难可靠计数多数子结构。  
2. **关键一步**：人设 \(\mathcal{H}\) → 匹配 → orbit 计数（GSN-v 节点 / GSN-e 边）→ 结构感知消息传递。代价：手选 H、计数开销。  
3. **目的**：文内同时有 A 表达力与 B 下游；**数字分表不混比**。  
4. **主线**：与 Kernel 都重模式；与 KSVD 都显式结构，但 GSN 预设计数、KSVD 学字典。

## 边界

- 子结构仍常手选；「数什么」是设计问题。  
- 官方复现与本仓简化实现不可混称同一方法主结论（见 gnn-gsn 轨）。
