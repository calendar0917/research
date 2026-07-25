# WL Graph Kernel（压缩精读）

> 概念：[concepts.md](concepts.md) · 完整旧笔记：`../../../paper/docs/literature/notes/wl_graph_kernel.md`

| 项 | 内容 |
|----|------|
| 标题 | Weisfeiler-Lehman Graph Kernels |
| 出处 | JMLR 2011 · Shervashidze et al. |
| 链接 | https://www.jmlr.org/papers/v12/shervashidze11a.html |

## 一句话

按 **1-WL 重标记** 抽多尺度子树型特征 → **核** 比两图 → SVM 等做图分类（非 MPNN）。

## 四问（摘要）

1. **旧缺陷**：更早图核在大图上贵/慢；需要可扩展的结构相似度。  
2. **关键一步**：迭代 hash(自己 + 邻居多重集) → 各轮直方图 → 核矩阵 → 分类。  
3. **目的**：主战场是 **B 图分类 / 相似度**，不是 >1-WL 专文。  
4. **主线**：经典 Kernel 坐标；与 GSN 都「结构模式 → 表示」，GSN 接可学习 MP 并可冲 >1-WL。

## 边界

WL **测试**与 WL **核**不是同一工具；核不是「低复杂度同构算法」。
