# 概念 · 谱系 · 对照（压缩）

> 全局唯一概念入口。精读：[gsn.md](gsn.md) · [wl_graph_kernel.md](wl_graph_kernel.md)  
> 组内 KSVD 管道细节：[tracks/ksvd/notes/definition.md](../../../tracks/ksvd/notes/definition.md)

## 1. 为何要结构 / 子图

普通 GNN 多聚合邻居**特征**，对局部子结构多是**隐式**；在强依赖结构的任务上常不够。  
结构类方法显式引入信号：Kernel、GSN、KSVD 等——要分清差别，不是只说「加结构」。

## 2. 1-WL · 表达力 · Acc

- **1-WL**：按「自己 + 邻居标签多重集」重标记；不同 → 不同构；相同 → 未必同构。  
- **WL 测试 ≠ WL Graph Kernel**（后者用 WL 抽特征做相似度/分类）。  
- 一类 **MPNN ≤ 1-WL**；GIN 是达上界的一种聚合设计；超 1-WL 要 GSN 等。  
- **表达力**（分得开？）≠ **Acc**（某协议上标签对了多少）。更强表达力不保证更高 Acc。

| 标签 | 在问 | 典型 |
|------|------|------|
| **A 表达力** | 分得开吗 | 合成图、WL 硬例 |
| **B 图分类** | 类别对不对 | TUD 等 Acc |
| **C 其它** | 回归/OGB… | 协议不同不横比 |

## 3. 谱系（口述）

| 方法 | 一句话 |
|------|--------|
| **Kernel** | 预定义结构特征（如 WL 直方图）+ 核 + SVM → 图分类 |
| **GNN/MPNN** | 端到端邻域聚合；结构多隐式；通常 ≤1-WL |
| **GSN** | 人设子结构 H → 同构/轨道计数 → 注入 MPNN；可 >1-WL |
| **KSVD** | 局部结构信号稀疏字典：\(Y \approx DX\)；原子**从数据分解**，可重构；字典大小/稀疏度手设；组内基线多为**一阶邻域**列 |

### 自查误解

| 误解 | 纠正 |
|------|------|
| Kernel = 更快的同构算法 | 测试 vs 核是两件事 |
| 核主用途是判是否同图 | 主用途是相似度 → 分类 |
| Acc = 表达力 | 分维，见上 |
| GSN 只是 GNN 小补丁 | 把手选结构请回；权衡不是单行进化 |
| KSVD 字典事先固定且跨图相同 | \(D,X\) 交替学；跨图对齐是约定问题 |
| 一阶 KSVD 已无感受野问题 | 组内：>1-WL 但受 hop 上限；扩大感受野是本轮课题 |

## 4. 对照表

| 维度 | GNN | WL-Kernel | GSN | KSVD |
|------|-----|-----------|-----|------|
| 结构怎么进 | 隐式聚合 | WL 特征 + 核 | 轨道计数 → MPNN | 字典学习 \(DX\) |
| 子结构 | 基本不显式 | WL 过程预定义 | **人设** H | **学出的**原子 |
| 可还原？ | 否 | 否 | 否（计数） | **是**（重构） |
| 典型目标 | 下游；≤1-WL | 图分类 | 可 >1-WL + 下游 | 显式结构表征 |
| 主要代价 | 表达力天花板 | 绑 1-WL | 手选 H、计数成本 | 字典大小、稀疏度、感受野 |
| 一句话 | 聚特征 | WL 特征 + 核 | 计数 + GNN | 字典 → 可还原结构 |

**GSN ↔ KSVD**：都做显式结构；GSN=预设子图计数，KSVD=分解学原子。  
**Kernel ↔ KSVD**：都是「核心结构 → 表征 → 分类」；一个核+SVM，一个稀疏编码+下游网络。

## 5. 精读

| 篇 | 文件 |
|----|------|
| GSN | [gsn.md](gsn.md) |
| WL-Kernel | [wl_graph_kernel.md](wl_graph_kernel.md) |
| node2vec | [node2vec.md](node2vec.md) |
| DeepWalk | [deepwalk.md](deepwalk.md) |
| GraphSAINT | [graphsaint.md](graphsaint.md) |
| RW Graph Kernel | [rw_graph_kernel.md](rw_graph_kernel.md) |
| RWPE（对照） | [rwpe.md](rwpe.md) |
| GSN | [gsn.md](gsn.md) |
| CIN / CWN | [cin.md](cin.md) |
| **GNN+Subgraph GNN 近期脉络（2023–26）** | [gnn_recent_survey.md](gnn_recent_survey.md) |
| HOD-GNN（导数表达力） | [hod-gnn.md](hod-gnn.md) |
| 组内 RW 定位 | [rw_survey v0.2](../../../tracks/ksvd/notes/rw_survey.md) |
| 新篇 | 复制 [_template.md](_template.md) |
