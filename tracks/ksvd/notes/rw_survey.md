# 随机游走 × 子结构 × KSVD — 文献与定位（v0.2）

> 导师 luyin10：RW 构子结构 → KSVD；少重复、可还原。  
> **v0.2**：加厚角色定位、与主线对照、证据分层；实验细节见 `results/*`。  
> 精读入口见文末。

---

## 0. 我们在问什么

| # | 问题 | 决定什么 |
|---|------|----------|
| Q1 | RW 产出什么「局部结构」？路径 / 诱导子图 / 返回统计 | \(y\) 的定义 |
| Q2 | 如何约束形态？\(p,q,L,r\)、自回避、降权 | 是否「有意义」 |
| Q3 | 如何减轻一阶边冗余、控感受野 | 相对 B0 的动机 |
| Q4 | 如何接到 **可还原** 字典 \(Y\!\approx\!DX\) | 与核/PE/嵌入分界 |
| Q5 | 分层证据：机制 / 表征 / 下游 | 什么叫「有效」 |

**非目标**：DeepWalk 式终点嵌入；仅为 GNN minibatch 加速采样（可借接口）。

---

## 1. 角色对照表（文献定位核心）

别人用 RW **干什么** vs 我们：

| 角色 | 代表 | 输入→输出 | 可还原子图？ | 与组内关系 |
|------|------|-----------|--------------|------------|
| **K. 核指纹** | RW-Kernel, Marginalized kernel | 图→相似度 \(K\) | 否 | 证明「游走=合法结构信号」；我们不走 SVM 终局 |
| **E. 嵌入采样** | DeepWalk, **node2vec** | 多 walk→节点向量 | 否 | 借 **r 条 walk、\(p,q\)**；不要 Skip-gram |
| **S. 训练子图采样** | GraphSAGE, **GraphSAINT** | 图→诱导 \(G[S]\) 训练 GNN | 否（采样为 SGD） | 借 **诱导子图接口**；目的改成 KSVD 的 \(Y\) |
| **P. 位置/结构编码** | **RWPE** 等 | 节点→统计向量 | 否 | **对照基线**（不可还原） |
| **H. 显式子结构** | **GSN**, **CIN** | 手数/环 lift→MP | 计数/cell，非字典原子 | 同「显式结构」；我们是 **数据驱动原子** |
| **D. 组内** | RW→诱导→**KSVD** | patch→\(D,X\)→\(s_v\)/读出 | **是（重构意义下）** | 本轨 |

一句话定位：

> 我们不处在「核 / 嵌入 / PE / 官方 GSN」任一终局，而处在 **S 的采样形态 + E 的 walk 旋钮 + D 的可还原分解** 交叉点。

---

## 2. 文献地图（按角色，含精读链接）

### K — 图核

| 工作 | 要点 | 精读 |
|------|------|------|
| RW / marginalized kernels | 共同游走、标签路径期望 | [rw_graph_kernel.md](../../../docs/literature/deep/rw_graph_kernel.md) |
| WL kernel | 另一类结构指纹 | [wl_graph_kernel.md](../../../docs/literature/deep/wl_graph_kernel.md) |
| Survey 2020 | 谱系地图 | 索引用 |

### E — 嵌入

| 工作 | 要点 | 精读 |
|------|------|------|
| DeepWalk | 均匀 RW + 多 walk + Skip-gram | [deepwalk.md](../../../docs/literature/deep/deepwalk.md) |
| node2vec | \(p,q\) 控局部/外扩 | [node2vec.md](../../../docs/literature/deep/node2vec.md) |

### S — 子图采样

| 工作 | 要点 | 精读 |
|------|------|------|
| GraphSAINT | RW→诱导子图；概率归一 | [graphsaint.md](../../../docs/literature/deep/graphsaint.md) |
| GraphSAGE | 固定邻居预算 | 对照 |

### P — PE

| 工作 | 要点 | 精读 |
|------|------|------|
| RWPE 族 | 返回概率等 | [rwpe.md](../../../docs/literature/deep/rwpe.md) |

### H — 显式结构 GNN

| 工作 | 要点 | 精读 |
|------|------|------|
| GSN | 人设子结构计数 | [gsn.md](../../../docs/literature/deep/gsn.md) |
| CIN | 环 lift + cell MP | [cin.md](../../../docs/literature/deep/cin.md) |

### 覆盖语言（导师「少重复」）

自回避、边降权、邮路/ path cover：**约束词汇**；实现上优先软降权，忌永久删边（氢原子反例）。

---

## 3. 组内方法（表征侧，写死）

### 3.1 管道（节点级主推）

```
对每个节点 v：
  采 r 个 patch（B0 星形 或 从 v 出发的 RW）
  每个 patch → 诱导 G[S] → 邻接 pad 向量 y_i
  共享字典 D（train 上所有 y 学习）
  x_i = OMP(D, y_i)
  s_v = pool_i(x_i)   # mean / max / …
下游：concat / residual / gate 进 GIN，或图级读出
```

### 3.2 与「简单」实现的边界

| 曾用过的简版 | 更完整 |
|--------------|--------|
| r=1 | r≥3～5（DeepWalk/n2v 习惯） |
| 图级 s 广播到所有节点 | **节点级** \(s_v\) |
| 每图独立 D | **共享 D**（train-only） |
| 用 recon 比 B0 vs RW | recon 仅 **同族诊断** |

### 3.3 \(p,q,L,m,r\)

- 语义见 [rw_params.md](rw_params.md)、[node2vec.md](../../../docs/literature/deep/node2vec.md)。  
- **尚未**在 MUTAG 上系统网格；报告中必须写「启发式 / 探针设定」，禁止写成最优超参。

---

## 4. 「有效」的三层证据（必须分表）

| 层 | 问题 | 当前证据 | 强度 |
|----|------|----------|------|
| **A 机制** | 更大感受野能否采到 1-hop 没有的闭包？ | C4 vs C8：B0~0.85，RW~1.0；可视化 | **强** |
| **B 表征** | \(D,X\) 是否对齐、可诊断？ | 共享 D 合成 0.55→0.86；recon 不可跨 B0/RW 比 | **中** |
| **C 下游** | 同协议是否稳定涨点？ | MUTAG Xu：GIN 89.4；结构最好约 +1pt 且不稳 | **弱** |

**禁止**：用 A 代替 C 宣称「方法在分子任务有效」。  
**允许**：用 A+文献定位完成「RW 能否接入 KSVD」的调研结论。

---

## 5. 与 GSN / CIN 的分界（答辩用）

| | GSN / CIN | 组内 RW×KSVD |
|--|-----------|----------------|
| 子结构来源 | 人设环/团等 | 数据驱动字典原子 |
| 结构进入方式 | 计数或 cell 消息传递 | 稀疏编码系数 |
| 可还原 | 弱（计数/隐状态） | 强（\(Y\!\approx\!DX\)） |
| 成熟下游 | 有 molhiv 等表 | **未**同协议竞争 |
| 量级 | 完整 GNN 体系 | 当前为 **结构通道探针** |

对标 CIN 数字仅当 **远期**；近期只对「问题同类：显式结构」。

---

## 6. 开放问题（成体系下一步）

1. 文献：覆盖/自回避与分子 motif 采样是否有现成约束可抄？  
2. 表征：多 walk + 何种 pool 在 **过程指标**（C4 覆盖率）上最优？  
3. 下游：NCI1 / 结构敏感合成 上 C 层是否站得住？  
4. 融合：残差/门控在 **结构信号足够强** 的数据上是否放大增益？  
5. 协议：任何分子主表必须 lock `protocol_id`（Xu vs strict 分表）。

---

## 7. 精读清单

| 优先级 | 文档 |
|--------|------|
| P0 | [node2vec](../../../docs/literature/deep/node2vec.md) · [graphsaint](../../../docs/literature/deep/graphsaint.md) |
| P0 | [deepwalk](../../../docs/literature/deep/deepwalk.md) · [rw_graph_kernel](../../../docs/literature/deep/rw_graph_kernel.md) |
| P1 | [gsn](../../../docs/literature/deep/gsn.md) · [cin](../../../docs/literature/deep/cin.md) · [rwpe](../../../docs/literature/deep/rwpe.md) |
| 组内 | [definition](definition.md) · [rw_params](rw_params.md) · [shared_dict](shared_dict.md) |

---

## 8. 版本

| 版 | 内容 |
|----|------|
| v0.1 | 初版菜单与 R1–R3 |
| **v0.2** | 角色对照表、证据三层、精读链、定位句 |
