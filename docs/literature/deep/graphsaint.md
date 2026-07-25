# GraphSAINT（P0 精读）

> 概念：[concepts.md](concepts.md) · 组内用途：[rw_survey.md](../../../tracks/ksvd/notes/rw_survey.md) §R2

| 项 | 内容 |
|----|------|
| 短名 | GraphSAINT |
| 标题 | GraphSAINT: Graph Sampling Based Inductive Learning Method |
| 作者 / 出处 | Zeng, Zhou, Srivastava, Kannan, Prasanna；ICLR 2020 |
| 链接 | [arXiv:1907.04931](https://arxiv.org/abs/1907.04931) · [code](https://github.com/GraphSAINT/GraphSAINT) |
| 层级 | C（采样接口 + 无偏归一；非 GCN 主线） |
| 日期 | 2026-07-24 |

## 一句话

**先对训练图采样得到子图，再在子图上建完整 GCN 做 minibatch**——用图采样替代逐层邻居采样，并做 **聚合/损失归一** 消偏。

## 四问

### 1. 旧方法缺陷？

- 深层 GCN **邻居爆炸**；GraphSAGE 等 **layer sampling** 限制每层邻居数，仍贵、层间连接易碎。  
- FastGCN 等层间独立采样：batch 过稀、精度伤。  
- Cluster-GCN 等子图训练：常 **启发式**，**不等概率** 采样节点/边 → 估计有偏，文中点名需归一。

### 2. 关键一步 / 假设与代价？

**视角翻转**：

```
旧：全图 GCN → 按层采节点/边 → 反传
新：SAMPLE(G) → Gs → 在 Gs 上完整 GCN → 反传
```

| 组件 | 作用 |
|------|------|
| **SAMPLE** | 节点 / 边 / **随机游走** / 多维 RW 等；返回 **诱导子图** \(G[S]\)（多连边，助收敛） |
| **\(\alpha_{u,v}\)** | 聚合归一：\(\alpha_{u,v}=p_v/p_{u,v}\)（无偏聚合估计） |
| **\(\lambda_v\)** | 损失归一：\(\lambda_v=\|V\|\,p_v\)，batch 损失期望对齐全图均值 |
| 预处理 | 反复 SAMPLE 估 \(p_v,p_{u,v}\)（计数）；子图可复用为 batch |

**RW 采样器（文中）**：\(r\) 个均匀根，各走 \(h\) hop；动机：忽略激活时 \(L\) 层 ≈ \(\tilde A^L\)，\(B_{uv}\) 像 \(L\) 步着陆概率，RW 是可实现近似。另有边采样：\(p_e \propto 1/\deg(u)+1/\deg(v)\)（低度相连更「互相重要」）。

**假设**：子图内传播足以支撑表示；拓扑定义的「影响」够用（未用属性联合采样）。  
**代价**：预处理估概率；采样仍随机；归纳设定下测节点训练时不可见。

### 3. 目的？（A / B / C）

| 标签 | 本文 |
|------|------|
| **C** | 大图 **归纳节点分类** 的高效训练（PPI / Reddit / Yelp / Amazon…） |
| 非 A | 无 WL 表达力理论 |
| 非组内 B 主表 | 不是图级结构字典论文 |

### 4. 与主线 Kernel–GNN–GSN–KSVD？

| | 关系 |
|--|------|
| GNN | 训练系统文；可与 GAT / JK 等架构组合 |
| GSN | 都可能用子图，但 GSN 是 **同构计数注入**，这里是 **SGD 子图** |
| **KSVD 轨** | **接口同构**：`SAMPLE → (Vs, Es) 诱导 Gs → 向量化 → KSVD`。借 **RW 根+长度、诱导、覆盖概率思想**；**不**训练 GCN，**不**需要 \(\alpha,\lambda\) 的 SGD 无偏（KSVD 是分解不是 mini-batch 梯度估计）——但「估 \(p_e\) / 降权已采边」可借鉴覆盖控制 |

## 数据流

```
G ──预处理──► 估 pv, pe；α, λ
   loop:
     Gs ← SAMPLE(G)          # node / edge / RW…
     在 Gs 上完整 L 层 GCN
     前向（边聚合 /α）→ 损失（/λ）→ 反传
```

组内最小对接（不做 GCN）：

```
Gs ← RW-SAMPLE(G; r, h, 可选边降权)
y  ← Vectorize(Gs)           # 定长；排序约定
Y  堆列 → KSVD(Y) → readout
```

## 边界

- 不证明子图结构可还原或 >1-WL。  
- 最优边概率推导含简化（去激活、独立边）；实践是启发式采样器族。  
- 大图节点分类 SOTA 数字 **不可** 横比进我们的图分类 / KSVD 表。  
- 诱导步骤会加边：覆盖统计要以 **Es（诱导后）** 还是 **轨迹边** 为准，组内要写死。

## 口述 3 点

1. 别一层层抠邻居，**先抠一张小子图再跑完整网络**。  
2. 采得不均就 **归一**：节点/边进 batch 的概率要进公式。  
3. 我们要的是 **SAMPLE 出 \(G[S]\)**，后面接 KSVD 不是 GCN。

## 对 RW-C0 的直接建议

| GraphSAINT 概念 | 组内用法 |
|-----------------|----------|
| 根数 \(r\)、长度 \(h\) | 与 n2v 的 \(r,l\) 同类；先小网格 |
| 诱导子图 | R2 默认开 |
| 预处理计数 \(C_e\) | 可改成在线 \(w_e \leftarrow \gamma w_e\) 降权（软覆盖） |
| \(\alpha,\lambda\) | **KSVD 阶段可不用**；若未来「可学习读出」再议 |
| 边 \(p_e\propto 1/d_u+1/d_v\) | 可作起点/边偏好，保护低度关键边 |
