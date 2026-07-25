# CIN / CWN（做法讲解 + 精度 + 协议）

> 组内对标动机：luyin10（molhiv 上预定义 5/6 环）。  
> 本文：**具体做法（含数据侧）** + 数字 + 协议；与 KSVD 距离。

| 项 | 内容 |
|----|------|
| 短名 | CIN（Cell Isomorphism Network；文中亦称 CW Networks） |
| 标题 | Weisfeiler and Lehman Go Cellular: CW Networks |
| 出处 | NeurIPS 2021；[arXiv:2106.12575](https://arxiv.org/abs/2106.12575) |
| 一句话 | 图 **lift** 成 cell complex（分子上常用 **induced cycle = 环作 2-cell**）→ 在点/边/环上 **分层 message passing** |
| 与 GSN | 都显式用环类结构；GSN=**计数拼进** MPNN；CIN=**先抬成拓扑对象再在上面传消息** |

---

## 0. 侧重点：为什么说「偏数据 / 偏分子」

| 点 | 含义 |
|----|------|
| **结构先验来自数据域** | 分子里 **5/6 元环** 化学上常见且任务相关；CIN 主实验方案就是 **ring lifting** |
| **不是从数据「学出」环列表** | 环是 **算法枚举** 的 induced cycle（有长度上限 \(k\)），不是 KSVD 式字典原子 |
| **强结果集中在分子基准** | ZINC / ZINC-FULL / molhiv 明显；TUD 社交集上不一定碾压 GSN |
| **表达力理论 + 分子实证** | CWL 理论一套；卖点表格大量是 **化学图回归/分类** |

导师说的「CIN 在 molhiv 上数 5/6 环」≈ 文中 **\(k=6\) 的 induced cycle 作 2-cell**，不是随便说说。

---

## 1. 具体做法（三步）

### Step A — Lifting：图 → cell complex（**数据预处理，关键**）

把一张普通图 \(G=(V,E)\) **抬升**成 **正则 cell complex** \(X\)：

| 维 | 对象 | 分子直观 |
|----|------|----------|
| **0-cell** | 顶点 | **原子** |
| **1-cell** | 边 | **化学键** |
| **2-cell** | 贴在 **induced cycle（环）** 上的面 | **化学环**（苯环等） |

**Ring lifting（分子主方案）：**

1. 在 \(G\) 上找 **induced cycles**（弦图意义下的环；长度 ≤ 超参 \(k\)）。  
2. 每个这样的环贴一个 **2-cell**（想象：给环「糊一层皮」）。  
3. 得到层次：原子 — 键 — 环，以及它们之间的 **boundary / co-boundary** 关系。

**\(k\) 随数据变（数据相关超参）：**

| 数据 | 典型 \(k\) | 含义 |
|------|-----------|------|
| TUD 分子 | ≤ **6** | 小环为主 |
| molhiv | val 在 {6,8,18} 里选 → **6** | 与 5/6 元环叙事一致 |
| ZINC | ≤ **18** | 允许更大环参与 lift |

**特征初始化（数据侧怎么填）：**

| cell | 常见做法 |
|------|----------|
| 0-cell | 原节点特征（原子类型等；OGB 用 atom encoder） |
| 1-cell | 边特征 / 键类型；或由端点 0-cell 聚合 |
| 2-cell | 环上 0-cell 特征的 **sum 或 mean**（超参） |

> 注意：lifting 是 **确定性预处理**（可并行），不是训练出来的；算力在「枚举环」+ 之后 MP。

### Step B — 在 complex 上定义邻接（消息从哪来）

对每个 cell \(\sigma\)，消息来自几类邻居（文中 boundary / co-boundary / 同维邻接等），直觉：

| 方向 | 分子直觉 |
|------|----------|
| **boundary** | 环 ← 组成它的键/原子；键 ← 两端原子 |
| **co-boundary** | 原子 → 它所在的键；键 → 它所在的环 |
| **同维** | 共享边界的边与边、环与环等 |

这样形成 **多层次、可跨 hop 的结构通路**：信息可沿「原子→键→环→另一键→原子」走，**不必**堆很深的普通 GNN 层才能「看见环对侧」（RingTransfer 合成实验讲这个故事）。

### Step C — Cellular message passing + 读出（CIN）

- 每一层对 **0/1/2-cell 的表示** 做聚合+更新（文中 CIN 用类似 **GIN 式** 的局部聚合）。  
- 多层后，对 complex 做 **readout**（对 cell 表示 sum/mean 等）→ 图级向量 → 分类/回归。  
- 可选：JK、dropout 位置、是否用 coboundary 等（TUD 上网格搜）。

**数据流（分子）：**

```
分子图 G
  → 枚举 induced cycles (len≤k) 贴 2-cell
  → 初始化 h_atom, h_bond, h_ring
  → L 层：在 boundary/co-boundary 结构上消息传递
  → readout → ŷ
```

---

## 2. 和 GSN 差在哪（做法级）

| | **GSN** | **CIN** |
|--|---------|---------|
| 结构对象 | 人设子图 \(H\)，算 **同构/轨道计数** | 环变成 **2-cell 几何/拓扑对象** |
| 怎么进模型 | 计数当 **额外节点/边特征** 喂 MPNN | **在 cell complex 上直接 MP** |
| 环的角色 | 特征维度 | **传消息的「节点」之一**（环也有隐状态） |
| 数据依赖 | 选哪些 \(H\)（三角、环…） | 选 **\(k\)** 与是否 ring-lift |

都「显式环」，但 CIN 更彻底：**环是计算图的一部分**，不是只拼一个计数。

---

## 3. 和组内 KSVD / RW 差在哪

| | CIN | 我们 |
|--|-----|------|
| 子结构从哪来 | **枚举环**（预定义族 + \(k\)） | **采样 patch + 学字典原子** |
| 结构是否可还原 | 环列表可知；表示是 MP 隐状态 | \(Y\!\approx\!DX\) 重构意义 |
| 数据侧重 | **分子环先验很强** | 先验弱，靠数据驱动 |
| 工程量级 | 完整 lifting + 多层 cell GNN | 结构通道探针 |

对标意义：同一 **「分子里环重要」** 的数据故事；**不是**同一实现量级。

---

## 协议要点（横比必读）

### 总览：三套协议，不可混

| 基准 | 协议家族 | 是否「GIN 乐观 10-fold」 | 我们主对标？ |
|------|----------|-------------------------|--------------|
| **TUD**（MUTAG/NCI1…） | **Xu et al. / GIN [74]** | **是** | 辅；勿当 strict 排名 |
| **ZINC / ZINC-FULL** | Dwivedi et al. 固定划分 | 否 | 二期 |
| **ogbg-molhiv** | OGB scaffold 固定划分 | 否 | **主** |

### TUD = 与 GIN 89.4 **同一套评估叙事（乐观族）**

CIN 原文（附录）：

> *On these datasets, we followed the approach in Xu et al. [74], which prescribes to run a **10-fold cross-validation** procedure and report the **maximum of the average validation accuracy across folds**.*

| 项 | CIN / GIN 文内做法 |
|----|---------------------|
| 划分 | 10-fold CV |
| 汇报 | **各折 validation 平均准确率，再取 maximum**（不是 nested test、不是 fair-comparison 式每折独立测集） |
| 超参 | **网格搜索**（batch、hidden、dropout、lr 与 decay、层数、cell 初始化 mean/sum、coboundary、dropout 位置等）→ Table 8 **按数据集一套** |
| 特征 | 0-cell 跟 Xu；高维 cell mean/sum 聚合 0-cell |
| 环 | induced cycle **k≤6** 作 2-cell |
| 训练细节 | 初始 lr + 固定步数 decay；JK + cell/complex readout；与 GIN 文一致风格 |

**因此**：表里 GIN **89.4**（MUTAG）与 CIN **92.7** 是在 **同一协议族** 下的文献数字，**可以互相引用作「同表」**，但：

- 都属于社区常批的 **乐观 / paper 协议**（选 max val、超参按数据集调、测试信息不干净风险）。  
- **不是** 你们 gnn-gsn 轨的 `strict-v0`，也 **不是** Errica et al. *A Fair Comparison* 那类更严设定。  
- **不能** 把 CIN 的 92.7 直接和 strict 下自跑 GIN 80% 横比当「结构方法赢了」。

主文一句：*“training setting and evaluation procedure follow those in Xu et al. [74]”* → 与附录一致。

### ZINC（更干净，仍非 TUD）

| 项 | 内容 |
|----|------|
| 划分 | Dwivedi 预定义 train/val/test |
| 指标 | MAE；early stop 后 test；**10 次随机初始化** mean±std |
| 优化 | batch 128；lr \(10^{-3}\)，val 不升 patience 20 则 ×0.5；lr<\(10^{-5}\) 停 |
| 环 | **k≤18** |
| 模型 | 全文 4 层 hidden 128；small ≈2 层 hidden 48（~100k 参） |

### Mol-HIV（主对标协议）

| 项 | 内容 |
|----|------|
| 划分 | OGB **scaffold** 固定 train/val/test |
| 指标 | **test ROC-AUC @ best validation epoch** |
| 重复 | **10** 次 weight init，报 mean±std |
| 环 k | 在 {6,8,18} 里 **按 val 选 k=6**（已有模型选择） |
| 架构超参 | **照抄 HIMP**：2 层、dropout 0.5、hidden 64、lr \(10^{-4}\)、batch 128、150 epoch |
| 小模型 | hidden 48 |
| 特征 | OGB atom/bond encoder → 0/1-cell；2-cell 类似 ZINC |

相对 TUD：**固定 scaffold 测集** 更接近现代标准；仍有 **k 与 epoch 按 val 选**（常见，但要在笔记里写明）。

### CSL

Dwivedi：**5-fold CV + 20 种初始化** 等（表达力合成集，另表）。

---

**禁止**：TUD Acc ↔ molhiv AUC ↔ ZINC MAE 混排；TUD 乐观数字 ↔ 本组 strict 数字混排。

---

## 主结果（论文 Table 摘录）

### 1) TUD 分类 · Accuracy %（Table 2 摘录）

| Dataset | GIN | GSN | **CIN** |
|---------|-----|-----|---------|
| MUTAG | 89.4±5.6 | 92.2±7.5 | **92.7±6.1** |
| PTC | 64.6±7.0 | 68.2±7.2 | **68.2±5.6** |
| PROTEINS | 76.2±2.8 | 76.6±5.0 | **77.0±4.3** |
| NCI1 | 82.7±1.7 | 83.5±2.0 | **83.6±1.4** |
| NCI109 | — | — | **84.0±1.6** |
| IMDB-B | 75.1±5.1 | **77.8±3.3** | 75.6±3.7 |
| IMDB-M | 52.3±2.8 | **54.3±3.3** | 52.7±3.1 |
| RDT-B | 92.4±2.5 | — | **92.4±2.1** |

文称：8 个里多 top/近 top；**分子/生化**更好。社交上 GSN 有时更高 → 环归纳偏置非处处最优。
### 2) ZINC · MAE ↓ / MolHIV · ROC-AUC ↑（Table 3）

| Method | ZINC（无边特征） | ZINC（有边特征） | ZINC-FULL | **MOLHIV ↑** |
|--------|------------------|------------------|-----------|--------------|
| GCN | 0.469±0.002 | N/A | N/A | 76.06±0.97 |
| GIN | 0.408±0.008 | 0.252±0.014 | 0.088±0.002 | 77.07±1.49 |
| PNA | 0.320±0.032 | 0.188±0.004 | N/A | 79.05±1.32 |
| DGN | 0.219±0.010 | 0.168±0.003 | N/A | 79.70±0.97 |
| HIMP | N/A | 0.151±0.006 | 0.036±0.002 | 78.80±0.82 |
| **GSN** | **0.139±0.007** | **0.108±0.018** | N/A | **77.99±1.00** |
| **CIN-small** | **0.139±0.008** | **0.094±0.004** | **0.044±0.003** | **80.55±1.04** |
| **CIN** | **0.115±0.003** | **0.079±0.006** | **0.022±0.002** | **80.94±0.57** |

要点（导师口述对齐）：

- molhiv 上 CIN ~**80.9 ROC-AUC**，GSN ~**78.0**，GIN ~**77.1**（论文表内）。  
- ZINC 上 CIN 相对 GSN/PNA 等 **MAE 明显更低**（环 lift k=18）。  
- **small（~100k 参）** 仍强，说明不纯靠堆参。

### 3) 合成（叙事用，非主表）

| 任务 | CIN |
|------|-----|
| CSL | 100.0±0.0（表中） |
| SR 族区分 | k=6 时文称可消歧所有对 |
| RingTransfer | 3 层即可跨环传信息；深 GIN 易崩 |

---

## 与组内 KSVD / RW 的距离

| CIN | 我们 |
|-----|------|
| 预定义 **环** 作 2-cell | 数据驱动 **字典原子**；RW 采诱导子图逼近「重要局部」 |
| 在 complex 上 **消息传递** | 稀疏编码 + readout（可后融 GNN） |
| 分子：k=6（HIV）/18（ZINC） | RW 的 \(L,m,q\) 需服务「环尺度」而不爆成半图 |

**对标策略（建议）**

1. **主对标数据**：`ogbg-molhiv`（AUC）+ 可选 `ZINC`（MAE）；TUD 仅作辅。  
2. **基线引用**：表内 GIN / GSN / CIN 数字可作 **文献对照**；自跑必须同协议。  
3. **阶段仍是**：先 L1–L4 采样+KSVD 管道 → 再 molhiv 小模型探路；勿一上来复现 CIN 全训练。

## 口述 3 点

1. CIN 把 **环** 抬成 cell 再传消息，分子任务很强。  
2. 论文 molhiv **≈80.9 AUC**，ZINC MAE **≈0.079**（有边特征、全文设定）。  
3. 我们比的是「**数据驱动结构**能否接近/解释这类增益」，不是先抄 complex 消息传递。
