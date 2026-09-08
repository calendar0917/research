# GNN 与 Subgraph GNN · 近年研究脉络（2023–2026）

> 范围：**GNN/MPNN** 与 **Subgraph GNN** 两条线近三年（约 2023–2026）的研究脉络梳理。
> 用途：综述草稿的「发展历程 + 任务范围」两段式材料；对应 luyin13 中导师布置的两件事（补近两年文献 + 每大类写一段总结）。
> 立场：**只收录经过核实的条目**；每篇给出 arXiv ID（逐条打开 abs 页核对标题/作者/日期）、venue（Crossref/DBLP/正文 Comments 核对）、官方代码（GitHub 逐一确认存在）与协议要点。**没有核实到的一律不写，或显式标注「未核实」。**
>
> 整理日期：2026-08-07 · 整理人：陈昱（负责 GNN + Subgraph GNN 两块）
>
> 关联：[concepts.md](concepts.md) · [gsn.md](gsn.md) · [cin.md](cin.md) · 分类表 `docs/literature/paper_classify.xlsx`（**v3 已回填：`paper_classify_v3.xlsx`，新增 13 条均为 2025–2026 顶会收录论文**）

---

## 0. 一句话结论（脉络总览）

**GNN 线**：主战场从「怎么把 1-WL 天花板捅破」转向「捅破之后怎么不崩、怎么训得动、怎么大规模」，并叠加**图基础模型/LLM** 的新范式冲击；2023–2024 理论密集（RNI、谱不变、同态计数、margin 上的表达力），2024–2026 工程与批判并行（过压缩/过平滑重连、state-space 序列建模、表达力基准与负面结果、基础模型综述）。

**Subgraph GNN 线**：先完成**表达力层级公理化**（SSWL 六级层次、乘积图/粗化统一框架、与 2-FWL 的差距），再集中解决**效率-表达力权衡**（子图采样、中心性、冗余消除、Subgraph-to-Node），最后被**链路预测子图范式**（SEAL→NBFNet→ELPH）和**子图 Transformer**（Subgraphormer）收编为工程可用的通用组件。与组内 KSVD/GSN 主线高度相关（详见 §4）。

---

## 1. GNN / MPNN 近期脉络（2023–2026）

### 1.1 表达力理论：从「≤1-WL」到「怎么越界」

传统 MPNN 的表达力上界是 **1-WL**（GIN 达上界）。近三年该线的核心问题是：**越界的手段是什么、代价多少、什么时候值得**。

| 手段 | 代表工作（已验证） | 核心论断 |
|------|--------------------|----------|
| **随机节点初始化 (RNI)** | *The Surprising Power of GNNs with Random Node Initialization* · IJCAI 2021 · [arXiv:2010.01179](https://arxiv.org/abs/2010.01179) | 随机噪声 + 平均嵌入可将 MPNN 推到 2-WL 之上（早期奠基石） |
| **谱不变特征** | *On the Expressive Power of Spectral Invariant GNNs* · ICML 2024 · [arXiv:2406.04336](https://arxiv.org/abs/2406.04336) | 用图谱（特征值/特征向量模）构造不变特征 → 严格 >1-WL；代价是全局特征不再局部 |
| **可学习约束投影** | *Boosting GNN Expressivity with Learnable Lanczos Constraints* · arXiv 2024（venue 未核实）· [arXiv:2408.12334](https://arxiv.org/abs/2408.12334) | 把 Lanczos 三对角化做进消息传递，用可学习约束提升表达力 |
| **同态计数编码** | *Homomorphism Counts as Structural Encodings for Graph Learning* · ICLR 2024 · [arXiv:2410.18676](https://arxiv.org/abs/2410.18676) | 节点同态计数作结构编码；理论定界 + 分子/合成任务实证（与组内 GSN 轨道计数同族） |
| **定量刻画（非二值）** | *Beyond Weisfeiler-Lehman: A Quantitative Framework for GNN Expressiveness* · ICLR 2024 Oral · [arXiv:2401.08514](https://arxiv.org/abs/2401.08514) | 用「同态计数距离」定量衡量两个图有多难分，指出 1-WL 失败图间的真实距离；官方仓库 `subgraph23/homomorphism-expressivity` |
| **什么时候值得越界** | *Weisfeiler-Leman at the margin: When more expressivity matters* · ICML 2024 · [arXiv:2402.07568](https://arxiv.org/abs/2402.07568) | 只有「距分类边界很近」时高表达力才在真实数据上兑现；合成 + 小 TUD |
| **核视角回看** | *Generalizing Weisfeiler-Lehman Kernels to Subgraphs* · ICLR 2025 · [arXiv:2412.02181](https://arxiv.org/abs/2412.02181) | 把 WL 核推广到子图特征 → 把「子图信号」接回经典核方法 |
| **超越 MP 的结构编码** | *Walking Out of the Weisfeiler-Leman Hierarchy* · TMLR 2023 · [arXiv:2102.08786](https://arxiv.org/abs/2102.08786) | 距离/随机游走结构编码（AWE、ShortestPath）即可越过 1-WL；后来是 Graph Transformer 位置编码的源头之一 |

**脉络小结（发展历程）**：2021–2023 在找「越界手段」（RNI、谱、计数、结构编码）并逐个证明严格越界；2024 出现**定量化**（不是能不能分，而是难分程度）与**场景化**（margin 上才有用）的收敛，标志着理论从「存在性」进入「实用性」；2025 用**核/谱**语言重新整合，并接受「表达力 ≠ 精度」（见 §1.6 负面结果）。

### 1.2 过平滑 / 过压缩与长程依赖

深/图变大的两个结构病：**过平滑**（节点表示趋同）与**过压缩**（长程信息挤不过瓶颈边）。2023 是「过压缩」的爆发年，2024–2026 转向「修图/换架构」工程解与批判性盘点。

| 工作（已验证） | 要点 |
|----------------|------|
| *Understanding Oversquashing in GNNs through the Lens of Effective Resistance* · **ICML 2023** · [arXiv:2302.06835](https://arxiv.org/abs/2302.06835) | 把过压缩量化成**有效电阻**（总有效电阻 / 敏感度），证明「瓶颈 ⇒ 大有效电阻」，为诊断和重连给出可算指标 |
| *Graph Rewiring in GNNs to Mitigate Over-Squashing and Over-Smoothing: A Survey* · arXiv 2024 · [arXiv:2411.17429](https://arxiv.org/abs/2411.17429) | 综述：重连（rewiring）方法论全家桶与评测协议 |
| *Are Graph Transformers Necessary? …Fractal Nodes in MPNNs* · **AAAI 2026** · [arXiv:2511.13010](https://arxiv.org/abs/2511.13010) | 「长程不一定要 Transformer」——分形虚拟节点把局部 MPNN 撑到长程；官方仓库 `jeongwhanchoi/MPNN-FN` |
| *Message-Passing State-Space Models* · arXiv 2025（venue 未核实）· [arXiv:2505.18728](https://arxiv.org/abs/2505.18728) | 用**状态空间/序列建模**（Mamba 族）替代部分消息传递，规避过压缩 |
| *Asynchronous Message Passing for Addressing Oversquashing in GNNs* · arXiv 2025 · [arXiv:2509.06777](https://arxiv.org/abs/2509.06777) | 已入分类表：按中心性异步更新节点，不改结构不增参 |
| *Position: Don't be Afraid of Over-Smoothing And Over-Squashing* · arXiv 2026（预印）· [arXiv:2601.07419](https://arxiv.org/abs/2601.07419) | 批判立场文：这两个「病」在正确初始化/宽度下未必致命，别一味修 |

**脉络小结**：2023 提供**诊断量**（有效电阻、灵敏度），2024 提供**修法综述**（重连家族），2025–2026 出现「**不一定需要修**」的立场文 + 「**用序列/虚拟节点换长程**」的替代架构。任务范围：图分类/回归、链路预测、大规模图。

### 1.3 结构编码与谱方法（承接主线）

与组内「结构如何被编码」直接相关。除 §1.1 的谱不变（ICML 2024）与同态计数（ICLR 2024）外，本条补充方向指针：

- 经典锚点：LapPE / RWPE / AWE（GraphGPS 之前）→ 近三年把这些编码**理论化**（哪些编码能把 MPNN 顶到几 WL，如 *Walking Out…*、谱不变工作均给出严格定界）。
- 注意：这些工作基本都依赖**一次性预处理**（谱分解/计数），与组内 KSVD「采样 patch + 学字典」的**数据驱动结构**形成对照——预定义 vs 学出的张力正是综述要写的点（见 §4）。

### 1.4 架构新范式：Graph Transformer 与序列模型（综述/盘点）

| 工作（已验证） | 要点 |
|----------------|------|
| *Graph Transformers: A Survey* · **TNNLS 2025** · [arXiv:2407.09777](https://arxiv.org/abs/2407.09777)（DOI 已核实） | 系统盘点 GT 的位置/结构编码、注意力、可扩展性 |
| *A Survey of Graph Transformers: Architectures, Theories and Applications* · arXiv 2025 · [arXiv:2502.16533](https://arxiv.org/abs/2502.16533)（ACM DOI 已核实） | 偏理论：GT 的表达力层级与「MPNN+GT」混合设计 |

**任务范围**：图回归（ZINC/PCQM）、图分类（OGB/Peptides）、节点分类（PATTERN/CLUSTER）；大图可扩展性仍是短板。

### 1.5 图基础模型与 LLM（2024–2026 新范式）

| 工作（已验证） | 要点 |
|----------------|------|
| *Graph Foundation Models: A Comprehensive Survey* · arXiv 2025 · [arXiv:2505.15116](https://arxiv.org/abs/2505.15116) | 「图基础模型」综述（预训练-适配-下游）；体系化任务范围 |

> 诚实标注：本块我只核实到这一篇综述；具体方法类工作（GraphGPT、ZeroG、GraphAgent 等）**本次未逐篇核实**，下一轮补。LLM+图、图基础模型的「方法级」条目不宜在此轮强塞。

### 1.6 基准与负面结果（2024–2026 批判期）

| 工作（已验证） | 要点 |
|----------------|------|
| *OpenGLT: A Comprehensive Benchmark of GNNs for Graph-Level Tasks* · arXiv 2025 · [arXiv:2501.00773](https://arxiv.org/abs/2501.00773) | 图级任务统一基准，暴露大量方法在统一协议下排名洗牌 |
| *Convexified Message-Passing Graph Neural Networks* · **AISTATS 2026** · [arXiv:2505.18289](https://arxiv.org/abs/2505.18289) | 已入分类表：把 MP 训练凸化，求全局最优 + 泛化界（理论类） |

**脉络小结（1.4–1.6 合并）**：2024–2026 GNN 线的「叙事回归」——GT/序列模型被证明未必比好的 MPNN 强（Fractal Nodes：AAAI 2026），统一基准（OpenGLT）与凸化理论（AISTATS 2026）都在收「表达力神话」的尾。

---

## 2. Subgraph GNN 近期脉络（2023–2026）

### 2.1 表达力层级公理化（2023 年是分水岭）

| 工作（已验证） | 要点 |
|----------------|------|
| *A Complete Expressiveness Hierarchy for Subgraph GNNs via Subgraph Weisfeiler-Lehman Tests* · **ICML 2023** · [arXiv:2302.07090](https://arxiv.org/abs/2302.07090) | **里程碑**：SWL 测试给节点级子图 GNN 建**六级严格层级**；SSWL 达到最大表达力；与 2-FWL 之间有固有差距；官方仓库 `subgraph23/SWL` |
| *From Relational Pooling to Subgraph GNNs: A Universal Framework* · **ICML 2023** · [arXiv:2305.04963](https://arxiv.org/abs/2305.04963) | FPS：把关系池化（RP）与子图 GNN 统一进一个框架，给出**最一般**子图表达力表达 |
| *Distance-Restricted Folklore WL GNNs with Provable Cycle Counting Power* · **NeurIPS 2023** · [arXiv:2309.04941](https://arxiv.org/abs/2309.04941)（Crossref DOI=NeurIPS'23 已核实） | 距离受限的 Folklore WL：在不做完整高阶的前提下拿到**环计数**能力 |
| *On the Expressive Power of Subgraph GNNs for Graphs with Bounded Cycles* · arXiv 2025（venue 未核实）· [arXiv:2502.03703](https://arxiv.org/abs/2502.03703) | 环长有界图上子图 GNN 表达力的精细刻画 |
| *Ordered Subgraph Aggregation Networks* · **NeurIPS 2022** · [arXiv:2206.11168](https://arxiv.org/abs/2206.11168) | OSAN：有序子图聚合，把「子图 + 顺序」顶到更高表达力 |

**脉络小结**：2022（SUN/ESAN/GNN-AK/NGNN）还是「各做个的 + 各证各强」；**2023 统一**——SSWL 给出六级层级、FPS 给统一框架、距离受限 FWL 给环计数上界。此后新理论工作基本都在该公理化框架内「填空」或「求更细粒度」（bounded cycles）。

### 2.2 统一框架：乘积图 / 粗化 / 与 Transformer 融合

| 工作（已验证） | 要点 |
|----------------|------|
| *A Flexible, Equivariant Framework for Subgraph GNNs via Graph Products and Graph Coarsening* · **ICML 2024** · [arXiv:2406.09291](https://arxiv.org/abs/2406.09291) | 子图 GNN = 原图 × 子图索引图的**乘积图**上消息传递；用图粗化统一已有子图方案；官方仓库 `BarSGuy/Efficient-Subgraph-GNNs` |
| *Subgraphormer: Unifying Subgraph GNNs and Graph Transformers via Graph Products* · **ICML 2024** · [arXiv:2402.08450](https://arxiv.org/abs/2402.08450) | 在上述乘积图上加**稀疏注意力 + 乘积图位置编码**，子图 GNN 与 GT 合流；官方仓库 `BarSGuy/Subgraphormer`（已入分类表） |

**脉络小结**：2024 年是「统一」之年——「子图 GNN 不过是乘积图上的一种特化」这一观察，把三条线（子图 GNN、Graph Transformer、图粗化）并到一起，并给出高效的乘积图位置编码/稀疏注意力。

### 2.3 效率与可扩展性（表达力 vs 算力）

| 工作（已验证） | 要点 |
|----------------|------|
| *Translating Subgraphs to Nodes (S2N)* · **ICML 2024** · [arXiv:2204.04510](https://arxiv.org/abs/2204.04510) | 子图 → 节点，把子图级任务转成节点级，省内存省算力；官方仓库 `dongkwan-kim/S2N`（已入分类表） |
| *Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality (HyMN)* · **ICML 2025** · [arXiv:2501.03113](https://arxiv.org/abs/2501.03113) | 用**游走中心性**做子图采样 + 当结构编码，理论证「采样与编码互补」；官方仓库 `jks17/HyMN`（已入分类表） |
| *Exact Acceleration of Subgraph GNNs by Eliminating Computation Redundancy* · **Frontiers of Computer Science 2026** · [arXiv:2412.18125](https://arxiv.org/abs/2412.18125) | 把子图 GNN 的重复计算（各根子图重叠）**精确消冗余**，不改精度只提速 |

**脉络小结**：子图 GNN 拿表达力换算力的老痛点，2024–2026 用三条路解：**换表示**（S2N 把子图当节点）、**采样**（HyMN 中心性采样）、**消冗余**（Exact Acceleration）。任务范围：图分类（molhiv/molbace/Peptides）、图回归（ZINC）、子图级任务、大图（MalNet-Tiny/RDT）。

### 2.4 链路预测：子图范式（SEAL → NBFNet → ELPH → 批判）

| 工作（已验证） | 要点 |
|----------------|------|
| *Link Prediction Based on Graph Neural Networks (SEAL)* · **IJCAI 2018** · [arXiv:1802.09691](https://arxiv.org/abs/1802.09691) | 开山：把每条候选边包一个**enclosing subgraph**，子图标签+GNN → 链路预测 |
| *Neural Bellman-Ford Networks (NBFNet)* · **NeurIPS 2021** · [arXiv:2106.06935](https://arxiv.org/abs/2106.06935) | 用**路径计数**泛化消息传递做链路预测，1-hop 解码即子图级聚合；官方仓库 `DeepGraphLearning/NBFNet` |
| *Graph Neural Networks for Link Prediction with Subgraph Sketching (ELPH)* · **ICLR 2023** · [arXiv:2209.15486](https://arxiv.org/abs/2209.15486) | 用**子图素描（sketch）**近似闭合子图特征，把子图开销降到线性；官方仓库 `melifluos/subgraph-sketching` |
| *Bring Your Own View (PS2)* · **WSDM 2023** · [arXiv:2212.12488](https://arxiv.org/abs/2212.12488) | 已入分类表：每条边**个性化选子图**（双层优化） |
| *Revisiting Link Prediction: A Data Perspective* · **ICLR 2024** · [arXiv:2310.00793](https://arxiv.org/abs/2310.00793) | **批判/负面结果**：大量 LP 结果用了有信息泄漏的划分，重评后排名洗牌 → 「先修数据，再谈模型」 |

**脉络小结**：子图范式在链路预测上是**主流工程范式**（包边子图 + GNN）。近三年从「怎么把子图建模到位」（NBFNet 路径计数）走向「怎么把子图开销降下来」（ELPH 素描、PS2 个性化采样），并用**数据视角批判**（Revisiting LP）回踩了整个基准体系。

### 2.5 与 Transformer 的关系（承接 §2.2）

Subgraphormer（ICML 2024）表明：子图 GNN 与 GT 不是竞争，而是**乘积图上的两种注意力/消息模式**。这也解释了为什么 2024 之后少有「纯子图 vs 纯 GT」的口水战。

---

## 3. 验证与复现记录

### 3.1 自跑验证实验：1-WL 天花板 + 子结构越界（本次实测）

**问题**：MPNN（GIN ≤1-WL）到底「差在哪」？子结构信号（GSN 式）与节点标记子图（Subgraph-GNN 式）为何能越界？

**实验**（CPU，秒级；脚本可复现）：
1. 用程序**搜索**出一对 7 节点、4-正则、**1-WL 不可区分但三角形数不同**的图（7 vs 6）。计算上证实：`isomorphic? False`，`1-WL hash equal? True`，度序列同为 `[4,4,4,4,4,4,4]`。
2. 用自包含 GIN 前向（4 层，随机初始化即可，因为结论是结构性的；实现见 `verify_expressivity.py` 内联版，旧仓 `paper/experiments/models.py` 已遗失）：
   - **GIN 对两张图输出完全相同的图嵌入**（`[-3.516, 3.430]` vs `[-3.516, 3.430]`）→ 无论怎么训练都分不开，实证 **MPNN ≤ 1-WL**。
3. 同样的 GIN 骨架，但拼上三角形计数结构特征（`GSNv`，GSN 式）：嵌入**不再相同**（`[14.56, 9.39]` vs `[11.38, 1.02]`）→ 结构计数越过 1-WL。
4. 节点标记（**Subgraph-GNN 机制**，即 SSWL/SUN 理论里的 marked-WL）：标记后两图的可根化 WL 颜色多重集**不同**（G0 只有 1 类着色、G1 有 2 类）→ 节点级子图 GNN 表达力 >1-WL。

**结论（已验证、可复现）**：这一对图把「1-WL 失败实例」具象化；三个机制（结构计数、谱/同态编码、节点标记子图）都能越界，但代价不同——正是综述要写清楚「表达力 ≠ 精度、越界有成本」的实证锚点。脚本在本目录 `verify_expressivity.py`（GIN vs GSNv）与 `verify_marking.py`（节点标记 WL），用 `../paper/.venv/bin/python` 直接跑即可复现。

### 3.2 官方代码核查表（GitHub 逐一确认存在，2026-08-07）

| 论文 | 官方仓库（已核实存在） | 备注 |
|------|------------------------|------|
| GSN | `gbouritsas/graph-substructure-networks` · `gbouritsas/GSN` | arXiv:2006.09252 一致 |
| ESAN | `beabevi/ESAN` | ICLR 2022 Spotlight |
| SUN | `beabevi/SUN` | NeurIPS 2022 |
| SSWL 表达力层级 | `subgraph23/SWL` | ICML 2023 |
| Beyond WL 定量框架 | `subgraph23/homomorphism-expressivity` | ICLR 2024 Oral |
| 乘积图统一框架 | `BarSGuy/Efficient-Subgraph-GNNs` | ICML 2024 |
| Subgraphormer | `BarSGuy/Subgraphormer` | ICML 2024 |
| S2N | `dongkwan-kim/S2N` | ICML 2024 |
| HyMN | `jks17/HyMN` | ICML 2025 |
| NBFNet | `DeepGraphLearning/NBFNet` | NeurIPS 2021 |
| ELPH | `melifluos/subgraph-sketching` | ICLR 2023 |
| Fractal Nodes | `jeongwhanchoi/MPNN-FN` | AAAI 2026 |

> 未核实到官方仓库（本轮只做「存在性」核查，未逐一克隆运行）：WL-at-the-margin、谱不变 GNN、Lanczos、有效电阻过压缩、Revisiting LP、Generalizing WL Kernels、FPS、DFWL、OSAN、PIN/Walking Out。**引用时只宣称「该文有/无我们核实的官方代码」**，不臆断其可用性。

### 3.3 协议合理性要点（横比必读）

沿用本仓 `cin.md` 的三协议分立原则，各近期工作的协议归属如下，**不可混排**：

| 协议族 | 代表数据集 | 近期子图 GNN / GNN 工作 | 注意 |
|--------|------------|--------------------------|------|
| **TUD（GIN 乐观 10-fold max-val）** | MUTAG/PTC/NCI1/… | ESAN、SUN、OSAN、SSWL（部分表）、WL-at-margin | 网格搜超参 + 选 max val，**乐观族**；只可与同族比 |
| **OGB 固定划分** | ogbg-molhiv / molbace / moltox21 | Subgraphormer、HyMN、S2N、NBFNet(LP)、ELPH | scaffold 划分相对干净；按 val 选超参仍在 |
| **Dwivedi 固定划分（LRGB/GTPA）** | ZINC / Peptides-func / Peptides-struct / PATTERN / CLUSTER | Subgraphormer、HyMN、GT 综述 | MAE/Acc，10 次初始化 mean±std |
| **LP 基准** | ogbl-*、FB15k、WN18RR | SEAL、NBFNet、ELPH、PS2 | **注意 Revisiting LP（ICLR 2024）：数据划分有泄漏时，排名会被重洗** |
| **合成表达力** | CSL / EXP / SR25 / 计数任务 | SSWL、FPS、谱不变、WL-margin | 表达力 ≠ 下游精度，单独成表 |

**判断**：近三年子图 GNN 主协议已从「TUD 乐观族」明显转向 **OGB + LRGB 固定划分**，这是协议可信度上升的积极信号；但 **LP 类基准的数据泄漏问题**（Revisiting LP）仍是横比的雷区，综述里必须写。

---

## 4. 与组内主线（Kernel → GNN → GSN → KSVD）的距离

| 近期文献信号 | 与主线的关系 |
|--------------|--------------|
| 同态计数编码（ICLR 2024）、Generalizing WL Kernels to Subgraphs（ICLR 2025） | 与 GSN 的**轨道/同态计数**、Kernel 的**预定义特征**同族：都「预定义 + 显式结构」 |
| 谱不变（ICML 2024） | 全局谱特征，一次性预处理；与 KSVD「数据驱动原子」形成**预定义 vs 学出**的张力，可作对照章 |
| 乘积图统一框架 / Subgraphormer（ICML 2024） | 给「子图之间的关系」一个**形式化**（乘积图边），与组内 patch 间「哈希记录节点对应」的直觉同构——**值得深读借形式化语言** |
| HyMN 中心性采样（ICML 2025） | 子图/结构**采样策略**；与组内 RW 采 patch 的「覆盖-成本」权衡直接对应 |
| 表达力层级 SSWL（ICML 2023） | 回答「子图 GNN 到底多强」的**上界**；组内 KSVD 若以子图 GNN 自居，需对照此层级声明位置 |

**一句话**：近年文献证明「显式结构 + 局部消息」仍是越过 1-WL 的主流且已被公理化；组内 KSVD 的差异化在于**结构原子从数据学出、可重构、可解释**——这正是预定义计数（GSN）与谱编码（Spectral Invariant）给不出的，综述里把它写成主线收口。

---

## 5. 缺口与下一步

1. **图基础模型 / LLM+Graph 方法级条目**：本轮只核实了综述，方法级（GraphGPT、ZeroG、GraphAgent、ReasonGraph 等）未逐篇核实，需要一轮专门核查再补表。
2. **未核实仓库的官方代码**：上表「未核实」项，可克隆后跑协议（尤其 WL-at-margin、Effective Resistance、Subgraphormer 的 LP 版）。
3. **回填 `paper_classify.xlsx`**：已出 **v3**（`paper_classify_v3.xlsx`，v2 为中间版、被 v3 取代）。按「24 年及以前已充分 + 只收顶会」的原则，v3 新增 **13 条**，全部是 2025–2026 顶会收录（从 DBLP 各大会正式收录列表核出）：
   - **GNN/MPNN +11**：Counting Substructures(ICLR'25)、Revisiting RW(RWNN, ICLR'25)、Residual+Norm 防过平滑(ICLR'25)、WLKS(ICLR'25)、Adaptive MP(ICML'25)、Long-range 度量(ICML'25)、IM-MPNN ERF(ICML'25)、MPC 表达力批判(NeurIPS'25)、同配基线 IGNN(NeurIPS'25)、谱不完备(NeurIPS'25)、Fractal Nodes(AAAI'26)。
   - **Subgraph GNN +2**：ISNN(ICML'25)、GPEN(ICML'25)。
   - v2 里 6 条 **arXiv 预印本**（MP-SSM、OpenGLT、Position 文、GFM 综述、Bounded Cycles、ENFA）**已从表内移除**（保留在本 md §1/§2 作脉络参考）。
4. **自跑实验已入仓**：`verify_expressivity.py` / `verify_marking.py` 已在 `docs/literature/deep/` 可复现；如需长期保留可再挪进 `tracks/` 并写结果 registry。
5. **深读候选**（每篇四问，按 3.2 口诀少而深）：`2406.09291`（乘积图形式化，与 patch 关系直觉最贴近）、`2501.03113`（中心性采样，与 RW 采样最贴近）、`2302.07090`（SSWL 层级，定组内位置）。
