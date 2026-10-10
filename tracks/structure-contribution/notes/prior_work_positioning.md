# 先行工作定位 — CSCL vs 已有文献

Track: `structure-contribution` · 建立 2026-10-10
目的：界定研究差异，不追求完整综述。**凡未经本次核验的条目一律标"待核对"，
不得据此声称新颖性。**

## 核验方式说明

2026-10-10 用 Crossref REST API 对论文名做了存在性核验（query.bibliographic）。
能确认"存在 + 发表载体"的记 ✅；只能确认部分信息或完全无法确认的记 ⚠️待核对。
未阅读全文，方法细节描述基于摘要级知识 + 常识推断，**逐条标注可信级别**。

## 逐篇定位表

### 1. MEGAN — Multi-explanation Graph Attention Network ✅

- 出处：A. Mosca et al., *Communications in Computer and Information Science*
  (ESANN 系列), 2023, DOI `10.1007/978-3-031-44067-0_18`。
- 结构单元：原子/边（GNN 注意力通道）。
- 贡献定义：多解释通道的 attention 权重（node/edge 级），多任务各自通道。
- 关系处理：消息传递内部隐式；解释即注意力，非加性恒等分解。
- 内生性：内生（与预测联合训练），但解释是注意力权重而非预测的精确分解。
- 跨分子比较：同一原子类型在不同分子中可比较性弱（每个分子一套权重）。
- 局限（我们押注的差异点）：attention 权重不构成预测的加和恒等式；
  无非重叠结构归属；跨分子类型对齐非设计目标。
- **MEGAN2** ⚠️待核对：本次 Crossref 查询未能确认 "MEGAN2" 独立条目
  （疑为后续期刊扩展版）。细节与差异**未核验，不得引用其内容下结论**。

### 2. MOSE-GNN ⚠️待核对

- 本次 Crossref 查询未找到名为 "MOSE-GNN" 的分子性质预测条目。
  最接近的是 substructure-aware GNN（Chemical Science 2022, DOI
  `10.1039/d2sc02023h`，但那是药物—药物相互作用任务，用 size-adaptive
  子结构）。**该名称对应的论文未确认，不写任何方法细节对比。**
- 待办：拿到原始引用后再补定位。

### 3. MAGE ⚠️待核对

- 本次 Crossref 查询未能在分子性质预测语境确认 "MAGE"。**未确认，不对比。**

### 4. FragNet — fragment-level GNN ✅（细节待核对）

- 出处："FragNet: A Graph Neural Network for Molecular Property Prediction
  with Four Layers of Interpretability"，Research Square preprint
  2024-11-08（DOI `10.21203/rs-5283906/v1`）；另见 JACS 附带 SI DOI
  `10.1021/jacs.5c22620.s001`（正式发表状态待核对）。
- 结构单元（据摘要级知识，待核对）：分子划分成片段（可能 BRICS/算法切分），
  atom-level 与 fragment-level 两级消息传递。
- 贡献定义：片段级可解释读出（四级可解释：原子/键/片段/分子，细节待核对）。
- 关系处理：片段间通过 GNN 消息传递。
- 内生性：内生预测模型。
- 跨分子比较：片段贡献可比（片段词表共享，细节待核对）。
- 局限（我们押注的差异点）：**核心预测机制是消息传递 GNN**——不符合我们的
  约束；其片段划分的原子/键归属与"贡献 = 预测的精确加和分解"关系待核对。
  我们的差异主张：非 GNN 核心 + 每键唯一归属 + 可辨识性约束 + 三层解释
  评价。**在 FragNet 细节未核对前，不声称"首个片段级可解释分子 GNN 替代"。**

### 5. Fragment-Level Shapley / "FragShapley" ✅（名称待核对）

- 出处："Chemically Interpretable Explanations for Molecular Property
  Prediction via Fragment-Level Shapley Values"，ChemRxiv 2026-07-06
  （DOI `10.26434/chemrxiv.15002302/v2`）；JCIM SI DOI
  `10.1021/acs.jcim.6c01425.s001`（发表状态待核对）。
- 结构单元：化学片段（划分方式待核对）。
- 贡献定义：片段级 Shapley 值（博弈论归因，事后计算）。
- 关系处理：Shapley 框架天然考虑子集组合 → 隐含交互，但计算代价高。
- 内生性：**事后**（post-hoc），作用于已训练模型（多为 GNN）。
- 跨分子比较：片段层面可比较。
- 局限（我们押注的差异点）：事后归因计算昂贵、解释不内生、不约束训练；
  Shapley 值对相关特征的解释语义有争议。我们做**内生**分解，训练时即约束
  可辨识性。**差异方向明确，但该方法是否也给出了加和恒等式（Shapley 性质
  上是 efficiency 满足的）→ 需核对后精确表述。**

### 6. 经典 group contribution / group-interaction 方法 ✅（DBGC 待核对）

- 传统 QSPR 方法（Joback–Reid、Abraham、 Benson 基团加和等，教科书级先例）：
  分子性质 = Σ 基团贡献（+ 成对/邻近交互项）。**线性、手工词表、小数据。**
- 贡献定义：回归系数（基团出现次数 × 系数）；交互 = 邻近基团组合系数。
- 关系处理：手工定义的邻近/键连组合。
- 内生性：本身就是加性预测模型（内生），但性能受限于线性 + 手工词表。
- 跨分子比较：天然跨分子（同一基团同一系数）。
- 局限：手工词表不覆盖复杂环系与上下文依赖；线性假设；无共享非线性上下文
  修正。我们的差异主张：确定性化学划分（自动、保环）+ 学习型共享非线性
  修正 δ + 学习型关系项 γ + 现代训练。
- **DBGC** ⚠️待核对：本次 Crossref 查询未确认该缩写对应文献。拿到原始引用
  后补。

## 其他相关但未逐篇展开的领域（常识级，细节待核对）

- molecular fragments/motif 词表 + GNN（MOGONET/MolBert 类各种）——共同点：
  核心是 GNN。
- Subgraph/fragment-level GNN 解释（GNNExplainer、PGM-Explainer、
  SubgraphX 等）——事后或注意力，非加性内生分解。
- GAM / EBM / GA2M 结构化表格模型——加性+成对交互、可辨识性处理成熟，
  但输入是手工描述子而非"结构单元 + 关系"图，分子级性能弱于图模型。
- Neural Additive Models / SIAN 等可加性网络——思路同源，输入单元不同。

## 我们的候选生态位（**未成立，待 v0 证据**）

> 在有竞争力的预测性能下，使跨分子共享的结构贡献与上下文交互
> **更可对齐、更可辨识、更可验证**：
> (a) 非重叠、保环、确定性划分 → 原子/键归属无歧义；
> (b) 与预测内生的加性+成对分解（含学习型类型词表与上下文修正）；
> (c) 可辨识性约束（参数分离 + 训练域中心化）与三层独立评价
>     （计算完整性 / 稳定性 / 合成真值）。

**不作为新颖性声称的内容**（均有先例）：motif-based 预测、跨分子共享 motif、
片段正负贡献、片段连接解释、不用 GNN、加性/二阶交互模型本身。

## 结论

v0 的对照实验若显示 B（完整分解）性能接近 D（不透明对照）且 B>C、B>A，
则我们的差异主张获得首个证据：显式分解不必以性能为代价，且其归因可以通过
完整性/稳定性/合成真值三层检验。反之，生态位不成立或需重新定位。
