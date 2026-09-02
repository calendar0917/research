# WL-KSVD：WL 子树核 + KSVD 字典学习图嵌入（压缩精读）

> 概念：[concepts.md](concepts.md)

| 项 | 内容 |
|----|------|
| 标题 | Dictionary Learning on Graph Data with Weisfeiler-Lehman Sub-Tree Kernel and KSVD |
| 出处 | ICASSP 2023 · Liyanage, Pearsall, Izurieta, Whitaker |
| 链接 | https://doi.org/10.1109/ICASSP49357.2023.10094980 |
| 访问 | 闭源（IEEE paywall），无可公开获取的预印本 |

## 一句话

用 **WL 子树核** 从图中提取固定维度的特征向量 → 在这些特征向量上跑 **KSVD 字典学习** → 稀疏编码系数作为图嵌入 → 下游分类。

## 四问（摘要）

1. **旧缺陷**：Graph2Vec 等图嵌入方法使用 Doc2Vec 范式训练，WS 子树核特征本身没有经过字典学习的压缩/去噪。
2. **关键一步**：将 Graph2Vec 的 Doc2Vec 训练部分替换为无监督 KSVD 字典学习。WL 子树核负责提取局部子图结构特征，KSVD 负责将这些特征压缩为稀疏编码。
3. **目的**：主战场是 **B 图分类**。与 Graph2Vec、node2vec 等图嵌入方法对比分类效果。
4. **主线**：与我们的 KSVD track 共享"图结构 → 字典学习"这一核心思想，但关键差异在于：
   - 他们用 WL 核做特征提取（固定过程），我们直接从原始邻接/节点特征出发
   - 他们做的是图级嵌入（每图一个编码向量），我们做了图级和节点级（每个节点一个编码）
   - 他们的字典学习是后处理步骤，我们的字典学习是核心管道

## 数据流

```
图 G
  → WL 子树核迭代（聚合邻居标签 → hash → 直方图）
  → 固定维度特征向量 y_G（对每个图）
  → 堆所有训练图的 y_G 为 Y = [y_1 | … | y_N]
  → KSVD：稀疏编码 X 与字典更新 D，使 Y ≈ DX
  → 稀疏编码 x_G 作为图嵌入
  → 下游分类器（MLP/SVM）
```

## 方法细节

1. **WL 子树核特征提取**：对每个图运行 1-WL 颜色迭代 h 轮，每轮对颜色标签做 hash 得到直方图，拼接所有轮的直方图作为图的特征向量。这是 Graph2Vec 中使用的同一套特征。
2. **KSVD 字典学习**：与标准 KSVD 一致——交替 OMP 稀疏编码 + 逐原子 SVD 更新。
3. **分类**：将稀疏编码作为特征输入监督学习器。

## 已知结果

**注意：论文全文不可获取，以下仅来自摘要。** 摘要声称：
- 分类结果与现有图嵌入方法（Graph2Vec 等）**持平（on-par）**
- 具体数据集和数字不可得
- 论文被引次数：1（Semantic Scholar），说明后续关注度较低

## 与 FDDL（ICCV 2011）的区别

用户提到的 "Fisher Discrimination Dictionary Learning for Sparse Representation"（Yang et al., ICCV 2011）是**另一篇独立的论文**，不是 WL-KSVD。FDDL 的核心贡献是：

- 在字典学习中加入 Fisher 判别准则：类内散度小、类间散度大
- 结构化字典：D = [D_1, D_2, ..., D_c]，每个子字典对应一个类别
- 分类使用重建误差 + 稀疏编码系数双重判别信息
- 数据集：Extended Yale B（~97%）、AR（~95%）、Multi-PIE（人脸识别）、Caltech 101/256（物体分类）、USPS（数字识别）
- **FDDL 是图像分类方法，与图结构无关**

## 边界

- WL-KSVD 的 WL 特征提取是固定的，不像我们的方法可以端到端学习
- 论文结果只声称"持平"，没有显著超越现有方法
- 关注度低（仅 1 次引用），在学术界影响力有限
- 全文不可获取，具体实验数字无法确认
- 该论文未解决 patch 采样、跨图字典对齐、节点级编码等我们面临的核心问题

## 对 KSVD track 的启示

1. WL-KSVD 证明了"WL 特征 + KSVD 字典"这条路在技术上可行，但效果并不突出（仅持平 Graph2Vec）
2. 他们将 KSVD 定位为一种"特征压缩/去噪"工具，而非"结构发现"工具——这与我们的 track 核心假设（KSVD 能从数据学到新的结构原子）有本质区别
3. 他们的 WL 特征提取与我们的 permutation-invariant patch 特征（`wl_patch_features`）在思路上接近，但我们的 pipeline 更灵活（可替换采样策略、可做节点级编码）
4. 该论文的"on-par"结果暗示：仅靠 KSVD 压缩 WL 特征，不足以产生超越现有图嵌入方法的增益——这与我们在 MolHIV 上的发现（KSVD 字典不及真实 patch 原型）方向一致