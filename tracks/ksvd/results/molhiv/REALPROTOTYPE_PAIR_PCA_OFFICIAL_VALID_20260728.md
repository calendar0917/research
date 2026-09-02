# Real-prototype covariance pair-PCA：official valid 结果与路线判断

日期：2026-07-28

## 1. 这次实际检验的模型

整条冻结链路是：

```text
真实局部结构 patch
→ 在 official train 中选择两组固定原型（farthest / scaffold-facility，各 32 个）
→ 统计原型出现情况
→ 统计距离 1、2 内的紧凑关系特征，形成 compact base
→ 对具体“原型 A 与原型 B 在距离 1/2 共现”的 1984 维向量做无标签 covariance PCA
→ 保留 144 个方向
→ 用一个幅度受限的小修正项补充 compact base
```

所有模型选择和超参数均在 official valid 编码前冻结。official test 未编码、未评估。

## 2. 数据隔离审计

- 拟合数据：exact official train，32,901 个图；
- official valid：4,113 个图，其中正样本 81 个；
- official-train 索引 SHA256：`4e77289653e41be5d2267f54d9f33fda4b70fd622c19b93a282db710a9318b0a`；
- valid 节点已编码；
- official-test 共有 103,927 个节点，latent 非零元素数量为 0；
- official-test evaluations：0。

冻结清单：

```text
results/molhiv/label_free_pair_pca_official_valid_freeze_v1.json
SHA256 fa2dff887b1b2e3bf0110dad5eb79e8a47a7fc47ec7562f21221c6f6210bfde7
```

## 3. Official valid 主结果

### 3.1 分层结果

| 层级 | valid AUC | 相对前一层 |
|---|---:|---:|
| 两组原型出现概率平均 | 0.808167 | — |
| uniform compact，真实距离 1/2 关系 | **0.812457** | **+0.004290** |
| covariance pair-PCA，3 seed 概率平均 | 0.811508 | **-0.000949** |

当前冻结的真实结构模型中，分数最高的是 compact base 的 `0.812457`，而不是 pair-PCA。不过 assignment-shuffled compact 达到 `0.814726`，所以不能把 compact 的增益直接解释为“具体原型对应关系”带来的；它更适合作为当前预测基线。

### 3.2 Pair-PCA 三个训练 seed

| seed | real AUC | 相对 compact base | real - assignment-shuffled |
|---:|---:|---:|---:|
| 0 | 0.811517 | -0.000940 | +0.004428 |
| 1 | 0.811269 | -0.001188 | +0.004100 |
| 2 | 0.811594 | -0.000863 | +0.004127 |
| 概率平均 | **0.811508** | **-0.000949** | **+0.004259** |

三个 seed 的预测相关系数均高于 `0.99999`。因此这里没有“换一个随机种子可能就翻盘”的迹象；优化过程非常稳定，稳定地没有超过 compact base。

### 3.3 冻结门槛

预先冻结的条件：

1. pair-PCA ensemble 相对 compact base 至少 `+0.002`；
2. real 必须超过 assignment-shuffled。

实际结果：

- 条件 1：`-0.000949 < +0.002`，失败；
- 条件 2：`+0.004259 > 0`，通过；
- 总结：**未通过 official-valid gate，停止该 pair-PCA 分支，不运行 official test。**

## 4. 与 full 三折开发结果的对照

| 指标 | full 三折均值 | official valid |
|---|---:|---:|
| occurrence base | 0.778154 | 0.808167 |
| compact real | 0.779491 | 0.812457 |
| compact 相对 occurrence | +0.001338 | +0.004290 |
| compact real - shuffled | +0.000039 | -0.002269 |
| pair-PCA real | 0.782674 | 0.811508 |
| pair-PCA 相对 compact | +0.003183 | -0.000949 |
| pair real - shuffled | +0.002248 | +0.004259 |

full 三折中 pair-PCA 的增益为：

```text
+0.004191 / +0.006706 / -0.001348
```

它从一开始就不是三个 fold 全胜，而是“两个明显为正、一个为负”。official valid 落在负增益一侧，说明该增益没有达到可依赖的跨划分稳定性。

## 5. 如何解释：不是“没有结构信息”，而是“没有形成可靠增量”

### 5.1 可以保留的正结论

真实 pair 特征在三个 seed 上都比 assignment-shuffled 高约 `0.0041–0.0044`。这说明：

> 哪些具体原型在真实分子中相邻或隔一个原子共现，确实包含打乱后会消失的信息。

所以，“具体局部结构组合存在信息”这个较弱结论仍然成立。

### 5.2 不能继续声称的结论

pair-PCA 没有超过 compact base，因此不能声称：

> 当前 covariance PCA + 线性受限修正，能够把这些具体组合信息稳定转化为额外的性质预测能力。

“含有信息”和“在已有强基线上提供增量”是两件不同的事。本次失败发生在第二步。

### 5.3 为什么不像是 rank 或随机训练问题

- 144 个 PCA 方向已经解释 official-train pair 方差的约 `93.4%`；
- 三个训练 seed 的预测几乎完全相同；
- 三个 seed 相对 base 都下降约 `0.001`。

因此，短期继续扫 rank、seed、学习率或残差上限，较可能是在 official valid 上追噪声，而不是解决主要矛盾。

### 5.4 更可能的问题：pair 表示与 compact base 重复

pair 向量的高方差方向，主要描述“训练集中哪些组合经常共同变化”。但这些变化中有相当一部分已经能由以下紧凑量解释：

- 每个原型出现多少；
- 关系矩阵每行的总量；
- 对角线和非对角线总量；
- 距离 1/2 的边对比例；
- 原型相似度加权总量。

因此 PCA 可能优先保留了“变化很大但 compact 已经知道”的方向，而不是“compact 不知道、又和标签有关”的方向。

## 6. 对 GNN/GINE 基础架构的判断

本次结果说明两点：

1. **不使用 GNN 也能得到有竞争力的结构模型。** 仅使用真实原型出现和明确的距离 1/2 关系，official valid 已达到 `0.812457`。
2. **当前失败不能简单归因于没有 GNN。** pair-PCA 的问题更像是新增表示与 base 重复、压缩目标和最终任务不一致。直接换成更深的 GINE，未必会自动解决。

所以更合理的定位是：

- 把 real-prototype + compact distance 模型作为一个独立、可解释的基础模型；
- GINE 可以保留为独立基线或最终的后融合分支；
- 不应默认让所有结构信息先经过 GINE，再期待网络自行分离“出现信息”和“组合信息”。

## 7. 下一条更有针对性的开发路线

当前最值得试的不是继续调 covariance pair-PCA，而是：

```text
具体 pair 向量
− compact 特征能够解释的部分
= pair-specific residual（真正新增的组合细节）
→ 再压缩、再预测
```

直观地说，先从具体组合中扣掉“只看原型总量和总体关系就能猜到的部分”，再学习剩下的细节。这样新分支被迫回答：

> 在 compact base 已知以后，哪些具体 A-B 组合仍然提供额外信息？

建议的开发纪律：

1. 只在 official train 内部三折开发；
2. 先做 8k 快速筛查；
3. 必须同时比较 real 与 assignment-shuffled；
4. 晋级后再做 full 三折；
5. **不再返回 official valid 调参；** 本次 official valid 已经被消费；
6. 在新的内部证据足够强之前，不触碰 official test。

另一个可并行但优先级较低的方向，是在固定宽词汇上做更直接的任务匹配；不过已有 smooth task gate 的优势很小，因此不建议重新做“正负样本各学一套字典”或大规模标签驱动字典搜索。

## 8. 最终结论

当前路线应拆成两部分看：

- **保留为预测基线：** 真实原型词汇 + 距离 1/2 compact 关系。它在 official valid 上相对 occurrence base 提升 `+0.004290`；但 shuffled control 更高，因此暂时不能把这部分增益解释成可靠的“具体结构配对效应”。
- **停止：** covariance pair-PCA 直接叠加到 compact base。它虽然保留了真实组合信息，但 official valid 相对 base 为 `-0.000949`，未通过冻结门槛。

下一步不应继续微调同一 pair-PCA，而应研究“去除 compact 可解释部分后的具体组合残差”。
