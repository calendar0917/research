# Beam8 如何转化为分类收益：机制分析与下一步

> 日期：2026-08-14  
> 数据：TU Mutagenicity、ENZYMES  
> 状态：完成 Mutagenicity 的 orbit-safe node-incidence、frozen residual、BAG specificity
> controls，以及 ENZYMES 完整属性、多模态基线、full-feature-canonical 与 canonical-slot
> Beam8 未见验证，并进一步完成 BZR/COX2/DHFR 的 patch-graph 预筛与 BZR 未见条件检验。
> Mutagenicity 的 BAG 收益不能归因于 Beam8；ENZYMES 的 patch mean 与 slot-aligned 表示均未
> 胜 matched controls；BZR 虽出现稳定 TRUE>BAG/SHUFFLED 的开发信号，但未在 GLOBAL_STATS
> 条件后确认正确 binding 或净增量。因此当前不授权扩大 Beam8 分类接口或训练 patch-GNN。

## 1. 原来的分类接法为什么弱

此前 BAG/compact/late fusion 都先把 patch token 池化成一个图向量，再与属性分支融合。
这一步丢失了三个分类可能需要的对应关系：

1. 哪个原子属于哪个结构 patch；
2. 同一个原子同时处在哪些连续 patches 中；
3. 哪一种 patch 内容与哪一种相邻 patch 内容共同出现。

因此 TRUE−SHUFFLED 可以检测到关系存在，却难以超过属性/BAG 基线。

## 2. 新入口：Beam8 patch → node incidence → GINE

将 attributed Beam8 patch token 通过 node–patch incidence 回写节点，并在全图 attributed
automorphism orbit 内平均，以保证 node relabel equivariance。原始 node/edge attributes 仍由
GINE 主干读取，结构通道通过每层 zero-init FiLM 注入。

无标签 feasibility：

| metric | value |
|---|---:|
| mean node coverage | 73.76% |
| multi-patch node fraction | 25.76% |
| chain-active node fraction | 56.16% |
| same-atom pair disambiguation | 80.34% |
| orbit-safe node equivariance | 100% |
| graph readout invariance | 100% |

这说明 Beam8 incidence 既有非平凡局部信息，也可以合法接入节点级 GNN。

## 3. 分类实验告诉了我们什么

### 3.1 冻结 Stage A

| variant | balanced accuracy |
|---|---:|
| GINE_ONLY | 72.68% |
| RAW FULL TRUE | 69.69% |
| INIT FULL TRUE | 73.54% |
| FINAL FULL TRUE | 72.40% |
| FINAL FULL SHUFFLED | 69.63% |
| FINAL BAG BROADCAST | 73.02% |
| FINAL NO RELATION | 71.22% |

关键配对：

- FINAL TRUE−SHUFFLED：`+2.76pt`，3/3；
- FINAL−RAW：`+2.70pt`，3/3；
- FINAL−INIT：`-1.14pt`，0/3；
- FINAL−GINE：`-0.28pt`；
- FINAL TRUE−BAG：`-0.62pt`。

因此正确 node binding 与 sparse compression 都是真信号，但普通 reconstruction KSVD updates
把 INIT vocabulary 中更有任务价值的方向削弱了。

### 3.2 INIT follow-up（结果可见后的诊断）

| variant | balanced accuracy |
|---|---:|
| INIT FULL TRUE | 73.54% |
| INIT FULL SHUFFLED | 74.45% |
| INIT BAG BROADCAST | 70.97% |
| INIT NO RELATION | 74.90% |

配对：

- INIT TRUE−GINE：`+0.86pt`，2/3；
- INIT TRUE−BAG：`+2.57pt`，3/3；
- INIT TRUE−SHUFFLED：`-0.91pt`，2/3；
- INIT TRUE−NO_RELATION：`-1.36pt`，1/3。

localized incidence 明显优于图级广播，但当前 slot/position/relation-degree metadata 不但没有
稳定帮助，反而压低 INIT。SHUFFLED 的均值更高也说明目前不能把 INIT 的增量归因于正确绑定；
仍需更有语义的 relation operator 和多 model seeds。

## 4. 真正值得继续的路线

### Stage B1：冻结 INIT，学习 content-aware patch relation

当前 relation channel 只告诉模型“有几个邻居、重叠多少”，没有告诉它“哪个结构连接哪个
结构”。对每个 relation type `r` 和 patch code `z_p`，构造：

`m_p^r = normalize(R^r) Z`

并使用低容量交互：

`[z_p, m_p^r, z_p ⊙ m_p^r, |z_p - m_p^r|]`

再通过 orbit-safe node incidence 回写 GINE。第一轮只用 INIT，不允许 KSVD updates，也不加
attention。必须比较：

- INIT LOCAL CONTENT；
- INIT RELATION TRUE；
- INIT RELATION SHUFFLED；
- INIT BAG；
- GINE_ONLY。

这一步直接测试“哪一种 patch 接在哪一种 patch 后面”是否能提供分类增量。

### Stage B2：让分类监督选择 atoms，而不是立即移动 dictionary

若 B1 通过，先冻结 `D_INIT`，只学习 24 个 atom 的对角 gate 或低秩投影，并与 GINE 联合
训练。这样分类标签可以选择有用 atoms，同时不会让 reconstruction objective 把 vocabulary
推向高频但无判别力的方向。

需要对照：fixed INIT、learned gate、label-shuffled gate、TRUE/SHUFFLED incidence。

### Stage B3：graph-supervised discriminative KSVD

只有冻结词汇的监督 gate 有效后，才把 dictionary 本身纳入任务目标。不能简单把 graph label
复制给每个 patch；这会产生严重 patch pseudo-label noise。应使用 graph-level multiple-instance
目标：

`min ||Y-DX||² + λ CE(y_g, classifier(IncidencePool(X_g))) + μ||X||₁`

也就是 reconstruction 与 graph classification 联合优化，分类损失经过 node incidence/graph
readout 回传到 codes/dictionary。对应 LC-KSVD/D-KSVD 的思想，但监督发生在 graph level。

### Stage B4：最后才考虑 cross-attention

只有 content-aware TRUE 稳定超过 SHUFFLED/BAG/GINE 后，cross-attention 才有明确对象：

- query：GINE node states；
- key/value：覆盖该节点的 Beam8 patch/neighbor-patch tokens；
- mask：真实 node–patch incidence 与 patch relation。

否则 attention 只是在放大尚未建立的信号。

## 5. Stage B1 与 model-seed 验证结果

### 5.1 Content-aware relation

按上面的 Stage B1 设计，固定 INIT vocabulary，使用 CHAIN/NONCHAIN_OVERLAP 邻居 code 的
message、乘积和差异：

| variant | balanced accuracy |
|---|---:|
| GINE_ONLY | 72.68% |
| INIT LOCAL CONTENT | 74.90% |
| INIT RELATION TRUE | 66.49% |
| INIT RELATION SHUFFLED | 69.52% |

- RELATION TRUE−LOCAL：`-8.42pt`，0/3；
- RELATION TRUE−SHUFFLED：`-3.03pt`，0/3；
- RELATION TRUE−GINE：`-6.19pt`，0/3。

这说明直接把高维邻居 message/interaction 经 FiLM 注入会形成强干扰；“内容感知”概念本身
不能自动保证分类收益。这条具体 patch-message 实现停止，不扩展 model seeds。

### 5.2 Localized INIT multi-model-seed

由于 seed0 的 LOCAL−GINE 为 `+2.23pt`，固定 split seed 0，扩展 model seeds 1/2：

| model seed | LOCAL−GINE |
|---:|---:|
| 0 | +2.23pt |
| 1 | -4.86pt |
| 2 | -4.36pt |

9 个 fold×model-seed 单元汇总：

- GINE：`73.91%`；
- localized INIT：`71.58%`；
- LOCAL−GINE：`-2.33pt`，W/T/L `2/0/7`。

因此 seed0 的增益是 neural optimization/checkpoint 偶然性，不能当作稳定分类收益，也不进入
split seeds 1/2 或监督 atom gate。

## 6. Frozen residual 设计（已完成）

当前 joint training 的问题是：即使 FiLM 从零初始化、初始函数等于 GINE，训练过程中结构支路
仍会改变 backbone optimization trajectory。不同 model seeds 下，GINE 本身和融合模型都高度
波动，弱互补信号被优化扰动淹没。

因此最后一个分类机制检查改成 **frozen-backbone residual calibration**：

1. 先按 strict protocol 训练 GINE_ONLY；
2. 冻结全部 GINE 参数与 node states；
3. 只训练一个很小的 Beam8 residual head，输入 orbit-safe localized INIT incidence；
4. residual 以零初始化加到冻结 GINE graph logits；
5. 对照 TRUE、orbit-safe SHUFFLED、BAG 和零 residual；
6. 使用 inner-OOF 或严格 inner-validation 决定 residual strength，outer-test 只评估一次。

该设计不会让 Beam8 改变 GINE 的特征学习过程，因此能更干净地回答：

> 在一个已经训练好的属性/边结构分类器条件下，Beam8 是否还包含可校准的独立残差信息？

具体的 multi-model-seed 与 multi-split 判定见第 8 节。

## 7. Frozen residual 前的阶段结论

Beam8 要产生分类收益，关键不是“把图级 KSVD 向量接到更大的分类器”，而是：

> patch→node incidence 是比图级 pooling 更正确的接入界面，但现有 INIT/FINAL token 在
> joint GINE training 中没有形成跨 model-seed 的稳定增量；普通 KSVD updates 与直接
> content-message 都会恶化结果。

在进入 frozen residual 前，已经证明 localized incidence、正确 FINAL binding 和 sparse
compression 包含机制信号；同时也已证明这些信号在 joint training 中不能稳定超过 GINE。
因此第 8 节只执行冻结 GINE 的 residual calibration，不恢复普通 KSVD updates、patch relation
message 或扩大融合模型。

## 8. Frozen residual：分类增量成立，但 binding gate 未成立

按 strict two-stage protocol 冻结 GINE 参数与 BatchNorm，只训练 rank-16、zero-init 的
conditional residual：

`u_v = tanh(W_h h_v) ⊙ tanh(W_s s_v)`

固定 split seed 0 时，三个 model seeds、九个 fold×model-seed 单元全部通过：

- TRUE−GINE：`+3.44pt`，W/T/L `9/0/0`；
- TRUE−SHUFFLED：`+1.10pt`，`8/0/1`；
- TRUE−BAG：`+1.69pt`，`8/0/1`。

随后按冻结的 multi-split protocol，固定 model seed 0，扩展 split seeds 1/2。九个
split-seed×outer-fold 单元汇总为：

| variant | balanced accuracy |
|---|---:|
| GINE_FROZEN | 73.32% |
| TRUE_RESIDUAL | 75.24% |
| SHUFFLED_RESIDUAL | 74.95% |
| BAG_RESIDUAL | 74.04% |

配对归因：

| comparison | mean | W/T/L | split0/1/2 mean |
|---|---:|---:|---:|
| TRUE−GINE | +1.92pt | 8/0/1 | +2.96 / +1.13 / +1.67pt |
| TRUE−SHUFFLED | +0.29pt | 6/0/3 | +0.83 / +0.46 / −0.42pt |
| TRUE−BAG | +1.20pt | 9/0/0 | +1.18 / +1.53 / +0.89pt |

因此预注册 gate 的 increment、localization、split-majority 和 worst-split increment 均通过，
只有 binding 失败：总体 `+0.29pt < +0.5pt`，且 split seed 2 的 mean 为负。

这将结论修正为：

1. **Beam8 可以转化成跨 split 的净分类收益。** 冻结主干后，TRUE residual 对 GINE 的增量
   在三个 split mean 中全部为正，且 8/9 folds 为正；
2. **收益需要 graph-specific、node-heterogeneous 的 Beam8 field。** TRUE 对 BAG 在 9/9
   folds 为正，说明简单图级均值广播不足；
3. **尚不能把收益归因于精确 patch→atom binding。** SHUFFLED 保留同图 row multiset 和
   node heterogeneity 后几乎追平 TRUE，split 2 甚至更优；
4. 当前可支持的机制是“Beam8 graph content + heterogeneous conditional calibration”，不是
   “正确 Beam8 patch 必须落在对应原子上”。

因此不应因为净分类增量为正就直接进入 cross-attention、普通 KSVD updates 或监督 dictionary。
若继续，最小且可归因的下一步应是对 SHUFFLED control 做多次置换重复，判断 binding 失败究竟
是单次置换方差，还是精确位置确实不重要；只有 TRUE 稳定超过 shuffle distribution 后，才有
依据学习 atom gate 或更强的局部交互。

## 9. Repeated-SHUFFLED：正确 binding 是小而稳定的附加效应

为判断第 8 节的 binding 失败是否来自单次置换方差，冻结原有 TRUE scores、base epochs、
model seed、dictionary recipe 与 residual capacity，在 3 split seeds × 3 folds 上各运行 8 个
独立 SHUFFLED realizations，共 72 个 TRUE−SHUFFLED comparisons。repeat 0 必须逐 fold 精确
复现原结果；所有 parity checks 均通过。

汇总结果：

| metric | value |
|---|---:|
| TRUE−SHUFFLED mean | +0.61pt |
| W/T/L | 51/0/21 |
| positive fold means | 7/9 |
| split0 mean | +0.55pt |
| split1 mean | +0.87pt |
| split2 mean | +0.40pt |

九个 fold-level means 为：

`+0.15, +0.49, +1.02, −0.49, +0.88, +2.21, +0.76, −0.28, +0.71pt`

因此 repeated-shuffle diagnostic gate 全部通过。原 single-shuffle 的总体 `+0.29pt` 以及
split 2 的 `−0.42pt` 确实受单次置换抽样影响；在错误绑定分布上取平均后，TRUE 在三个 splits
中均为正。

机制结论进一步修正为：

1. Beam8 frozen residual 的最大来源仍是 graph-specific content 与 node heterogeneity：
   TRUE−GINE 为 `+1.92pt`，TRUE−BAG 为 `+1.20pt`；
2. 精确 patch→atom binding 在此基础上再提供约 `+0.61pt` 的小效应；
3. binding 不是每个 fold 都成立，2/9 fold means 为负，单个 shuffle realization 的方差也很大；
4. 因而当前可以说“正确 binding 获得 shuffle-distribution 支持”，但不能把全部分类增量都
   归因于 binding，也不能把该 post-hoc audit 当成独立 confirmatory evidence。

如果继续进入监督机制，优先级应是低容量、冻结 dictionary 的 atom/channel gate，并继续保留
repeated-SHUFFLED 与 BAG 对照；不应回到普通 KSVD updates、content relation message 或直接
上 cross-attention。更严格的做法是先在未见的 split seeds 上冻结同一 repeated-shuffle gate
做确认，再决定是否学习 gate。

## 10. 未见 split 3/4：classification 弱保留，localization 与 binding 未确认

在第 9 节结果可见后，先冻结未见 split seeds 3/4 的 confirmatory protocol，再运行完整
frozen residual 与每 fold 8 次 SHUFFLED，共 6 outer folds、48 个 binding comparisons。
所有 GINE 与 repeat0 parity errors 均为 0。

六个未见 split×fold 单元：

| variant | balanced accuracy |
|---|---:|
| GINE_FROZEN | 73.68% |
| TRUE_RESIDUAL | 74.66% |
| SHUFFLED_RESIDUAL | 75.14% |
| BAG_RESIDUAL | 75.56% |

配对结果：

- TRUE−GINE：`+0.98pt`，W/T/L `5/0/1`，split 3/4 为 `+1.02/+0.93pt`；
- TRUE−BAG：`−0.91pt`，`0/0/6`，split 3/4 为 `−1.29/−0.52pt`；
- repeated TRUE−SHUFFLED：`−0.47pt`，`18/0/30`，split 3/4 为
  `−1.29/+0.35pt`；
- 6 个 repeated-shuffle fold means 中仅 split 4 的 3 folds 为正，split 3 的 3 folds 全负。

因此 confirmatory gate 失败：classification 比 `+1pt` 门槛少 `0.02pt`，更关键的是
localization 在 6/6 folds 反向，binding 的 mean、win rate、fold majority、split majority 与
worst-split checks 全失败。第 9 节的 binding 小效应不能外推到未见 splits，不授权进入
atom/channel gate。

将 model seed 0 的 split seeds 0–4 共 15 folds 合并，只作为机制综合：

| comparison | mean | W/T/L | split-level pattern |
|---|---:|---:|---|
| TRUE−GINE | +1.54pt | 13/0/2 | 5/5 split means 正 |
| BAG−GINE | +1.18pt | 13/0/2 | 4/5 正，split 1 为负 |
| SHUFFLED−GINE | +1.56pt | 13/0/2 | 5/5 正 |
| TRUE−BAG | +0.36pt | 9/0/6 | split 0–2 正、split 3–4 负 |

这说明可泛化部分更接近：

> graph-specific Beam8 content 对 frozen GINE states/logits 的低容量条件校准。

它不要求正确 localized binding；BAG 与 SHUFFLED 也能获得相近甚至更高收益。TRUE incidence
仍可作为一种实现，但不再具有机制优先级。

因此当前路线边界为：

1. 停止 atom/channel gate、graph-supervised dictionary、cross-attention 与普通 KSVD updates；
2. 不再声称正确 patch→atom binding 是稳定分类收益来源；
3. 若仍需继续 Beam8 分类路线，只探索更简单的 **graph-conditioned frozen residual**：直接以
   Beam8 graph summary/BAG 条件化冻结 GINE，并做跨 model-seed × split-seed 验证；
4. 这是一条从局部融合退回到可归因简化模型的路线，不再扩大 localized architecture。

## 11. Graph-conditioned BAG：完整 5×3 双轴矩阵通过

Localized 路线停止后，只保留 graph-specific Beam8 BAG summary 与 frozen GINE conditional
residual。先在最有压力的 split seeds 3/4 × model seeds 0/1/2 上确认，再冻结完整矩阵 gate，
补齐 split seeds 0–4 × model seeds 0–2，共 15 cells、45 outer-fold units。

最终结果：

| metric | value |
|---|---:|
| GINE | 73.58% |
| BAG residual | 75.10% |
| BAG−GINE | +1.52pt |
| W/T/L | 36/0/9 |

五个 split means 全部为正：

`+1.75, +1.42, +1.04, +2.59, +0.78pt`

三个 model-seed means 全部为正：

`+1.18, +0.70, +2.66pt`

15 个 split×model cell means 中 12 个为正；完整矩阵的 mean、fold wins、split majority、
model majority、worst split、worst model 与 cell majority gates 全部通过。因此：

`FROZEN_BAG_CONFIRMED_AS_STABLE_BEAM8_CLASSIFICATION_INTERFACE`

但 post-hoc 稳定性诊断给出重要边界：

- `corr(GINE, BAG−GINE) = −0.80`；
- GINE 标准差从 `4.57pt` 降到 BAG 的 `2.83pt`；
- 最弱 5 个 GINE units 平均从 63.87% 提升到 70.84%，即 `+6.97pt`；
- 对 GINE≥74% 的 24 个 units，BAG 平均增量为 `−0.03pt`，W/L `16/8`。

因此最终机制不是“Beam8 普遍抬高一个已经很强的 GINE”，而是：

> Beam8 graph summary 为 frozen GINE 提供低容量、graph-conditioned calibration，主要修复
> 弱 checkpoint 并压缩跨 split/model seed 方差。

这是 specificity controls 运行前唯一通过完整双轴验证的经验接口；它证明“某种图级条件器”
可以修复弱 checkpoint，但当时尚未证明条件器必须来自 Beam8。第 12--13 节随后完成这一归因
检查，并取代“稳定 Beam8 分类接口”的方法性表述。

## 12. Beam8-specific matched controls：BAG 收益由普通统计更强地复现

按结果不可见前冻结的 specificity protocol，在 split seeds 3/4 × model seeds 0/1/2 × 3
folds 的 18 个压力单元上，保持同一 GINE、base epoch、rank-16 residual、输入 padding 与
checkpoint 流程，只替换条件输入：

| variant | balanced accuracy |
|---|---:|
| GINE_FROZEN | 73.31% |
| BEAM8_FULL | 75.00% |
| BEAM8_CODE_ONLY | 74.47% |
| BEAM8_HIST_ONLY | 75.33% |
| BEAM8_RANDOM_DICTIONARY | 74.93% |
| GLOBAL_STATS | 78.67% |

关键归因：

- FULL−GINE：`+1.68pt`，15/18；经验校准再次出现；
- FULL−GLOBAL_STATS：`−3.67pt`，0/18，两个 split 与三个 model means 全负；
- FULL−HIST_ONLY：`−0.33pt`，5/18；
- FULL−RANDOM_DICTIONARY：`+0.07pt`，9/18。

所有 GINE/FULL parity 精确通过。因此不能用容量、checkpoint 或实现差异解释失败。导师在
`luyin12` 中提出的风险——字典取平均后“还不如直接统计这个图”——在当前 Mutagenicity
BAG 接口上被直接验证：真实 dictionary、随机 dictionary 和 patch histogram 基本不可区分，
而无需 Beam8 cover 的基础全图统计显著更强。

判定为：

`BAG_GAIN_NOT_ESTABLISHED_AS_BEAM8_SPECIFIC`

## 13. 在 GLOBAL_STATS 条件后的增量：Beam8 与统计信息基本冗余

单独模型的强弱仍不能排除 Beam8 含有较弱但独立的信息，因此进一步冻结顺序条件检验：先训练
并冻结 `GINE + GLOBAL_STATS`，再训练第二个同容量 residual 读取 Beam8 FULL 或消融输入。

| variant | balanced accuracy |
|---|---:|
| GLOBAL_STATS_STAGE1 | 78.67% |
| GLOBAL_PLUS_BEAM8_FULL | 78.59% |
| GLOBAL_PLUS_BEAM8_CODE_ONLY | 78.58% |
| GLOBAL_PLUS_BEAM8_HIST_ONLY | 78.57% |
| GLOBAL_PLUS_BEAM8_RANDOM_DICTIONARY | 78.60% |

配对结果：

- FULL−GLOBAL：`−0.08pt`，W/T/L `9/1/8`；split3/4 为 `−0.25/+0.09pt`；
- FULL−CODE_ONLY：`+0.01pt`，9/9；
- FULL−HIST_ONLY：`+0.02pt`，6/12；
- FULL−RANDOM_DICTIONARY：`−0.02pt`，9/9。

所有冻结 gate 除 parity 外均失败。部分单元确有约 0.1--0.4pt 的小扰动，但没有跨 split/model
的稳定增量，也不要求真实 dictionary。这将结论从“GLOBAL_STATS 比 Beam8 更强”推进为：

> 在当前 graph-level BAG 接口下，Beam8 可转化的分类信息已被基础全图统计基本吸收；没有证据
> 表明它还包含可由同容量 residual 读取的独立 dictionary/cover 信息。

判定为：

`BEAM8_INCREMENT_BEYOND_GLOBAL_STATS_NOT_ESTABLISHED`

## 14. GLOBAL_STATS 来源：主要是宏观拓扑，而非隐藏的 Beam8 模式

最后将统计条件器拆成节点属性 mean/max/sum、全部属性统计和纯拓扑统计。18 units 汇总：

| variant | balanced accuracy | delta vs GINE |
|---|---:|---:|
| ATTR_MEAN | 76.29% | +2.98pt |
| ATTR_MAX | 75.25% | +1.94pt |
| ATTR_SUM | 75.92% | +2.60pt |
| ATTR_ALL | 76.07% | +2.76pt |
| STRUCT_ONLY | 77.92% | +4.61pt |
| GLOBAL_FULL | 78.67% | +5.35pt |

纯拓扑统计包含图大小、边数、density、degree moments/quantiles、连通分量、三角形、
transitivity 与 cycle rank。它单独解释了完整统计相对 GINE 增益的约 86%，且在若干最弱
checkpoint 上超过 GLOBAL_FULL。属性统计再提供组合补充，但任一属性 component 都明显弱于
完整统计。

这也解释了 BAG 双轴矩阵中 `corr(GINE, BAG−GINE)=−0.80` 的现象：弱 GINE checkpoint 没有
稳定读取本来就显式可算的宏观拓扑，任何相关的图级条件器都可能表现为强校准；Beam8 BAG
只是其中一个较弱代理。

## 15. 与导师方向的最终对应及下一步边界

当前工作与 `luyin12--14` 的对应已经比较清楚：

1. **连续、重叠、有相对关系的 patch substrate 已实现。** Beam8 比 radius-2 更贴近
   `luyin12--13` 对连续 patch 的要求，且 NCI1 上 TRUE−SHUFFLED 曾证明 relation binding
   可检测；因此 Beam8 作为 sampling/compression substrate 仍有用。
2. **结构与节点特征融合已按 `luyin14` 实际尝试。** attributed token、GINE、node incidence、
   graph-conditioned residual 都属于结构/属性融合，而不是只做纯结构分类。
3. **导师关于平均读出退化成统计量的警告得到确认。** BAG 的经验收益稳定，但 FULL 不胜
   histogram/random dictionary，更在 18/18 units 输给 GLOBAL_STATS；加入统计后也无条件增量。
4. **因此当前结果符合导师的研究问题，却不支持把现有实现包装成正方法结果。** 继续上
   cross-attention、atom gate 或监督 dictionary 会绕过已经失败的低容量归因 gate。

后续只保留两条真实数据路线：

- Mutagenicity 与 NCI1 上停止扩大 Beam8 分类模型；Beam8 只报告 coverage、continuity、
  relation detectability、semantic recall、compression 与运行成本；
- 若继续真实 TUData 分类，先对新的 attributed 小数据集冻结筛选：`GNN`、`GLOBAL_STATS`、
  `GNN+GLOBAL_STATS`。只有当统计不形成强 shortcut，且低容量 Beam8 TRUE 能稳定超过
  BAG/SHUFFLED/HIST/RANDOM 后，才进入多模态融合确认。第 16 节的实际筛选随后将 ENZYMES
  提升为优先对象，并将 PROTEINS 降为低优先级。

这不是放弃 Beam8，而是把它从“已经证明能提分类”的结论退回到“已证明几何合法、但分类独立
信息尚未建立”的正确位置。下一轮数据集选择也不应只看规模小，而应先排除宏观统计主导标签的
任务，否则会再次复现当前 Mutagenicity 的结局。

## 16. TU ENZYMES 完整属性预筛：数据更有区分度，但尚未授权 Beam8

为回答失败是否主要来自 Mutagenicity/NCI1 的 benchmark shortcut，进一步检查本地真实 TU
数据。首先发现一个此前容易忽略的加载边界：PyG `TUDataset` 默认 `use_node_attr=False`，原
loader 在 PROTEINS/ENZYMES 上只读取离散 node labels，没有读取原始连续 node attributes。
现已增加显式 `use_node_attr` 入口，保持旧实验默认不变；ENZYMES 的合法完整输入为：

- 18 维连续 node attributes；
- 3 维离散 node-label one-hot；
- 合计 21 维。

当时的初始设计让 canonicalization 只使用3维离散 labels；连续属性不能通过 `argmax` 当作
节点颜色，只进入 GIN 和 patch content。后续第17节的不变性审计证明，这一分离不足以保证
attributed patch content 的 relabel invariance，最终已改为完整 feature-row canonical colors；
连续属性仍不通过 `argmax` 离散化，dictionary structural vector 也仍只编码离散 labels。

### 16.1 预注册 27-unit screen

在 split seeds 0/1/2 × model seeds 0/1/2 × 3 folds 上，采用 inner-validation 选 epoch：

| variant | balanced accuracy |
|---|---:|
| GLOBAL_STATS_LINEAR | 48.29% |
| GIN_LABEL_ONLY | 23.86% |
| GIN_FULL_ATTRIBUTES | 49.85% |
| GIN_FULL_PLUS_GLOBAL | 55.97% |

配对结果：

- FULL−GLOBAL：`+1.56pt`，16/27；split means `+4.72/+0.81/−0.85pt`；
- FULL−LABEL_ONLY：`+25.99pt`，27/27；
- FULL+GLOBAL−GLOBAL：`+7.68pt`，25/27；三个 split 与三个 model means 全正；
- FULL+GLOBAL−FULL：`+6.12pt`，26/1/0。

因此 ENZYMES 与 Mutagenicity 明显不同：连续属性极其重要，而且 GIN 与宏观统计具有强互补。
但原先冻结的晋级 gate 要求 `GIN_FULL−GLOBAL >=3pt` 且所有 split means 为正；该条件失败，
判定必须保持：

`ENZYMES_DO_NOT_ADVANCE_TO_BEAM8`

失败的原因不是任务被统计完全吃掉，而是 full-attribute GIN 单独的跨 split/model 优化方差仍
较大，不能在看到组合结果后事后降低门槛。

### 16.2 容量匹配 follow-up：连续属性的图内组织是真实独立信号

为判断 FULL+GLOBAL 的增益是否只是统计 residual 对任意弱 GIN 的通用校准，冻结结果可见后的
matched follow-up：给 label-only GIN 完全相同的 GLOBAL_STATS residual，再与 full-attribute
分支比较。

| variant | balanced accuracy |
|---|---:|
| GLOBAL_STATS_LINEAR | 48.29% |
| GIN_LABEL_ONLY_PLUS_GLOBAL | 45.07% |
| GIN_FULL_PLUS_GLOBAL | 55.97% |

归因结果：

- FULL+GLOBAL−LABEL+GLOBAL：`+10.90pt`，26/27；
- split means：`+12.70/+7.30/+12.70pt`；
- model means：`+10.76/+9.77/+12.17pt`；
- label-only+global 反而比 global linear 低 `3.22pt`。

全部 gate 与 parity 通过，判定：

`CONTINUOUS_ATTRIBUTE_ORGANIZATION_ADDS_BEYOND_GLOBAL_STATS`

这说明 ENZYMES 的正信号确实来自连续节点属性在图结构中的组织，而不是给任何模型加统计量都
会提高。它直接符合 `luyin14` 所说的“结构与节点特征融合”问题，也证明 TUData 本身不是统一
的失败原因。

但证据边界必须保持：

1. 已证明的是 full-attribute GIN 与 global statistics 的互补，不是 Beam8 的贡献；
2. 原预筛 Beam8 gate 失败，不能在 split0--2 上继续结果驱动地添加 Beam8；
3. 若继续，应先为 `GIN_FULL_PLUS_GLOBAL` 在未见 split seeds 冻结确认协议；只有该多模态基线
   跨未见 splits 成立后，才能在新的未见轴上测试 Beam8 是否提供 TRUE>BAG/SHUFFLED/HIST/
   RANDOM 的条件增量；
4. PROTEINS 的廉价筛选中完整属性+结构统计已约 74.3%，仍更像 shortcut-dominated 数据，优先级
   低于 ENZYMES。

因此更准确的当前结论是：

> Mutagenicity/NCI1 不适合继续证明 Beam8 分类价值；ENZYMES 更适合研究真实结构—连续属性
> 融合，但必须先确认强多模态基线，再用严格未见 controls 判断 Beam8 是否有独立贡献。

## 17. ENZYMES Beam8：不变性修复成功，但特异分类增量未建立

### 17.1 split5/6 结果为什么必须作废

第一次 ENZYMES Beam8 controls 只用3维离散 node labels 定义 canonicalization/orbits，而将
18维连续属性仅作为 patch content。这个设计在离散结构对称、连续属性不同的节点之间留下了
歧义：节点重标号后，cover 可能选择另一个离散等价节点，进而读取不同的连续内容。

在64 graphs × 3 permutations 的审计中，token rows、token multiset、graph readout 与
node-incidence equivariance 的匹配率都只有约 `0.71`。因此 split5/6 的分类数值无论正负都
不是合法证据，已在原报告中标记：

`INVALIDATED_BY_RELABEL_INVARIANCE_FAILURE`

### 17.2 full-feature canonical 修复与合法性

修复后，每个节点的完整21维 feature row 在图内通过确定性的字典序 unique 编号，所得颜色仅
用于 component order、patch canonicalization 和 attributed automorphism orbits。dictionary
的 structural vector 仍只编码3维离散 labels 与单一边类型，保持52维；patch histogram 则
继续读取完整21维属性。

新的不变性审计结果为：

| invariant | match rate | role |
|---|---:|---|
| token rows | 100% | required |
| token multiset | 100% | required |
| graph readout | 100% | required |
| node-incidence equivariance | 100% | required |
| chain exact | 97.92% | diagnostic only |

因此 split7/8 的分类实验在 permutation/relabel 层面是合法的；失败不能再归因于旧的
canonical binding bug。

### 17.3 split7/8 未见 controls

在 split seeds 7/8 × model seeds 0/1/2 × 3 folds 的18个未见单元上，先冻结已确认的
`GIN_FULL_ATTRIBUTES + GLOBAL_STATS residual` 强基线，再以相同容量测试 Beam8 条件增量：

| variant | balanced accuracy |
|---|---:|
| BASE_MULTIMODAL | 54.19% |
| BEAM8_FULL_TRUE | 53.83% |
| BEAM8_FULL_BAG | 53.97% |
| BEAM8_FULL_SHUFFLED | 53.94% |
| BEAM8_CODE_ONLY_TRUE | 54.11% |
| BEAM8_HIST_ONLY_TRUE | 54.89% |
| BEAM8_RANDOM_DICTIONARY_TRUE | 54.61% |

关键配对归因：

| comparison | mean | W/T/L | split7/8 means | model0/1/2 means |
|---|---:|---:|---:|---:|
| FULL−BASE | −0.36pt | 1/11/6 | −0.67 / −0.05pt | −0.17 / −0.59 / −0.33pt |
| FULL−BAG | −0.14pt | 4/9/5 | −0.44 / +0.17pt | −0.25 / −0.17 / +0.01pt |
| FULL−SHUFFLED | −0.11pt | 3/9/6 | +0.05 / −0.27pt | +0.58 / −0.51 / −0.41pt |
| FULL−HIST | −1.06pt | 1/8/9 | −1.83 / −0.28pt | −0.58 / −1.24 / −1.34pt |
| FULL−RANDOM | −0.78pt | 3/8/7 | −1.83 / +0.28pt | −1.00 / −0.74 / −0.59pt |

冻结 gate 要求 FULL 同时超过 BASE、BAG、SHUFFLED、HIST 与 RANDOM，并满足双 split 和多数
model seed 的方向一致性。除不变性外，increment、localization、binding、HIST/RANDOM
specificity、双 split 与 model-majority checks 全部失败，最终判定为：

`ENZYMES_FEATURE_CANONICAL_BEAM8_SPECIFIC_INCREMENT_NOT_ESTABLISHED`

### 17.4 这次失败具体否定了什么

ENZYMES 本身并不是没有结构—属性信号：前面的容量匹配与未见确认已经证明，完整连续属性沿
图结构的组织能在 GLOBAL_STATS 之外提供约10pt的稳定增量。失败的是更窄的主张：当前
Beam8 cover、52维结构字典、21维 patch histogram 与低容量 frozen residual 并没有在这个
强多模态基线上提取出额外的、Beam8-specific 的可分类信息。

尤其是 HIST_ONLY 与 RANDOM_DICTIONARY 都高于 FULL，说明问题不能通过强调精确 binding 或
真实 KSVD dictionary 来解释；更大的融合头可能改变分数，却会绕过本轮要求回答的归因问题。

因此按冻结协议停止 ENZYMES 上的当前 Beam8 分类路线，不进入 cross-attention、监督 atom
gate、graph-supervised dictionary 或更大 residual。Beam8 仍可作为连续 patch 的采样、覆盖、
压缩和 relation-detectability substrate；但若要继续寻找分类正结果，需要更换真实数据任务
或重新定义 Beam8 表示本身，并从新的未见筛选协议开始，而不是继续扩大同一个分类接口。

## 18. Canonical-slot 属性表示：优于 patch mean，但收益不是结构槽位或 Beam8-specific

第17节的 FULL token 将一个 patch 的21维节点属性直接取均值，可能在分类前就丢失了
“哪个连续属性属于哪个结构节点”的对应关系。为检查这个信息瓶颈，在任何 split9/10 结果
可见前冻结新表示：

- 结构仍使用 train-only 24-atom INIT dictionary 的3-sparse code；
- 连续属性按 patch 的 canonical `slot_nodes` 放入8个 slots；
- 每个 slot 保留18维连续属性，并追加8维 occupancy mask；
- FULL patch token 为 `24 + 8×18 + 8 = 176` 维；
- GIN、GLOBAL_STATS、rank-16 frozen residual、dictionary pool 和训练协议均不改变。

### 18.1 TRUE 与 shuffle control 都通过100%不变性

在64 graphs × 3 permutations 上，`SLOT_FULL_TRUE` 与 patch 内属性打乱 control 的 token rows、
token multiset、graph readout 和 mapped node-incidence equivariance 全部为100%。chain exact
仍为97.92%，只作 diagnostic。因此本轮分类在 relabel 层面合法，TRUE−shuffle 可以直接用于
检查属性—结构槽位 binding。

### 18.2 split9/10 的18-unit结果

| variant | balanced accuracy |
|---|---:|
| BASE_MULTIMODAL | 56.71% |
| SLOT_FULL_TRUE | 56.96% |
| SLOT_FULL_BAG | 57.07% |
| SLOT_FULL_INCIDENCE_SHUFFLED | 57.07% |
| SLOT_FULL_WITHIN_PATCH_SHUFFLED | 56.93% |
| SLOT_CODE_ONLY_TRUE | 56.65% |
| MEAN_FULL_TRUE | 56.43% |
| SLOT_RANDOM_DICTIONARY_TRUE | 56.99% |

关键配对：

| comparison | mean | W/T/L | split9/10 means | model0/1/2 means |
|---|---:|---:|---:|---:|
| SLOT FULL−BASE | +0.25pt | 6/8/4 | +0.83 / −0.33pt | +0.92 / −0.08 / −0.09pt |
| SLOT FULL−BAG | −0.11pt | 4/9/5 | +0.05 / −0.27pt | +0.67 / −0.16 / −0.84pt |
| SLOT FULL−incidence shuffle | −0.11pt | 6/6/6 | −0.22 / +0.01pt | +0.51 / +0.35 / −1.18pt |
| SLOT FULL−within-patch shuffle | +0.03pt | 6/7/5 | +0.12 / −0.05pt | +0.59 / −0.24 / −0.25pt |
| SLOT FULL−MEAN FULL | +0.53pt | 7/6/5 | +0.78 / +0.28pt | +0.92 / +0.09 / +0.58pt |
| SLOT FULL−random dictionary | −0.03pt | 2/9/7 | +0.49 / −0.55pt | +0.17 / −0.25 / −0.01pt |

冻结 gate 中只有 `slot-vs-mean` 的均值、双 split 和 model-majority 方向检查通过，但它只有
7/18 wins，未达到12/18。increment、localization、node binding、within-patch binding、真实
dictionary specificity 及其双轴稳定性均失败，最终判定：

`ENZYMES_CANONICAL_SLOT_ATTRIBUTE_BEAM8_INCREMENT_NOT_ESTABLISHED`

### 18.3 新表示真正告诉了我们什么

canonical slots 相比单一 patch mean 保留了更多有用的连续属性信息，这个 `+0.53pt` 的方向
在两个 splits 和三个 model seeds 上都为正，说明“只取均值”确实是一个信息瓶颈。但几乎
相同的 patch 内 shuffled slots 也保留这项收益，说明有用部分更接近：

> patch 内连续属性 row 的丰富分布，而不是这些 rows 与 canonical structural slots 的精确对应。

同时 BAG、node-incidence shuffle 与随机 dictionary 都不弱于 TRUE，进一步排除了三个更强的
方法主张：收益不要求真实 node binding，不要求精确 slot binding，也不要求真实 KSVD atoms。
因此不能把 slot-vs-mean 的小幅正向结果包装成 Beam8 结构—属性融合收益。

这一轮已经按最小表示修改回答了“是不是 patch mean 太粗糙”。答案是“部分是”：更丰富属性
分布优于均值，但它没有转化成稳定的 Beam8-specific 分类增量。按冻结协议停止 ENZYMES Beam8
分类，不继续添加 attention、slot encoder、监督 dictionary 或更大 head；若以后研究 patch
属性分布，应将其作为独立的 set/distribution baseline，而不是继续归入当前 Beam8 方法。

## 19. 直接检查 patch graph：BZR 出现机制信号，但未形成条件分类收益

前面的大部分接口最终都把 patches 汇总成 graph vector 或 node field。为了检查是否遗漏了
Beam8 最独特的“patch 作为节点、overlap/chain 作为边”，新增三个未使用过的小型真实 TU
分子数据：BZR、COX2、DHFR。三者只有405/467/756张图，远小于 OGB。

三个数据集的官方连续属性均为中心化3D coordinates。为避免任意旋转和坐标轴依赖，本轮只用
官方离散 atom labels 与无类型边；不拟合 KSVD，直接使用8个 canonical slots 的离散 labels
和 patch adjacency 作为 RAW token。固定无参数 PREVIOUS/OVERLAP/SLOT pair statistics 后，
统一使用 balanced logistic regression。

### 19.1 三数据集 patch-graph prescreen

| dataset | GLOBAL_STATS | BAG | TRUE patch graph | token shuffled | TRUE−shuffle | TRUE−BAG |
|---|---:|---:|---:|---:|---:|---:|
| BZR | 68.54% | 61.76% | 63.45% | 61.36% | +2.09pt | +1.69pt |
| COX2 | 66.78% | 57.02% | 57.24% | 55.84% | +1.40pt | +0.22pt |
| DHFR | 67.60% | 64.89% | 64.93% | 64.07% | +0.86pt | +0.04pt |

三个数据集 required relabel checks——token rows、relation matrices、TRUE feature 与 shuffled
feature——全部100%通过。mapped patch-set exact 只有34%--63%，再次说明 automorphism 下不应
要求逐节点 cover identity；分类需要的是表示与关系图不变。

COX2 与 DHFR 的 relation increment 或跨 split 方向失败。BZR 则是目前最强的真实 patch-graph
substrate：

- TRUE−token shuffled：`+2.09pt`，7/9，split0/1/2 means 全正；
- TRUE−BAG：`+1.69pt`，7/9，split0/1/2 means 全正；
- 但 TRUE−GLOBAL_STATS：`−5.09pt`，仅3/9，前两个 split 明显为负。

因此 BZR 的正确 token-position relation 在开发 splits 中确实可检测，也能超过相同 token
multiset 的无序聚合；但它不是有竞争力的 standalone classifier。原冻结 gate 失败，不能据此
直接训练 patch-GNN。

### 19.2 BZR 未见 split3/4：GLOBAL_STATS 条件检验

为区分“relation 单独较弱”与“relation 含有独立补充”，在结果不可见前冻结 split3/4 的同维
conditional controls：GLOBAL_ONLY、GLOBAL+BAG、GLOBAL+TRUE、GLOBAL+TOKEN_SHUFFLED。

| variant | balanced accuracy |
|---|---:|
| GLOBAL_ONLY | 69.43% |
| GLOBAL_PLUS_BAG | 62.70% |
| GLOBAL_PLUS_PATCH_GRAPH_TRUE | 63.95% |
| GLOBAL_PLUS_PATCH_GRAPH_TOKEN_SHUFFLED | 64.45% |

配对结果：

- TRUE−GLOBAL：`−5.48pt`，2/6；split3/4 为 `−3.48/−7.48pt`；
- TRUE−BAG：`+1.24pt`，4/6；split3/4 为 `+2.14/+0.34pt`；
- TRUE−TOKEN_SHUFFLED：`−0.50pt`，2/1/3；split3/4 为 `−0.68/−0.32pt`。

TRUE 相对 BAG 的 relation block 增量在两个未见 splits 中仍为正，说明 pair statistics 并非
完全无效；但正确 binding 不胜 shuffle，而且把任一高维 patch block 与 GLOBAL_STATS 联合后
都显著恶化，TRUE 没有独立净增量。最终判定：

`BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_INCREMENT_NOT_ESTABLISHED`

### 19.3 对“Beam8 能否分类”的最新回答

BZR 结果比此前更接近 Beam8 的原始思想：不用 KSVD、不经过 node residual，直接读取 patch
graph，就能在多个开发 splits 上稳定检测真实 relation binding。这说明 Beam8 patch graph 并非
完全没有分类信号，不能笼统地说“Beam8 无法分类”。

但未见条件检验又说明，这个信号目前具有两项限制：

1. 它弱于简单全图统计，并且没有在统计条件后带来净收益；
2. TRUE−shuffle 没有在新 splits 复现，relation block 的增量更像总体 pair/distribution
   information，而不是稳定的具体 patch identity composition。

因此当前最准确的结论是：

> Beam8 patch graph 能暴露可检测的关系信号，但尚未证明该信号是强基线之外稳定、正确绑定
> 特异且可泛化的分类信息。

按冻结协议不训练一层 patch-GNN。继续扩大模型可能提高训练或单次 CV 分数，却会越过已经失败
的条件增量与 binding gates。Beam8 仍适合报告为连续 cover 与 relation substrate；若未来有
标签确实由中尺度组合关系决定、且低阶统计不占主导的新真实任务，再从相同低容量 prescreen
开始，而不是在当前 TU 数据上继续增加容量。
