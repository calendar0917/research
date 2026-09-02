# KSVD on OGBG-MolHIV：冻结后最终实验报告

**报告日期：2026-07-26**  
**冻结配置 ID：** `molhiv-r2-d32-t3-h64l3-zscore-signed-alllayer-v1`  
**主指标：** ROC-AUC，OGB `ogbg-molhiv` official scaffold split  
**最终状态：** 配置已冻结，5-seed official-valid / controlled official-test 均已完成

---

## 0. 一页结论

### 0.1 最终回答

1. **KSVD 确实提供了可重复的结构信号，但目前没有在 official test 上稳定超过 GINE。**
   - 5-seed official-valid：KSVD `0.79457 ± 0.03744`，GINE-h64 `0.76995 ± 0.03241`；paired mean delta `+0.02461`，3/5 wins。
   - 5-seed official-test：KSVD `0.75189 ± 0.01514`，GINE-h64 `0.75270 ± 0.01469`；paired mean delta `-0.00080`，2/5 wins。
   - 对参数匹配的 GINE-h70，KSVD test 还低 `-0.01139`。

2. **5-seed probability ensemble 能显著降低 seed 方差，但仍未达到 CIN。**
   - KSVD ensemble：valid `0.82861`，test `0.77036`。
   - GINE-h64 ensemble：valid `0.79862`，test `0.77521`。
   - 参数匹配 GINE-h70 ensemble：valid `0.81064`，test `0.77762`。
   - 文献 CIN：MolHIV 约 `0.8094 ± 0.0057`；当前 KSVD ensemble test 仍差约 `0.039` AUC。

3. **参数效率是真实优势，而且幅度不小。**
   - 本方案 KSVD：`43,655` 个 downstream trainable parameters。
   - 若连同固定 KSVD 字典 `848×32=27,136` 个值也计入，总 stored learned values 为 `70,791`。
   - CIN-small 官方实现：`138,385` trainable parameters。
   - CIN 官方 h64：`239,809` trainable parameters。
   - 因而本方案 trainable parameters 约为 CIN-small 的 `31.5%`、CIN 的 `18.2%`；即分别小约 `3.17×` 与 `5.49×`。
   - 即便把固定字典完整计入，本方案仍小约 `1.95×` 和 `3.39×`。

4. **继续单纯加宽、加深或增大字典已经显示为负收益。**
   - KSVD h64/l4：相对 h64/l3 inner mean `-0.00150`，且方差大幅增加。
   - KSVD h80/l3：`-0.01872`。
   - D48/T4：`-0.07070`，是明确失败。
   - node-adaptive router：`-0.01498`，3/3 失败。

5. **所以当前正确叙事不是“已经达到 CIN”，而是：**

> 我们得到了一条参数高效、以无监督 KSVD 字典为核心、在 official-valid 上显示明显结构增益的路线；但这个增益没有可靠迁移到 MolHIV scaffold test。当前瓶颈不是容量不足，而是结构表示与训练/选择协议的 scaffold-robust generalization 不足。

---

## 1. 研究问题与方法边界

### 1.1 核心研究问题

目标不是把 CIN 的显式 ring lifting 搬进来再改名，而是验证：

> 能否通过 **无监督 KSVD 学习出来的局部结构字典**，给普通分子 GNN 提供参数高效、可解释、可泛化的高阶结构信号？

因此本路线必须保持以下边界：

- KSVD 字典是方法核心，而不是附属可有可无的特征工程；
- 字典只从训练图 patch 学习；
- 不用标签学习字典；
- 不把 induced ring / 5、6 元环直接作为最终主路线的显式特征；
- 与 GINE 对照时尽量保持初始化、数据顺序、epoch 选择等配对条件一致；
- 对 official-valid 与 official-test 的使用进行严格记录。

### 1.2 与 CIN 的方法论区别

| 维度 | CIN | 当前 KSVD 路线 |
|---|---|---|
| 结构对象来源 | 预先枚举 induced cycles，长度上限人工指定 | 从训练数据的局部 patch 中无监督学习字典原子 |
| 高阶对象 | ring 被显式 lift 为 2-cell | dictionary activation 是每个 atom 的 learned structural token |
| 消息传递 | 在 0/1/2-cell complex 上传递 | 在 GINE node states 中注入 KSVD token |
| 化学先验 | 强、明确、拓扑定义 | 弱、数据驱动、非标签监督 |
| 参数量 | 多维 cell 通路，较大 | 小 GINE backbone + 低维 token projection |
| 研究卖点 | 显式拓扑表达力 | learned dictionary、参数效率、非手工 motif |

这一区别必须保留。显式 ring ablation 可以作为诊断实验，但不应成为最终方法核心，否则会削弱“KSVD 自学习结构原子”的主张。

---

## 2. 最终方法

### 2.1 局部 patch

对每个原子节点构造 radius-2 ego patch，并变换为 permutation-invariant 的固定维表示。最终 patch 向量维度为 `848`。

关键选择：

- `radius = 2`；
- 每个 patch 最多 `8` 个节点；
- 结构与原子/键信息在 canonicalized patch 表示中编码；
- 字典只从 official-train patch 拟合；
- 最大字典训练 patch 数 `6000`。

### 2.2 KSVD 字典与 OMP code

冻结配置：

- dictionary atoms `D = 32`；
- OMP sparsity `T = 3`；
- KSVD iterations `3`；
- dictionary shape `848 × 32`；
- 每个节点得到 32 维 signed sparse code。

字典本身不在下游 GINE 训练中更新。它是从训练 patch 无监督学习的固定结构基底。

### 2.3 Token normalization

对节点 sparse code 使用 train-only z-score：

- inner selection 阶段只用 inner-train 统计量；
- full retrain 阶段只用 official-train 统计量；
- official-valid/test 均只应用已拟合统计量，不反向影响 normalization。

最终保留 signed coefficients，不额外拼接 absolute magnitude 或 support mask，因为这些扩展没有通过 inner-only screen。

### 2.4 与 GINE 融合

最终 backbone：

- hidden `64`；
- GINE layers `3`；
- dropout `0`；
- learning rate `1e-3`；
- batch size `128`。

KSVD token 经 projection 后，在每个 GINE layer 注入。每层使用一个 **zero-initialized scalar gate**：

- 初始时模型严格等价于 GINE baseline；
- 训练中网络自行决定每一层是否以及以何符号使用 KSVD 通路；
- 避免一开始用未校准 token 扰乱 backbone；
- 相比 node-specific router 参数更少、泛化更稳。

seed 0 最终 gate 为：

```text
[-0.3727, +0.0571, +0.2335]
```

它说明不同层确实学习了不同符号和幅度的 KSVD 注入，而不是所有 gate 始终停在零点。

---

## 3. 实验协议与防泄漏设计

### 3.1 数据划分

OGB official scaffold split：

| split | 图数 | 正样本数（已记录部分） |
|---|---:|---:|
| official train | 32,901 | 1,232 |
| official valid | 4,113 | 81 |
| official test | 4,113 | — |

每个 neural seed 使用 official train 内部固定 stratified split：

- inner split seed `1729`；
- inner-valid fraction `0.15`；
- inner-train `27,965`；
- inner-valid `4,936`；
- inner-train positives `1,047`；
- inner-valid positives `185`。

### 3.2 Epoch 选择

每个 seed 的流程：

1. 在 inner-train 训练最多 30 epochs；
2. 只根据 inner-valid ROC-AUC 选 epoch；
3. 用相同 seed 在完整 official train 上从头重训到 selected epoch；
4. official-valid 评估一次；
5. 冻结后 terminal suite 中 official-test 评估一次。

没有使用 official-test 选择 epoch、模型宽度、字典大小、融合方式或 seed。

### 3.3 Frozen-test cache

原始 full cache 的 official-test token rows 保持全零，防止探索阶段无意使用 test structural codes。

冻结后通过：

```text
code/encode_molhiv_node_tokens_test.py
```

执行以下操作：

- 不重新学习或更新 dictionary；
- 精确复制 train/valid token rows；
- 只对 4,113 个 official-test graphs 编码；
- 已验证所有 test graphs 均有非零 token；
- dictionary hash：`c5179600...d6cfbb`。

### 3.4 重要 disclosure

仓库中的早期 feasibility 文件曾查看过 official test。因此本轮只能称为：

> **冻结后受控 terminal evaluation**，不能声称是完全 untouched test。

冻结之后没有根据 test 结果选择或修改模型。当前 test 数字可以用于判断既定方案是否泛化，但不能再被用于发起一轮以提升 test 为目标的超参数选择，并继续宣称同一个 test 是无偏终局评估。

---

## 4. 从早期 KSVD 到 localized node token 的路线演化

### 4.1 Standalone graph-level KSVD

最早路线将 graph patches 编码后做分布统计，并接 balanced logistic regression。

主要结果：

- KSVD-rich valid AUC：`0.71213 ± 0.00988`；
- size-only：`0.67874`；
- 相同 patch pool 的 random-patch control：KSVD 平均 `+0.02750`，5/5 wins；
- matched PCA-rich：`0.65432`。

结论：

- KSVD 字典学习不是完全无效；
- 相对 random patch 和 PCA，确实能学出更有用的结构基底；
- 但 graph-level 聚合损失了“哪个节点激活哪个原子”的局部对应关系，上限明显不足。

### 4.2 Graph-level residual to GINE

将 graph-level KSVD readout 作为 GINE residual 后，提升很小且不稳定：

- 8000 图、5 neural seeds：4/5 wins；
- paired mean 仅约 `+0.00164`；
- 存在 `-0.04273` outlier。

结论：结构向量放在图级太晚，难以在 message passing 中参与局部组合。

### 4.3 Localized node tokens

关键突破是把每个 atom-centered patch 的 sparse code 保留在节点层面，并注入每一层 GINE。

这使方法从：

```text
graph → many patches → one graph vector → classifier
```

变为：

```text
each atom → local patch → KSVD sparse code
             ↓
GINE layer-wise message passing with localized structural tokens
```

这是当前 valid 提升的主要来源，也是比 standalone / graph residual 更合理的结构使用方式。

---

## 5. 冻结前 official-valid 结果

### 5.1 五个 seed

| seed | GINE h64/l3 | KSVD h64/l3 D32/T3 | paired delta |
|---:|---:|---:|---:|
| 0 | 0.731800 | 0.805644 | +0.073844 |
| 1 | 0.813801 | 0.767845 | -0.045956 |
| 2 | 0.778448 | 0.855551 | +0.077102 |
| 3 | 0.744828 | 0.768476 | +0.023647 |
| 4 | 0.780895 | 0.775328 | -0.005567 |
| **mean** | **0.769955** | **0.794569** | **+0.024614** |
| **sample std** | 0.032409 | 0.037439 | 0.052609 |

统计：

- KSVD wins：3/5；
- paired t-test p：`0.3545`；
- 由于 n=5 且方差较大，不能把 valid 提升描述为统计显著；
- 但从平均提升、3/5 wins 和两个大幅正增益 seed 看，KSVD 通路包含真实可利用信号。

### 5.2 Valid ensemble

五个 seed 的 positive-class probability 做算术平均：

| family | official-valid ensemble AUC |
|---|---:|
| GINE h64 | 0.798620 |
| GINE h70 parameter-matched | 0.810635 |
| **KSVD h64 D32/T3** | **0.828606** |

ensemble 上 KSVD 的 valid 优势更明显：

- 相对 GINE-h64：`+0.029985`；
- 相对 GINE-h70：`+0.017970`。

这表明不同 seed 的 KSVD 模型可能捕获了互补排序信号。不过该结论必须结合 test ensemble 一起看。

---

## 6. 冻结后 official-test 结果

### 6.1 单模型五 seed

| seed | GINE h64 test | GINE h70 test | KSVD test |
|---:|---:|---:|---:|
| 0 | 0.763371 | 0.764752 | 0.755633 |
| 1 | 0.727673 | 0.774426 | 0.741803 |
| 2 | 0.755341 | 0.776743 | 0.736453 |
| 3 | 0.763398 | 0.754885 | 0.775548 |
| 4 | 0.753694 | 0.745596 | 0.750030 |
| **mean** | **0.752695** | **0.763280** | **0.751893** |
| **sample std** | 0.014685 | 0.013142 | 0.015143 |
| **SEM** | 0.006567 | 0.005877 | 0.006772 |
| **min–max** | 0.727673–0.763398 | 0.745596–0.776743 | 0.736453–0.775548 |

### 6.2 Paired test comparison

KSVD minus GINE-h64：

```text
[-0.007739, +0.014129, -0.018888, +0.012150, -0.003664]
mean = -0.000802
wins = 2/5
paired t-test p = 0.9036
```

KSVD minus parameter-matched GINE-h70：

```text
[-0.009120, -0.032623, -0.040290, +0.020663, +0.004434]
mean = -0.011387
wins = 2/5
paired t-test p = 0.3719
```

结论：

- 相对小 GINE-h64，test 上基本打平，而非稳定更强；
- 相对参数匹配 GINE-h70，test 上平均更弱；
- 因此无法支持“当前 KSVD 已经强于 GINE”的最终主张。

### 6.3 Test ensemble

| family | official-test ensemble AUC |
|---|---:|
| **GINE h70 parameter-matched** | **0.777620** |
| GINE h64 | 0.775208 |
| KSVD h64 D32/T3 | 0.770363 |

与 valid ensemble 排名相反：

- KSVD 相对 GINE-h64：`-0.004846`；
- KSVD 相对 GINE-h70：`-0.007258`。

ensemble 能把 KSVD 从单模型均值 `0.7519` 提升到 `0.7704`，说明 seed averaging 有价值；但 GINE ensemble 同样受益，而且 test 上更高。

---

## 7. 为什么 valid 接近/超过 0.80，test 却只有约 0.75–0.77

### 7.1 Seed 级 valid-test mismatch

五 seed 内，valid 最好的 seed 并不对应 test 最好的 seed：

- KSVD seed 2：valid `0.855551`，却是最低 test `0.736453`；
- KSVD seed 3：valid 仅 `0.768476`，却是最高 test `0.775548`。

KSVD 五 seed valid-test Pearson correlation 为 `-0.534`。样本数只有 5，不能把相关系数作稳定统计结论，但它明确说明：**当前 valid ranking 不适合用于选择部署 seed。**

同样地：

- GINE-h64 correlation `-0.914`；
- GINE-h70 correlation `-0.091`。

因此问题不只属于 KSVD；当前单 inner split + 单 official-valid 的 epoch/seed 稳定性，对 MolHIV scaffold generalization 本身就较弱。

### 7.2 可能机制

#### A. KSVD 学的是训练 scaffold 的重构基底，不是跨 scaffold 判别基底

标准 KSVD 优化 patch reconstruction。它可能把训练 scaffold 中频繁出现、容易重构的模式学得很好，但这些模式不一定是跨 scaffold 保持标签相关性的化学结构。

#### B. z-score 会把低频 activation 放大

train-only z-score 本身没有泄漏，但稀疏 code 中低频 atom 的尺度可能在新 scaffold 上不稳定。valid 上偶然有利的 rare activation，test 上可能转为噪声。

#### C. 单一 inner split 对 epoch 的选择噪声较大

MolHIV 正样本少，inner-valid 只有 185 个正例。ROC-AUC 的 epoch 曲线容易受少数排序变化影响。当前每个 seed 只用一个固定 inner split，selected epoch 的估计方差仍可能很大。

#### D. 当前 token 是局部结构摘要，尚未学到 chemistry-preserving motif equivalence

radius-2 patch 同时混合拓扑、原子与键属性，但 KSVD 的欧氏重构几何未必与分子 scaffold 变化下的任务相似性一致。

### 7.3 最关键判断

当前瓶颈不是“模型看得不够大”或“参数不够多”，而是：

> **哪些 learned atoms 在新 scaffold 上仍然语义稳定，以及如何避免模型利用只在 train/valid 成立的 activation。**

---

## 8. 参数量审计：是否比 CIN 小很多？

### 8.1 我们的模型

| 模型 | trainable parameters | fixed dictionary values | 合计 stored learned values |
|---|---:|---:|---:|
| GINE h64/l3 | 37,380 | 0 | 37,380 |
| GINE h70/l3 | 43,404 | 0 | 43,404 |
| **KSVD h64/l3 D32** | **43,655** | **27,136** | **70,791** |

KSVD 相对 GINE-h64 只增加：

```text
6,275 trainable parameters，约 +16.8%
```

这部分主要来自 32 维 token projection 和 3 个 layer gates。

### 8.2 CIN 官方代码参数量复核

基于 `twitter-research/cwn` 官方仓库 commit：

```text
c4ddd24e251929f934f8a2467da98fdba376864d
```

MolHIV 官方脚本：

- `exp/scripts/cwn-molhiv-small.sh`：2 layers, hidden 48；
- `exp/scripts/cwn-molhiv.sh`：2 layers, hidden 64；
- `exp/scripts/cin++-molhiv.sh`：2 layers, hidden 64。

按官方 model definition 实例化并统计 trainable tensors：

| 官方模型 | trainable parameters | 相对本 KSVD trainable | 相对本 KSVD 含字典 |
|---|---:|---:|---:|
| CIN-small h48 | 138,385 | 3.17× | 1.95× |
| CIN h64 | 239,809 | 5.49× | 3.39× |
| CIN++ h64 | 365,377 | 8.37× | 5.16× |

因此“参数比 CIN 少非常多”是成立的，但论文中应同时报告两种口径：

1. **下游可训练参数：43,655**；
2. **包含固定字典的 stored learned values：70,791**。

只报 43,655 而完全不披露字典，会被质疑把学习得到的参数藏在预处理里；只报 70,791 又会低估“不需要反向传播和 optimizer state”的训练优势。双口径最公平。

### 8.3 参数效率能否成为核心贡献

可以，但不能单独成立。合理主张是：

> 在约 4.4 万 downstream trainable parameters 下，KSVD-GINE valid mean 接近 0.795，ensemble valid 达到 0.829；其参数量显著低于 CIN，但 terminal test 尚未复现 CIN 级泛化。

参数效率应该与以下证据配套：

- parameter-matched GINE-h70 control；
- FLOPs / peak memory / wall-clock；
- dictionary learning 与 token encoding 的离线成本；
- 单模型和 ensemble 的 test；
- 不把 ensemble 的总训练成本伪装成单模型成本。

---

## 9. 容量 scaling：继续往上堆会不会适得其反？

### 9.1 已完成 inner-only screen

8000 图、3 seeds、只看 inner-valid：

| 配置 | 参数量 | inner mean | std | 相对 KSVD h64/l3 |
|---|---:|---:|---:|---:|
| KSVD h64/l3 reference | 43,655 | 0.766344 | 0.001972 | — |
| GINE h70/l3 | 43,404 | 0.766810 | 0.015727 | +0.000466 |
| KSVD h64/l4 | 52,105 | 0.764841 | 0.017716 | -0.001503 |
| KSVD h80/l3 | 63,527 | 0.747625 | 0.017807 | -0.018719 |

结论非常明确：

- 增加一层没有带来平均收益，却把 std 从 `0.0020` 放大到 `0.0177`；
- hidden 从 64 提到 80 明显变差；
- 更大 backbone 更容易吸收训练 scaffold 的偶然相关性；
- 当前小模型不是明显 underfit 状态。

### 9.2 Dictionary scaling

D48/T4：

```text
inner values = [0.684253, 0.701479, 0.701209]
mean = 0.695647
relative to D32/T3 = -0.070697
wins = 0/3
```

这是最强的反证之一。更大字典并没有提供更丰富的有用 motif，反而：

- 增加高度相关或低支持 atom；
- 让 OMP support 选择更不稳定；
- 放大 train-only z-score 后的稀有 activation；
- 增加下游 token projection 的优化难度。

### 9.3 Adaptive router

node-specific per-layer router：

```text
adaptive mean = 0.751363
reference mean = 0.766344
delta = -0.014981
wins = 0/3
```

router 虽然“更智能”，但它给了模型过多自由度去学习 scaffold-specific token usage。简单 global scalar gate 反而是有效 regularizer。

### 9.4 容量结论

因此答案不是“永远不能增大”，而是：

- **不能继续用无约束的 width/depth/D scaling；**
- 如果增加容量，必须同时增加明确的跨 scaffold 约束；
- 新增参数应放在“让字典原子更稳定/可迁移”的位置，而不是普通 GINE backbone。

当前推荐保留 h64/l3/D32/T3 作为参数效率 reference，不再扩大主干。

---

## 10. 主要失败路线与得到的知识

### 10.1 Radius 1+2 multiscale

普通 radius-1 + radius-2 拼接没有优于单 radius-2。原因可能是：

- radius-2 已包含 radius-1 的多数信息；
- 多尺度简单拼接造成冗余；
- 小数据/少正例下额外 channel 增加优化噪声。

因此此前问题的答案是：**1+2 radius 不如最终单独 radius-2。**

### 10.2 显式 ring variants

尝试过：

- ring-member injection；
- ring-member readout；
- typed ring；
- graph ring statistics。

最佳 conservative ring-member readout（gate scale 0.25）：

```text
mean gain = +0.00077
wins = 1/3
```

没有通过预声明 `+0.003`、至少 2/3 wins 的 advancement gate。

方法论上也不建议把显式 ring 变成主路线：这会向 CIN 的手工拓扑对象靠拢，削弱 KSVD 自学习结构字典的创新点。

### 10.3 Support / magnitude channels

额外暴露 coefficient absolute magnitude 与 exact OMP support 没有改善。signed coefficient 已包含主要信息；额外 channel 可能使网络更容易依赖 activation frequency 而非稳定语义。

### 10.4 Token regularization variants

尝试过：

- token dropout；
- gate L2；
- limited-layer injection；
- reduced gate scale。

没有形成稳定超过全层、zero-init global gate 的配置。

### 10.5 Motif-slot readout

按 dictionary atom slot 做图级 motif aggregation/readout 没有通过 screen。它可能重新退化为 graph-level histogram，丢掉 localized message passing 的优势。

---

## 11. 当前可以和不可以做的主张

### 11.1 可以做

- KSVD 学习到的 dictionary atoms 比 matched random patches / PCA 更有用；
- localized node-token 比 graph-level KSVD readout 更有潜力；
- 在 frozen official-valid 五 seed 上，KSVD 相对 GINE-h64 平均 `+0.0246`；
- 当前模型只有 `43,655` trainable parameters，显著少于官方 CIN；
- 增大 width、depth、dictionary 和 router 均未提升，说明 compactness 不是偶然遗漏；
- 5-seed ensemble 能显著提升稳定性。

### 11.2 不能做

- 不能说已经稳定到 official test 0.80；
- 不能说已经达到或超过 CIN；
- 不能说在 test 上稳定强于 GINE；
- 不能把 seed 2 的 valid `0.8556` 当作可复现总体性能；
- 不能称本轮 test 为 untouched test；
- 不能只报 43,655 而不披露固定字典；
- 不能根据本轮 test 再选择新配置后继续把同一 test 当最终无偏评估。

---

## 12. 下一阶段可行突破方向

下面按“保持 KSVD 核心”与“最有可能解决 test 泛化”排序。

### 路线 A：多 inner-fold 的 epoch / checkpoint 稳定选择

**目的：** 先解决 selection noise，而不是改模型。

建议：

1. official train 内固定 3 个 stratified inner folds；
2. 每个 epoch 计算跨 fold mean AUC 或 rank-stability；
3. 选择平均最好、且 fold variance 受控的 epoch；
4. 最终仍只在 official train 重训一次；
5. 不查看 official test。

风险：计算约为当前 inner selection 的 3 倍。  
收益：很可能比继续加参数更直接地缓解 valid-test seed mismatch。

由于 official-test 已经看过，后续开发需要建立新的内部 scaffold split 或外部数据集作为 development benchmark，不能继续拿 official-test 调参。

### 路线 B：Scaffold-aware dictionary stability，而非显式 ring

保持 KSVD 无监督目标，但要求 atom 在不同 train scaffolds 上稳定：

- 将 official-train 按 scaffold 分组；
- 学多个 bootstrap/scaffold dictionaries；
- 计算 atom matching / activation stability；
- 只保留跨 scaffold 稳定 atom，或对不稳定 atom 加 shrinkage；
- 固定 D32 上限，不增加字典大小。

这是当前最符合研究初衷的突破：不是告诉模型“环是什么”，而是让数据自己学出 **跨 scaffold 可重复的 dictionary atoms**。

### 路线 C：Group-sparse / structured OMP

当前每个节点独立 OMP。可考虑让同一 molecule 内邻近节点的 support 更平滑：

- group OMP；
- graph-fused sparse coding；
- 邻接节点 support consistency regularizer；
- molecule-level atom usage budget。

目的不是增加 atom 数，而是减少单节点 noisy support flip。

注意：应先在 inner scaffold splits 上验证；实现复杂度高于路线 A/B。

### 路线 D：Discriminative but label-safe dictionary selection

不要直接用 official labels 端到端更新 dictionary；可以尝试：

- 先无监督学习较大的 candidate dictionary；
- 只在 official-train inner splits 内做 atom stability / mutual-information selection；
- 外层严格重新拟合；
- 保持最终 dictionary 小于等于 32 atoms。

这会从纯 reconstruction 转向 task-relevant atom selection，但协议必须 nested，避免 feature-selection leakage。

### 路线 E：自监督 chemistry-preserving patch metric

KSVD 的欧氏重构距离可能不对应化学结构相似性。可先学习一个不使用 HIV label 的 patch embedding：

- masked atom/bond reconstruction；
- augment-consistency；
- scaffold-preserving contrastive pretraining；
- 再在 embedding 空间做 KSVD。

KSVD 仍负责学习 dictionary 和 sparse decomposition；自监督 encoder 只改变 patch metric。

这是潜在上限最高、但工作量最大的路线。

### 路线 F：效率主线的完整测量

如果参数效率要成为论文卖点，下一步应测：

- trainable parameters；
- dictionary stored values；
- optimizer state memory；
- peak GPU/CPU memory；
- train seconds / epoch；
- dictionary fitting time；
- test encoding time；
- 单模型与 5-seed ensemble 总成本。

当前 frozen test 编码 4,113 图、103,927 节点约 `50.8 s`，这说明离线 token encoding 成本是可控的，但还需要统一硬件下与 CIN/GINE 对比。

---

## 13. 推荐路线与停止条件

### 13.1 当前冻结 reference

继续保留：

```text
radius 2
D32 / OMP T3
hidden64 / GINE 3 layers
zscore signed token
all-layer zero-init scalar gate
```

它是目前最合理的 **compact KSVD reference**，不是因为 test 最好，而是因为：

- 它是冻结前最强 KSVD 配置；
- 在 valid 上有结构信号；
- 参数量小；
- capacity/dictionary/router scaling 均失败；
- 机制最简洁，最容易归因。

### 13.2 下一轮优先顺序

1. 建立只基于 official-train 的多 scaffold development protocol；
2. 做 checkpoint/epoch stability；
3. 做 cross-scaffold dictionary atom stability pruning；
4. 之后才考虑 structured sparse coding；
5. 暂停 width/depth/D 的普通 scaling；
6. 暂停显式 ring 主路线。

### 13.3 晋级门槛

建议新路线至少满足：

- 3 个或以上 scaffold inner folds；
- 相对 frozen KSVD reference mean `≥ +0.005`；
- 至少 2/3 或 3/5 paired wins；
- fold/seed std 不明显增加；
- 参数量增幅不超过 25%，除非收益超过 `+0.01`；
- 不使用 official-test 做选择。

否则停止，不再把小幅单 seed 波动当突破。

---

## 14. 复现文件

### 14.1 冻结配置与汇总

```text
results/molhiv/node_token_frozen_config_v1.json
results/molhiv/frozen_test_5seeds_summary.json
results/molhiv/cin_parameter_audit.json
```

### 14.2 Token cache

```text
results/molhiv/node_tokens_full_a32_t3.npz
results/molhiv/node_tokens_full_a32_t3_frozen_test.npz
results/molhiv/node_tokens_full_a32_t3_frozen_test.json
```

### 14.3 逐 seed final results

```text
results/molhiv/frozen_test_ksvd_seed{0..4}.json
results/molhiv/frozen_test_gine_h64_seed{0..4}.json
results/molhiv/frozen_test_gine_h70_seed{0..4}.json
```

### 14.4 核心代码

```text
code/build_molhiv_node_tokens.py
code/molhiv_node_tokens.py
code/run_molhiv_node_tokens_inner.py
code/encode_molhiv_node_tokens_test.py
code/summarize_molhiv_frozen_test.py
```

### 14.5 关键 screen summaries

```text
results/molhiv/node_token_capacity_inner_only_screen_summary.json
results/molhiv/node_token_adaptive_router_inner_only_screen_summary.json
results/molhiv/node_token_dictionary_scaling_inner_only_screen_summary.json
results/molhiv/node_token_multiscale_inner_only_screen_summary.json
results/molhiv/node_token_stability_inner_only_screen_summary.json
results/molhiv/ring_cell_inner_only_screen_summary.json
```

---

## 15. 复现命令模板

环境：

```bash
cd /home/calendar/code/research/tracks/ksvd
PY=/home/calendar/.conda/envs/gsn-official/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

冻结 KSVD seed 0 terminal evaluation：

```bash
$PY code/run_molhiv_node_tokens_inner.py \
  --token-cache results/molhiv/node_tokens_full_a32_t3_frozen_test.npz \
  --family ksvd \
  --hidden 64 \
  --layers 3 \
  --epochs 30 \
  --max-graphs 0 \
  --seed 0 \
  --data-seed 0 \
  --inner-split-seed 1729 \
  --token-normalization zscore \
  --token-channels signed \
  --token-fusion inject \
  --evaluate-test \
  --frozen-config-id molhiv-r2-d32-t3-h64l3-zscore-signed-alllayer-v1 \
  --output results/molhiv/frozen_test_ksvd_seed0.json
```

汇总：

```bash
$PY code/summarize_molhiv_frozen_test.py \
  --results-dir results/molhiv \
  --output results/molhiv/frozen_test_5seeds_summary.json
```

注意：上述 `--evaluate-test` 只能用于复现已冻结 terminal evaluation，不应再用于新的超参数选择。

---

## 16. 最终判断

### 科学结论

KSVD 的价值已经从“可能只是随机 patch 特征”推进到更明确的层次：

- dictionary learning 相对 random/PCA 有优势；
- localized sparse code 能在 GINE 层内被利用；
- valid 上平均提升较大；
- 参数效率显著优于 CIN。

但 terminal test 同时揭示了核心缺陷：

- valid 增益没有可靠转移到新 scaffold；
- parameter-matched GINE-h70 test 更强；
- 当前仍未达到 CIN 约 0.81 的水平；
- 继续扩大普通模型容量会适得其反。

### 项目决策

**不放弃 KSVD，但停止“堆模型”路线。**

下一阶段应把创新集中到：

> **跨 scaffold 稳定的 learned dictionary atoms + 更稳健的内部选择协议**。

这既保持 KSVD 的研究初衷，也直接针对本轮 test 暴露出的真实瓶颈。参数量小是重要优势，但只有当泛化差距继续缩小时，它才能从“附加亮点”升级为方法的核心竞争力。
