# XGBoost / KSVD 路线实验总整理：MolHIV 与 ZINC

> 截止日期：2026-09-04
> 范围：本仓库中与 KSVD、局部 patch、XGBoost 下游建模直接相关的 MolHIV / ZINC 实验；早期合成图与 MUTAG 仅用于解释路线为什么改变。
> 重要说明：本文不是把所有数字放进一个排行榜，而是按实验目的、协议和证据强弱整理。不同数据集、指标和 split 不直接横比。

## 0. 先看结论

这条路线实际上包含两条容易混淆的分支：

```text
真正的 KSVD 路线：局部 patch → 共享 KSVD 字典 → 稀疏码 → LR/GINE/MIL/关系模型

XGBoost 路线：固定的拓扑/化学/统计特征 → XGBoost
                         ↑
                 很多实验明确关闭了 KSVD
```

因此，XGBoost 得到高分，不能自动说明 KSVD 有效。判断 KSVD 是否贡献了什么，必须看 `raw / INIT / FINAL / random / PCA` 等匹配对照。

### 当前最可靠的判断

| 问题 | 目前答案 |
|---|---|
| 共享 KSVD 字典能否工作？ | 能。合成三角任务约 `0.55→0.86`，说明跨图共享 atom 身份是必要的。 |
| KSVD 在合成的确需要大感受野任务上是否有用？ | 能。C4/C8 中严格 1-hop 约 `0.842`，RW + 共享 KSVD 约 `0.996–1.000`。 |
| MolHIV 上 KSVD 字典是否比随机字典/PCA 有用？ | 有。独立图级 rich readout 的 valid AUC 为 `0.71213`，高于 random-patch `0.68463` 和 PCA `0.65432`。 |
| MolHIV 上 KSVD 是否稳定超过强 GNN？ | 没有。localized KSVD-GINE 的 valid 均值高于 GINE，但 controlled test 基本打平且略低。 |
| MolHIV 上 XGBoost 是否能做出强结果？ | 能。clean 的 `S+marginal` official-valid 为 `0.841285`；但它不是 KSVD 结果。 |
| ZINC 上 K-SVD update 是否是主要收益来源？ | 没有证据。半径变大、保留 raw typed distribution 是主要收益，`FINAL` 未稳定优于 `INIT/raw`。 |
| ZINC 上当前 XGBoost 最强结果来自哪里？ | 来自 train-only 的 typed rooted-WL motif count 与 hierarchical backoff；该路线关闭 KSVD。 |

一句话总结：**KSVD 作为可还原、共享、稀疏的结构字典已经被证明“有信息”，但在 MolHIV/ZINC 上，当前最强的下游增益主要来自局部对象定义、分布统计和 XGBoost 的结构化读出，而不是 K-SVD 的重构更新本身。**

---

## 1. 统一背景：实验到底在比较什么

后文每组实验都按“目的 → 问题 → 协议/设置 → 结果 → 解释 → 决定”展开。表中的 `valid` 和 `test` 若未特别注明，分别指 official split；内部 scaffold fold 或小切片会明确标出。

### 1.1 KSVD 的基本含义

对一批局部 patch 向量组成的矩阵 `Y`，KSVD 学一个共享字典 `D`，并用 OMP 求稀疏系数 `X`：

```text
Y ≈ D X
```

- `D` 的每一列是一个 learned atom；
- `X` 表示每个 patch 使用哪些 atom、使用强度是多少；
- `INIT` 是 KSVD 更新前的初始化字典/稀疏重构；
- `FINAL` 是运行若干轮 KSVD 更新后的结果；
- `raw` 是未经字典压缩的原始 patch 统计；
- `random-patch`、`random direction`、`PCA` 是匹配对照。

最关键的判别不是“重构误差有没有下降”，而是：**重构更好以后，下游任务是否也更好。** 多次实验显示二者经常不一致。

### 1.2 XGBoost 路线的基本含义

XGBoost 路线把每个分子的局部 patch 转成固定长度统计向量，例如：

- 全图节点数、边数、原子/键组成；
- 每个中心的 rooted-WL 拓扑角色和化学属性分布；
- patch 内的结构—属性共现；
- 不同中心之间的协方差；
- ZINC 上多层 typed-WL token 的计数和低阶 backoff。

这类特征通常直接送入 `XGBoost(binary:logistic)` 或 `XGBoost(reg:absoluteerror)`。如果没有执行字典学习或 sparse coding，就只能称为 **XGBoost 统计路线/非 KSVD 对照**。

### 1.3 两个数据集

| 数据集 | split | 任务 | 指标 |
|---|---:|---|---|
| OGB `ogbg-molhiv` | train/valid/test = `32901/4113/4113`；已记录正例 train/valid/test = `1232/81/130` | HIV 活性二分类 | ROC-AUC，越高越好 |
| PyG `ZINC(subset=True)` | train/valid/test = `10000/1000/1000` | 分子性质回归（penalized logP / constrained solubility） | MAE，越低越好 |

### 1.4 评估边界

本文会区分三种结果：

1. **机制/开发筛选**：只在 official-train 内部做 scaffold folds 或小切片；用于决定是否继续，不是最终榜单。
2. **official-valid 冻结评估**：特征、字典、PCA 和超参数先只用 train 拟合，再评估 valid。
3. **controlled terminal test**：方案冻结后才评估 test；但 KSVD 早期 feasibility 阶段已经查看过 MolHIV test，因此不能称为完全 untouched test。

所有字典、PCA、词表、标准化参数和 XGBoost 超参数原则上只在训练范围内拟合。ZINC 的 test 结果均应理解为冻结后的终端检查，不应再用于调参。

---

## 2. 总时间线

| 时间 | 阶段 | 主要问题 | 结果方向 |
|---|---|---|---|
| 2026-07-23～24 | 合成图 / MUTAG 管线 | 采样、向量化、KSVD 是否能跑；字典是否必须共享 | 管线通；共享字典重要；简单度特征常常更强 |
| 2026-07-24～25 | RW 与 C4/C8 | 一阶邻域看不到环时，RW 是否提供必要感受野 | C4/C8 中 RW + 共享 KSVD 接近满分；distant-triangle 反例说明 RW 不是万能的 |
| 2026-07-25 | MolHIV 图级 KSVD | KSVD 是否比 random/PCA 有用 | rich sparse-code readout `0.71213`，为最强 KSVD-specific 证据 |
| 2026-07-25～26 | MolHIV GINE 融合 | 图级向量或节点 token 能否改善 GINE | 图级 residual 很小；节点 token valid 有增益，test 不稳定 |
| 2026-07-28～29 | prototype / occurrence / MIL | KSVD atom 是否应成为局部实体，是否需要关系传播 | 真实 patch prototype 往往比 KSVD direction 更适合分类；关系机制有信号但 KSVD 特异性不足 |
| 2026-08-17～24 | 槽位桥、task-aware 字典 | 结构恢复能力能否转成 MolHIV 分类；标签是否应改造字典 | 重构改善明显，但分类没有稳定改善；监督字典不如 matched dense/random |
| 2026-08-29～09-03 | MolHIV XGBoost clean statistics | 高分 proxy 的真实来源是什么 | 主要来自拓扑/化学边际分布、跨中心异质性；不是 KSVD update |
| 2026-08-30～09-04 | ZINC | 半径、长程对象、typed motif、层级回退和 KSVD 归因 | radius-3 和 typed-WL count 很有效；hierarchical backoff 当前最强；KSVD 暂作压缩/诊断 |

---

## 3. 早期机制验证：为什么后来坚持“共享字典 + 结构化对照”

这些实验不是 MolHIV/ZINC 主结果，但决定了后续协议。

### 3.1 初始管线与失败教训

**目的**：确认 `采样 → patch 向量化 → KSVD → readout → 分类` 能否跑通，并排除“看起来能跑但字典不可比”的实现错误。

**设置**：合成 ring+chords、cycle/path，以及 MUTAG；使用诱导子图、OMP 稀疏编码和浅层 LR。最早版本曾为每张图单独学习字典。

**结果**：

| 实验 | 结果 |
|---|---:|
| 合成 cycle/path，结构 KSVD | `0.550–0.590`，而 degree statistics `1.000` |
| MUTAG，per-graph / shared 结构 KSVD | 约 `0.65` / `0.672` |
| MUTAG，degree statistics | `0.878` |
| 合成三角形 vs 长环，per-graph KSVD / shared BFS KSVD | 约 `0.54–0.64` / `0.855±0.05` |

**解释**：不是 KSVD 算法本身完全无效，而是两个问题同时存在：

- 每图一套字典时，“图 A 的 atom 1”和“图 B 的 atom 1”没有共同语义；
- 有些任务可以被度数、规模等简单统计直接解决，结构字典没有展示空间。

**决定**：之后所有正式路线使用训练集拟合的一套共享字典，并强制加入 degree/size、random、PCA 等对照。

### 3.2 C4 vs C8：RW 真正有用的场景

**目的**：构造一个严格 1-hop 看不到答案的任务，测试随机游走是否能扩大感受野。

**设置**：C4 与 C8 图的节点数、边数相同；环上节点的 1-hop 诱导子图都是路径 `P3`。比较 B0 一阶 patch、RW patch 和偏置 RW；使用共享 KSVD。

**结果**：

| 方法 | 5-fold accuracy |
|---|---:|
| degree | `0.829` |
| B0 strict 1-hop | `0.842` |
| B0 + shared KSVD | `0.833–0.854` |
| RW `L=4,m=6` + shared KSVD | `1.000` |
| RW `L=8,m=8` + shared KSVD | `0.996` |
| 偏置 RW `L=8,m=8` | `1.000` |

感受野曲线也很清楚：B0 的 C4 命中率为 `0`，`L=4,m=4` 为 `0.325`，`L=6,m=6` 为 `0.887`，`L=8,m=8` 为 `0.988`。

**解释**：当任务确实需要多跳闭环时，RW 提供了 B0 没有的信息。

**反例**：distant-triangle 任务中 B0 已能从局部三角形和度数识别标签，B0 约 `1.00`，RW 只有约 `0.80–0.95`。所以 RW 不是默认越多越好。

在图级闭环中，CoverageRW 使用 degree-stratified seed、node2vec `p=0.5,q=2`、`L=8,m=8`、最多 12 次 walk、边权软衰减 `0.7`、目标边覆盖率 `0.95`；C4/C8 上 B0 为 `82.0%`，CoverageRW + 共享 KSVD 为 `91.0%`。平均约 `5.7` 次 walk 就达到目标覆盖，而 B0 使用约 `14` 个局部 patch。这个结果说明采样效率有改善，但只在合成的长程闭环任务上成立。

**决定**：CoverageRW 保留为机制和采样候选，但真实分子任务必须先比较简单统计和局部对象，不能只凭合成环任务宣称 RW 普遍有效。

---

## 4. MolHIV：真正使用 KSVD 的路线

### 4.1 第一轮结构-only 探针：先发现图级聚合太粗

**目的**：在 MolHIV 上确认结构通道是否有非零信号，并比较 mean/max/attention。

**设置**：OGB official split 的 8,000 图开发子集；CoverageRW；结构-only LR；只作 smoke，不作为正式全量结果。

**结果**：

| 方法 | valid AUC | test AUC |
|---|---:|---:|
| degree histogram | — | `0.483` |
| mean pool | `0.549` | `0.526` |
| max pool | **`0.571`** | `0.611` |
| attention pool | `0.490` | `0.629` |

随后在 3,000 图、25 epoch 的双通道 smoke 中，GINE-only 为 `valid 0.904 / test 0.781`，图级 concat 为 `0.874 / 0.698`。正例很少、训练很短，因此只能说明**图级硬拼结构向量风险很高**，不能当作正式性能。

**设计消融**（6,000 图、seed 0）进一步显示：

| 对象 | 结果 |
|---|---:|
| size `(n, |E|)` baseline | `valid 0.765 / test 0.773` |
| max vs mean pool | `0.637` vs `0.491` valid |
| Coverage vs B0 | `0.637` vs `0.448` valid |
| 小字典 A8T2 vs A16T3 | `0.702` vs `0.637` valid |
| A8T2 + max + coverage + size | `0.803` valid，较 size `+0.037` |

但换训练子集和多 seed 后，`+0.037` 不再稳定，结构相对 size 的平均增益接近 0。由此停止“只改 pool、cover、字典大小”的图级路线。

**表示修复与早期归因**：旧的 BFS 邻接向量在 446 个 patch 中有 `36.5%` 会因 node relabel 改变；修正为 invariant WL patch 后，变化为 `0/446`。在同一 invariant patch 集上的早期 full-split sanity check 中，五个 KSVD seeds 相对 size 的 valid residual 均值为 `+0.01750`、test residual 均值为 `+0.00767`；相对 matched random-patch 的 paired 均值为 valid `+0.00528`、test `+0.00585`。这个结果促成了后面的 rich readout，但效应仍然偏小，不能与后来的完整 GINE/XGBoost 数字混为一谈。

早期 MUTAG 节点级对照也确定了融合粒度：node-level KSVD token + GIN 为 `93.6%`，全图结构向量广播为 `83.5%`，纯 GIN 为 `94.6%`；说明“哪个节点激活了什么局部模式”比把同一个图向量复制给所有节点合理，但尚未形成超过纯 GIN 的稳定增益。

### 4.2 图级 standalone KSVD：当前最干净的 KSVD 正证据

**目的**：回答“KSVD 学到的 dictionary 是否比随机抽取真实 patch 或 PCA 更有用”，不让 GINE 的容量掩盖归因。

**协议**：

- 数据：MolHIV official scaffold train/valid；test 不参与本阶段选择；
- patch：CoverageRW，`L=8,m=8`，每图最多 8 个 patch；
- patch descriptor：`wl_chem_ring`，`524D`，包含 OGB atom/bond label、labeled-WL、cycle/ring statistics；
- patch 列 L2 normalize；
- 共享字典：`8 atoms`，OMP `T=2`，KSVD `4 iterations`，最多 `4000` 个 train patch；
- readout：每个 atom 的 mean/max/top-3/std/usage/分位数/能量/signed mean/winner 等统计，加 reconstruction error、patch 数和 size，共约 `90D`；
- 下游：`StandardScaler + balanced LogisticRegression(C=1)`；
- `random-patch` 使用同一个训练 patch pool、同一 seed、同一 readout。

**结果**：

| 方法 | valid ROC-AUC | 相对 size |
|---|---:|---:|
| size-only | `0.67874` | — |
| random-patch + rich | `0.68463 ± 0.01616` | `+0.00589` |
| **KSVD + rich** | **`0.71213 ± 0.00988`** | `+0.03338` |
| PCA + rich | `0.65432` | `-0.02442` |
| random-patch + moments | `0.69043 ± 0.01244` | `+0.01169` |
| KSVD + moments | `0.70219 ± 0.00099` | `+0.02345` |

五个 dictionary seeds 的 rich paired 结果：

```text
random-patch: 0.70928 / 0.66573 / 0.68758 / 0.68436 / 0.67619
KSVD:         0.71636 / 0.72742 / 0.70292 / 0.70628 / 0.70766
paired Δ:     +0.00708/+0.06169/+0.01533/+0.02192/+0.03147
```

即 KSVD `5/5` 高于同 seed random-patch，说明这里的增益不只是“用了 invariant patch”或“用了真实 patch”，而是 KSVD 更新后的字典/稀疏编码确实提供了额外信息。

**局限**：绝对 AUC 仍明显低于 GNN；rich readout 的优势也不等于每个 atom 都能被解释成一个清晰的 HIV motif。该结果是当前最强的 **KSVD-specific** 证据，但不是“KSVD 已经超过 GINE”的证据。

### 4.3 图级 KSVD residual 接入 GINE：增益太小、方差太大

**目的**：把 graph-level KSVD 向量作为 `GINE logit + Linear(KSVD)` 的 residual，测试其是否能补充强 GNN。

**协议**：8,000 图子集；6400/800 的 official train/valid；固定 `inner_split_seed=1729`，inner train/valid 比例 85/15；最多 30 epoch；只用 inner-valid 选 epoch；同 seed 配对 DataLoader；KSVD residual head 零初始化，学习率为 base 的 `0.1`，weight decay `1e-3`。

**结果**：

| 模型 | valid AUC mean | 备注 |
|---|---:|---|
| GINE-only | `0.75847` | — |
| GINE + KSVD residual | `0.76011` | `4/5` seeds 胜，平均仅 `+0.00164` |

paired delta 为 `+0.01993/+0.01483/-0.04273/+0.01523/+0.00096`。一个 `-0.04273` 的 outlier 基本抵消了其余正增益。

**决定**：保留为 secondary signal，不再根据 official-valid 继续扫 residual 学习率或结构 seed；图级向量进入 GNN 太晚，转向节点级 token。

### 4.4 Localized node-token KSVD-GINE：valid 有增益，但 test 没有迁移

**目的**：不把所有局部 patch 先压成一个图向量，而是让每个原子保留自己的局部 KSVD token，再随 GINE 消息传递。

**冻结配置**：

- 每个原子中心的完整 radius-2 patch；
- patch descriptor `848D`；
- 共享 KSVD：`D=32`，OMP `T=3`，KSVD `3 iterations`；
- GINE：hidden `64`，`3 layers`，batch `128`；
- signed token train-only z-score；
- 每层 zero-init scalar gate；
- inner split seed `1729`，最多 `30 epochs`，按 inner-valid 选 epoch；
- full official-train 重训后再评估 official valid；test 只在冻结后受控检查。

**五 seed 结果**：

| 模型 | valid mean | test mean |
|---|---:|---:|
| GINE h64 | `0.769955` | `0.752695` |
| **KSVD-GINE h64** | **`0.794569`** | `0.751893` |
| 参数匹配 GINE h70 | `0.789699` | **`0.763280`** |

KSVD 相对 GINE h64：valid `+0.024614`、`3/5` wins；test `-0.000802`、仅 `2/5` wins。五模型 probability ensemble 为：

| 模型 | valid ensemble | test ensemble |
|---|---:|---:|
| KSVD-GINE | `0.828606` | `0.770363` |
| GINE h64 | `0.798620` | `0.775208` |
| GINE h70 | `0.810635` | `0.777620` |

**容量消融**也全部不支持继续堆大：

| 改动 | 相对冻结配置的结果 |
|---|---:|
| 加深到 h64/l4 | `-0.00150` |
| 加宽到 h80/l3 | `-0.01872` |
| 字典 D48/T4 | `-0.07070` |
| 自适应 router | `-0.01498` |

**解释与决定**：KSVD token 在 valid 上包含真实结构信号，但当前 token 注入方式的 scaffold 泛化不稳定。瓶颈不是模型不够大，因此关闭普通 width/depth/D/T/router 搜索。

### 4.4.1 字典目标和监督改造：为什么没有继续“让字典更任务化”

**问题**：如果普通 reconstruction-only KSVD 的任务增量不稳定，是否应该用阳性 patch 重加权、正负条件字典或标签增强重构目标来修正？这些实验仍以图标签为监督，但必须和 matched random dictionary 比较。

| 尝试 | 关键设置 | 结果 | 决定 |
|---|---|---|---|
| positive-patch reweight | 正负 patch 各 50%，`D=8,T=2`，5 seeds | KSVD 相对 size `+0.01961`；相对 random-patch 仅 `+0.00457`，`3/5` 胜 | 有方向但不稳，不替代普通字典 |
| `D+ / D−` 条件字典 | 正/负图分别学字典，3 seeds | activation concat 的 KSVD−random 为 `−0.00605`；重构差为 `+0.00908`，仍不稳定 | 停止 |
| label-augmented reconstruction | 在 patch target 追加图标签行，5 seeds | KSVD 相对 size `+0.01577`，random `+0.01504`，paired 仅 `+0.00073` | 不能归因于 KSVD |
| overcomplete + atom selection | 16 个候选 atom 中按正负 usage 选 8 个 | KSVD 相对 size `+0.02214`，matched random `+0.02296` | random 已能复现，停止 |

**结论**：图标签适合用于下游 occurrence/readout 的权重学习，不适合粗暴地把 molecule-level label 贴到每个局部 patch 或自由改写 atom identity。后续 task-aware 实验也遵守了这一边界。

### 4.4.2 KSVD 初始化的可微 task-aware dictionary

**目的**：如果 reconstruction-only KSVD 没有稳定分类增量，直接让图标签参与字典和稀疏码优化，能否恢复任务信号？同时要用 dense projection 对照回答：增益来自稀疏 KSVD，还是来自一般的监督表示旋转。

**协议**：8,000 图开发子集；每次只用 official-train 内层数据，最多 8 个 chemical random-walk patch；task-aware 分支从 ordinary KSVD 初始化字典，用 soft-threshold code 联合更新字典、阈值和图级 MIL head，损失为

```text
graph BCE + 0.2 × patch reconstruction + 0.05 × atom incoherence
```

在三个 scaffold fold（model seed 0）上的结果为：

| 方法 | 三折均值 AUC | 含义 |
|---|---:|---|
| raw patch max | `0.6116` | 不经过字典的局部读出 |
| frozen ordinary KSVD | `0.5913` | 只优化重构，不看标签 |
| task-aware sparse dictionary | `0.6510` | 标签参与字典/编码优化 |
| matched dense task-aware projection | **`0.6521`** | 去掉稀疏阈值的控制 |
| original-node GINE | `0.6936` | 强神经基线 |

早期固定内层 split 的三次复核中，task-aware sparse 为 `0.6734`，frozen KSVD 为 `0.6313`；但 dense task-aware 为 `0.7077`，与 sparse 的 `0.7075` 几乎相同。也就是说，**监督适配比冻结重构字典更适合标签任务**，但目前不能把增益写成“稀疏 KSVD 特有”。该路线没有进入 official valid/test。

这一区分很重要：ordinary KSVD 是无监督压缩；task-aware dictionary 已经把 HIV 标签放进了优化目标，因此应称为“KSVD 初始化的监督字典/投影学习”，不能继续当作普通 KSVD 的公平归因结果。

### 4.5 Motif-slot / occurrence / prototype：把 atom 变成图中的实体

这一组实验不是继续给 GINE 加一个 side feature，而是测试 KSVD assignment 能否生成额外的 motif 节点或 occurrence 关系。

#### A. Motif-slot incidence

**设置**：8,000 图子集；3 个 Bemis–Murcko scaffold folds；每折 train-only 拟合字典/PCA；`D=32,T=3`；在第 1 个 GINE layer 后做一次 `atom→dictionary slot→atom` transport；rank-16 bottleneck；zero-init residual gate；3 layers GINE。

| 模型 | 三折/多 seed均值 | 相对 GINE |
|---|---:|---:|
| GINE-JK | `0.71742` | — |
| KSVD motif-slot | `0.72112` | `+0.00370`，`6/9` wins |
| PCA motif-slot | `0.72538`（seed 0 screen） | KSVD 并非全面领先 |
| random-patch motif-slot | `0.72047`（seed 0 screen） | — |

KSVD 相对 shuffled-ID 和 no-ID 的平均 margin 约为 `+0.00852`、`+0.00798`，说明 persistent atom identity 有作用；但平均增益没有达到预设 `+0.005`，不进入 official valid/test。

#### B. Atom-occurrence 二部传播

**设置**：同一 8,000 图开发子集；3 scaffold folds、seed 0；完整 radius-2 invariant patch → PCA64；不做原始 atom GNN，只允许 `atom→occurrence→atom` 两轮传播；KSVD occurrence GINE 约 `64,611` 参数。

| 模型 | 三折均值 |
|---|---:|
| farthest real-patch node MIL | `0.719503` |
| farthest real-patch bipartite | **`0.736263`** |
| KSVD node MIL | `0.704892` |
| KSVD bipartite | `0.729134` |
| PCA bipartite | `0.698553` |
| random real-patch bipartite | `0.697971` |

二部传播相对对应 node MIL 有 `+0.016760`（真实 patch）和 `+0.024242`（KSVD），说明 occurrence 作为中间实体有机制价值；但 KSVD bipartite 比 farthest real-patch 低 `-0.007129`，所以**不能把这次结果称为 KSVD 优势**。

#### C. 节点特征驱动的 occurrence graph

**设置**：每个原子先经一层带键特征的 GINE，再形成 prototype occurrence；occurrence 按共享原子/真实化学键连边；比较有无 atom message passing、occurrence message passing 和 prototype ID。

| 模型 | 三折均值 |
|---|---:|
| 直接原子池化 | `0.677142` |
| atom1 + occurrence GNN | **`0.721288`** |
| atom1 + occurrence MIL | `0.672454` |
| atom1 + occurrence GNN（去掉 prototype ID） | `0.655378` |

有 ID 版本比无 ID 平均高 `+0.065910`，说明“这个局部片段是什么”很重要；但该模型内部均值仍低于更强的 prototype MIL，且训练 AUC 约 `0.9164`、held-out 仅 `0.7213`，过拟合明显，不进入 official split。

#### D. GNN-free observed real-prototype MIL

这是一个重要但必须正确命名的对照：prototype 是训练集真实 patch，不是 KSVD atom。

**设置**：完整 MolHIV official split；radius-2 invariant raw patch → train-only PCA64；official-train 选 32 个 observed real prototypes；无 supervised message passing；3 个 prototype-bank seeds；fixed 30 epochs。

| family | valid AUC | test AUC |
|---|---:|---:|
| real-prototype MIL ensemble | **`0.802488`** | `0.767771` |
| matched 3-seed GINE ensemble | `0.781385` | **`0.777993`** |

它证明真实 patch vocabulary 有竞争力，但不是 KSVD-specific 证据。KSVD direction 在多个 prototype/occurrence 实验中没有稳定超过 observed real patch。

先在 8,000 图开发集上做的 raw-patch metric 对照也给出了同样的方向：完整 radius-2 invariant raw patch 先在每个 scaffold-fit fold 做 PCA64，再用完全无监督的 node-level MIL；比较 真实 observed prototypes 与 KSVD directions。3 folds × 3 seeds 的 grand mean 为：真实 prototype `0.735432`，KSVD direction `0.724532`；matched fixed-epoch original-node GINE 的 grand mean 为 `0.717420`。因此，KSVD 可以形成可用坐标，但“真实 patch identity”在这条 MIL 管线中更适合分类；不能把 node-level MIL 的效果表述为 KSVD 特异增益。

#### D.1 真实 prototype 的距离关系与 pair-PCA：有信息，但没有稳定增量

这条旁线用于区分“原型出现了多少”和“哪些原型在分子中相邻或隔一个原子共现”。它是 real-prototype 对照，不是 KSVD 结果。

**距离尺度与关系读出。** 在 8,000 图、6,400 个 official-train 图、3 个 scaffold folds 上，冻结 occurrence base，构造距离 1、2、3+ 的 prototype relation，并用低容量 residual。两组 prototype bank 的 pair ensemble 结果如下：

| view | AUC | 相对 occurrence base | real − assignment-shuffled |
|---|---:|---:|---:|
| occurrence base | `0.755736` | — | — |
| distance 1+2 compact | **`0.758018`** | `+0.002282` | `+0.001957` |
| full connected compact | `0.758382` | `+0.002646` | `+0.001338` |

距离 1+2 是最稳定的尺度；distance 3+ 没有额外收益。更大的 full relation（每个 family 2,650 个参数）与 210 参数的 compact connected relation 几乎相同：`0.758440` vs `0.758382`。random-walk relation 的最佳 pair 为 `0.757736`，比 exact distance 1+2 低 `0.000282`，因此没有证据支持用 RW 替换简单距离统计。所有这些结果均未进入 official valid/test。

**无标签 covariance pair-PCA。** 对距离 1/2 内具体 prototype pair 的 1,984 维计数，在 outer-fit 分子上拟合 covariance PCA（rank 144），再作为 compact base 的有界 residual。三折 development 中：compact base `0.779491`，pair-PCA `0.782674`，平均增益 `+0.003183`，且 real 相对 assignment-shuffled 为 `+0.002248`；因此通过了预先的内部晋级门槛。

但冻结后 official-valid 结果反转：

| view | valid AUC |
|---|---:|
| occurrence probability ensemble | `0.808167` |
| compact real distance 1+2 | **`0.812457`** |
| covariance pair-PCA | `0.811508` |
| compact assignment-shuffled | `0.814726` |

pair-PCA 相对 compact 反而为 `-0.000949`，因此未通过 official-valid gate，不运行 test。另一次冻结的 compact terminal test 为 real `0.751461`、assignment-shuffled `0.752067`；这说明 compact 可能提供小的总体关系增量，但不能证明具体 prototype identity 是增益来源。

**task-aware relation sidecar。** nested task-aware prototype bank 在开发集 residual cap=`0.3125` 时使 broad occurrence base 从 `0.755736` 提升到 `0.761675`（`3/3` folds，real−shuffled `+0.005947`），但更换 candidate-bank seed 后增益只剩 `+0.001884`，而 shuffled-label selector 有时更强。robustness gate 失败，因此没有进入 official split。

#### E. Task-aware dictionary / prototype selector

**目的**：让图标签参与字典或 prototype 选择，看看是否能把无监督字典改成任务相关字典。

**设置**：8,000 图、3 scaffold folds；图标签只通过 graph-level loss 进入，避免把分子标签粗暴复制给每个 patch；比较 frozen KSVD、task-aware sparse、matched dense projection、random。

| 模型 | 三折/九单元均值 |
|---|---:|
| frozen ordinary KSVD | `0.7088` |
| adapted KSVD sparse | `0.7095` |
| frozen random patch | **`0.7222`** |
| matched dense task-aware projection | `0.6521`（3-fold seed-0 screen） |

task-aware sparse 相对 ordinary KSVD 只 `+0.00071`，且只 `5/9` wins；random 反而高 `+0.01339`。结论是：**任务信息更适合进入 occurrence weighting/readout，而不是自由塑造 dictionary identity。**

另一版 prototype-constrained selector 做了更直接的 matched test：从 256 个真实 fold-fit local context latents 中选 32 个 prototype，使用图级 label ranking + diversity-constrained MMR，再用 top-3 occurrence MIL 预测；它没有 supervised message passing，但局部 context 本身来自冻结的无标签两层 masked-context GINE。三折 seed-0 结果为：random real `0.7310`、task-aware real `0.7056`、label-shuffled selector `0.7373`、KSVD directions `0.6896`。真实标签 selector 不仅没有超过 random，还低于 shuffled selector，说明高维单切分 selector 很容易拟合 scaffold-specific noise；该分支停止。

#### F. 槽位—共享原子桥

**目的**：结构重构中，遮住一个局部块后，正确的共享原子关系能否恢复其内部边；并观察是否转化为 MolHIV 分类。

**设置**：MolHIV official-train 内部 scaffold folds；不使用 KSVD、GINE 或 HIV 标签做重构任务；后续才做分类小试和全量内部折。

| 任务/分支 | 结果 |
|---|---:|
| 全量内部折结构重构，无桥 / 正确桥 / 打乱桥 RMSE | `0.4937 / 0.3637 / 0.4932` |
| 对应 F1 | `0.4229 / 0.6926 / 0.4243` |
| 全量内部折分类，无桥 / 正确桥 / 打乱桥 AUC | **`0.6430`** / `0.6382` / `0.5969` |

正确桥把重构误差降低约 `26%`，但分类不超过无桥。说明结构恢复目标与 HIV 标签目标错位；不再把 KSVD 加进这个关系层制造新变量。

### 4.6 MolHIV KSVD 路线小结

已经成立：

- 共享字典比每图独立字典可比较；
- invariant patch、CoverageRW 和 rich sparse-code distribution 都能携带信息；
- KSVD rich 在 matched random/PCA 上有清楚的 standalone 优势；
- 节点级 token 比图级广播合理，motif/occurrence identity 不是无意义编号。

没有成立：

- KSVD-GINE 在 scaffold test 上稳定超过 GINE；
- 加大字典、增加层数、自由 router 能解决泛化问题；
- task-aware dictionary 比 random/real prototype 稳定更好；
- 结构重构改善自动变成分类改善。

补充一个容易被忽略的终端对照：在 8,000 图 stratified 子集上，曾将 raw patch/PCA64、32 个 KSVD direction 和原始节点 GINE 都固定 30 epochs，并保留 OGB train/valid/test 成员关系。该小规模 terminal 中，GNN-free random-real-prototype MIL、KSVD-direction MIL、3-layer GINE 的 test ensemble 分别为 `0.7006`、`0.6777`、`0.7238`。这再次说明：KSVD direction 的 test 结果不能替代真实 prototype，也不能直接击败 GINE；该结果只是受控子集终端检查，不是完整 MolHIV 榜单。

---

## 5. MolHIV：luyin16 的 XGBoost / 统计路线

这一阶段的目的，是复现导师所描述的“固定结构/化学特征 + XGBoost”思路，并把高分到底来自哪里拆出来。由于导师原始上游代码、69D/624D schema 和 payload 不完整，以下需要区分 **proxy** 与 **clean invariant route**。

### 5.1 导师固定特征的 proxy 复现

**目标**：在没有导师 exact feature builder 的情况下，区分显式统计、raw patch、K-SVD update 和融合的贡献。

**统一设置**：MolHIV official train/valid；XGBoost `binary:logistic`；三折 official-train scaffold 内部搜索；所有特征/字典 train-only；本阶段 test 未编码。

#### 5.1.1 显式 `S` 基线

`S=205D`，由显式拓扑、atom composition 和 bond composition 组成。

| 特征块 | valid ROC-AUC |
|---|---:|
| topology 18D | `0.6831` |
| atom composition 174D | `0.7066` |
| bond composition 13D | `0.5993` |
| atom + bond composition 187D | `0.7196` |
| **topology + chemistry：S 205D** | **`0.7817 ± 0.0066`** |

这里的高分来自拓扑与化学组成在树模型中的互补交互，不能叫“纯 KSVD”或“纯结构”。

#### 5.1.2 `T_init / T_final` 与 624D reconstruction proxy

| view | valid AUC |
|---|---:|
| `T_init` | `0.6718` |
| `T_final` | `0.6521` |
| `S+T_init` | `0.7717` |
| `S+T_final` | `0.7624` |
| `R_raw` 624D proxy | `0.6649` |
| `R_init` | `0.6521` |
| `R_final` | `0.6542` |

K-SVD 将 mean graph reconstruction error 从 `0.1732` 降到 `0.1204`，但分类反而下降。`S+R_raw`、`S+R_init`、`S+R_final` 也没有超过 `S`。

**结论**：普通无监督 K-SVD update 没有稳定的 MolHIV label increment；重构目标和 scaffold-held-out 分类目标错位。

#### 5.1.3 typed-slot proxy v2

为了使 624D 更像“有位置/槽位的局部对象”，又测试了：

```text
8 个 atom slots × 64 bins + 28 个 slot pairs × 4 bond bins = 624D
```

结果：`r_raw=0.7014`，`r_init=0.6535`，`r_final=0.6752`；按 occurrence sum 聚合更差，`r_raw=0.5821`。slot 对齐比全局统计 proxy 好约 `+0.0365`，但仍远低于 `S=0.7817`，且 `S+r_final` 没有增益。

#### 5.1.4 纯结构 69D/624D proxy：去掉化学属性后并没有得到增量

**目的**：判断导师路线中的高分是否可能主要来自“纯结构 `S_struct` + 结构 K-SVD reconstruction”，而不是化学组成。由于导师真实的 69D/624D schema 不在仓库中，这里只做维度对齐 proxy。

**设置**：`S_struct=69D`（topology、degree、degree-pair、shortest-path/disconnected 统计）；拓扑 patch 为 28D induced adjacency；`R_struct=624D`，由 `K=24` 个 atom 的 26 组系数统计组成；K-SVD 使用 OMP `T=3`、5 次 update；8-trial Optuna、3-fold train-inner CV、5 个模型 seeds。

| view | valid AUC |
|---|---:|
| `S_struct` | **`0.7723±0.0071`** |
| `R_init_struct` | `0.6460±0.0035` |
| `R_final_struct` | `0.6385±0.0053` |
| `S_struct+R_init` | `0.7333±0.0089` |
| `S_struct+R_final` | `0.7367±0.0045` |

重构误差从 `0.2029` 降到 `0.1179`，但分类没有同步改善；`S_struct` 自身已经强于两种 reconstruction view。结论只能是“这个纯结构 proxy 不支持 K-SVD 增量”，不能据此否定导师的真实 69/624 schema。

#### 5.1.5 T readout 消融：624D 的表面维度不等于有效信息维度

**目的**：直接检查 `T` 应该读 raw patch、重构 patch、residual，还是 sparse-code summary。当前 624D 仍是未知导师 schema 的 typed-slot proxy；实验只用 official train/valid，没有 test。

在 `K=64`、每图最多 8 个 patch 的 proxy 中，固定参数与 8-trial 搜索结果为：

| T view | 维度 | fixed valid | tuned valid |
|---|---:|---:|---:|
| raw `mean(Y)` | 624 | `0.7014` | `0.6742` |
| reconstruction `mean(DX)` | 624 | `0.6687` | `0.6567` |
| residual `mean(abs(Y-DX))` | 624 | `0.7116` | `0.7068` |
| rich sparse-code summary | 640 | `0.6920` | `0.6741` |
| absolute code mean | 64 | `0.6732` | `0.6758` |

随后对唯一保留下来的 `S+T_raw` 做匹配复核：inner-CV 为 `0.7671`，但 official-valid 只有 `0.7689`，而 `S` 为 `0.7848`，5 个 seeds 中 `0/5` 胜出。`mean(DX)=D\,mean(X)` 最多落在 K-SVD atom 张成的低维子空间中，因此不能因为它“也是 624D”就把它当成导师的 `recon_typed[624]`。

### 5.2 旧的 `0.8307` 历史高分为什么不能使用

曾有 radius-2、all-center 的 `S+R_raw≈0.8307` valid 结果，另有冻结后 `S+R_final` test `0.8022`。后续对象审计发现该实现：

- 用 node-ID tie-break，重标号后特征会变；
- `14.12%` patch 被截断；
- atom category modulo 造成 `16` 个真实类别碰撞；
- 冻结模型重标号后单图预测最大漂移 `0.3454`。

所以该 patch 不是 permutation-invariant 的有效图对象。该数字只能作为**历史调试结果**，不能作为 KSVD/导师路线的正式结果，也不能据此把 `R_final` 选成主模型。

### 5.3 修正后的 invariant local object 与分布 readout

审计修正为：完整 radius-2 induced ego、all-center、topology-only rooted-WL、strict chemistry（去掉 degree/ring 重复字段），并通过随机重标号检查。

#### 5.3.1 clean patch 机制 screen

三折 official-train 内部结果：

| view | mean fold AUC |
|---|---:|
| `S` | `0.693865` |
| invariant topology raw | `0.664597` |
| invariant attributes raw | `0.735942` |
| invariant joint raw | `0.712774` |
| `S+joint raw` | `0.720513` |
| `S+joint INIT` | `0.748289` |
| `S+joint FINAL` | `0.740832` |

INIT/FINAL 的重构误差确实下降了约 `20.5%–26.3%`，但 `FINAL−INIT=-0.007457`。稳定信号是 all-center 局部化学分布，不是 K-SVD update。

#### 5.3.2 读出诊断：真正被漏掉的是“中心分布”

早期把所有中心 patch 简单取 mean；后来恢复 mean/std/分位数/极值等 patch population distribution：

| view | 维度 | valid AUC |
|---|---:|---:|
| `T+A mean` | 154 | `0.794468 ± 0.003303` |
| `T+A distribution` | 1793 | **`0.817198 ± 0.002827`** |
| `S+T+A mean` | 359 | `0.802208 ± 0.002013` |
| **`S+T+A distribution`** | 1998 | **`0.827574 ± 0.004311`** |

快速归因显示主要增量来自跨中心 `std`，不是“维度越高越好”。这解释了为什么 raw typed distribution 往往比 `mean(DX)` 更有任务信息：前者保留一个分子内部局部环境的异质性，后者把它压平或重构掉。

随后对三个候选做 train-only scaffold tuning，结果为：

| view | train scaffold CV | official-valid | train+valid refit test |
|---|---:|---:|---:|
| `S+T+A mean` | `0.7776` | `0.8128` | `0.7697` |
| `S+T+A mean+std` | `0.7856` | `0.8266` | `0.7723` |
| `S+T+A all12` | **`0.7895`** | **`0.8341`** | `0.7779` |
| fixed 50/50 mean+std/all12 | — | — | **`0.7814`** |

五模型 ensemble test 为 `0.7849`。这仍是 XGBoost clean statistics 结果，不是 KSVD 结果。

### 5.4 MolHIV 结构—属性融合的逐层尝试

#### 5.4.1 coarse role → rooted-WL role

**问题**：之前的 `shell×degree×cycle` role 是否太粗，导致融合失败？

**设置**：完整 radius-2 all-center patch；topology-only rooted-WL 两轮；strict atom/bond semantics；比较 topology `T`、attribute `A`、`T+A`、factorized raw、centered binding 和 matched shuffle；不使用 KSVD。

内部三折：

| view | mean AUC |
|---|---:|
| coarse `T` | `0.6608` |
| rooted-WL `T` | `0.6941` |
| rooted-WL `T+A` | `0.6959` |
| rooted-WL factorized raw | **`0.7327`** |
| centered binding | `0.7257` |

rooted-WL 修复了“结构坐标过粗”的问题，但 official-valid 三 seed 中 factorized raw 为 `0.7731`、centered 为 `0.7872`，没有稳定超过简单 `T+A=0.7865`。所以结构—属性依赖存在，但不等于稳定标签增量。

#### 5.4.2 exact topology / orbit / conditional pairing

**问题**：属性落在精确 structural orbit 上是否重要？或者只要同一个 patch 的 topology 与属性保持配对即可？

**设置**：三折 `1200/600`，XGBoost 固定，model seeds `0/1/2`；top-32 exact topology；每种 topology 最多 4 个 empirical templates；两个 matched shuffle：

- orbit shuffle：保持同 patch 的属性 multiset，但破坏属性落在何种 orbit；
- patch-pair shuffle：保持 topology patch bag 和 attribute patch bag，但错配它们属于哪个 patch。

结果：

| view | mean AUC |
|---|---:|
| `S+WL+attribute`（最强 unbound） | `0.7145` |
| exact conditional | `0.6891` |
| exact conditional + WL hybrid | `0.7021` |
| exact conditional true − patch-pair shuffle | `+2.36pt`，`8/9` wins |
| orbit true − orbit shuffle | 约 `-0.90pt` |

结论：patch-level topology—attribute pairing 的机制可以检测，但 exact orbit 位置绑定没有通过，joint XGBoost 也没有超过强 marginal baseline；不继续扫 exact vocabulary、prototype 数或 K-SVD。

#### 5.4.3 cross-centre interaction：把“分子内部异质性”显式化

**问题**：有用的交互是否不是单个 patch 内的 role-binding，而是不同中心环境如何共同变化？

**设置**：invariant radius-2 rooted-WL patch；`cross_cov` 为不同中心的 topology row 与 attribute row 的中心化协方差；`binding` 为 patch 内 role-attribute centered binding 的中心级统计；高维块每折 train-only PCA；XGBoost；先做 3 个完全不重叠的 train-only scaffold scale-up，再做 official-valid/test 冻结检查。

scale-up 的三折结果：

| view | mean AUC |
|---|---:|
| `S+marginal` | `0.709778` |
| `S+cross_cov` | `0.713307` |
| `S+binding` | `0.714842` |
| **`S+both`** | **`0.720022`** |

`S+both` 相对 marginal `+0.010243`，相对 matched double-shuffle `+0.010226`，均为 `3/3` folds；但在更完整的 official-valid 独立调参后，简单 marginal 反而是最好的 valid view。

### 5.5 MolHIV XGBoost 的 official-valid / controlled-test 结果

**冻结协议**：

- official-train `32901` 图、`1232` positive；
- 每个 view 独立做 `16-trial Optuna TPE`；目标只用 3 个 official-train scaffold folds；
- 交互块 PCA-8 只在训练范围拟合；
- 五个 model seeds `0–4`；
- XGBoost `binary:logistic`，tree method `hist`；
- `S+marginal` 选出的参数为：`n_estimators=582`、`max_depth=7`、`learning_rate=0.02846`、`min_child_weight=2.055`、`subsample=0.7195`、`colsample_bytree=0.4433`、`reg_lambda=32.88`、`reg_alpha=13.16`、`gamma=1.596`、`max_bin=256`。

#### 5.5.1 official-valid

| view | train scaffold CV | tuned valid mean | valid ensemble |
|---|---:|---:|---:|
| `S` | `0.761323` | `0.791586` | `0.793498` |
| **`S+marginal`** | `0.787106` | **`0.841285`** | **`0.845047`** |
| `S+cross_cov`（PCA-8） | `0.789985` | `0.839257` | `0.842530` |
| `S+binding` | `0.788789` | `0.838859` | `0.841449` |
| `S+cross_cov+binding` | **`0.790662`** | `0.838222` | `0.841259` |

这说明 0.84 valid 的主要来源是**稳定的边际分布统计**；交互块在 train-only folds 有机制价值，但没有稳定超过 marginal。

#### 5.5.2 controlled official-test

下表分为严格 train-only fit 和通常的 train+valid refit。test 未用于选择，但由于更早实验已经查看过 MolHIV test，这里称 controlled terminal test。

| view | strict test mean | strict ensemble | train+valid refit test mean | refit ensemble |
|---|---:|---:|---:|---:|
| `S` | `0.743173` | `0.746054` | `0.753994` | `0.756810` |
| `S+marginal` | `0.788872` | `0.791753` | `0.785279` | `0.788797` |
| `S+cross_cov` PCA-8 | `0.800937` | `0.805587` | `0.799824` | `0.803652` |
| `S+binding` | `0.789697` | `0.792804` | `0.788038` | `0.790477` |
| **`S+both` PCA-8** | **`0.808749`** | **`0.813525`** | `0.795642` | `0.800228` |

这是冻结后的泛化证据，不是重新选择的排行榜。valid 排名和 test 排名反转，说明 scaffold shift、PCA scope 和仅 `130` 个 test positive 带来了较大不确定性。当前谨慎表述是：

- valid 性能主线：`S+marginal`；
- 机制核心：`cross_cov`；
- `binding`：条件辅助块；
- `S+both` 的 strict test 表现较强，但还不能升级为不依赖 PCA scope 的最终稳定方案。

#### 5.5.3 PCA-16 敏感性

同一 `cross_cov` 改为 PCA-16、仍只在 train scaffold folds 调参：valid `0.837160`、strict test `0.803619`、train+valid refit test `0.797590`。它说明 cross-centre covariance 不是偶然噪声，但 rank 改变会改变 valid/test 排名，不能凭这一轮替换 `S+marginal`。

### 5.6 MolHIV 下游模型与融合控制

前面的 XGBoost 结果还需要一个重要控制：高分是否只是 XGBoost 的特殊性，或者只要换一个可学习 readout 就能同样得到？因此又做了 MLP、patch-level attention 和独立专家 late fusion。它们都**不使用 KSVD**，用途是解释归因，不是和 KSVD 主线争夺同一榜单。

#### 5.6.1 MLP readout 与损失函数控制

**协议**：完整 MolHIV official split（`32901/4113/4113`）；输入为 `508D` 的 `S_v1 + radius-2 center marginal mean/std + 5 个 graph context`；CPU、seed 0、最多 50 epochs；按 official-valid 选 epoch，再用 train+valid 重训后检查 test。比较普通 BCE、balanced BCE 和 `pos_weight∈{2,5,10}`。

| loss / 设置 | `S+marginal` valid / test | center-level fusion valid / test |
|---|---:|---:|
| unweighted BCE | `0.809022 / 0.731598` | `0.830176 / 0.747878` |
| balanced BCE | `0.816225 / 0.745364` | `0.805118 / 0.744364` |
| positive-weight search（最优 weight=10） | `0.830666 / 0.732772` | `0.827075 / 0.741370` |

center fusion 的形式是每个中心拼接 `[s_v,a_v,s_v*a_v]` 后再做 sum/mean/std pooling。结果说明 readout 和 class weighting 会显著改变 valid，但没有稳定解决 valid→test 的落差；因此 `0.84` 左右的 XGBoost valid 不能简单归因于某个万能的可学习融合器。

#### 5.6.2 可学习 patch 融合：cross-attention 没有超过属性边际

**协议**：只用 official-train 内部三个 Bemis–Murcko scaffold folds，每折 `1200/600` 图；完整 radius-2 patch；hidden=48、2 层 topology GIN、attribute set encoder、15 epochs；无 K-SVD、无 cycle 手工特征、无 node ID。比较结构-only、属性-only、简单 concat、structure-query cross-attention，以及随机和 size-matched attribute shuffle。

| view | mean AUC |
|---|---:|
| structure-only | `0.6171` |
| attribute-only | **`0.6683`** |
| concat | `0.6330` |
| cross-attention | `0.6297` |
| size-matched shuffle | `0.6255` |

cross-attention 比 concat 低 `0.00335`，比 size-matched shuffle 只高 `0.00414`，而属性边际明显最强。另一个完整 official-split 的 centre conditional fusion（rooted-WL radius-3、100 epochs）为 valid/test `0.814636/0.701877`，也没有显示稳定迁移。结论是：同一 patch 中确实可以构造结构—属性交互，但在当前 MolHIV scaffold 泛化上继续堆 attention、FiLM、MoE 或 interaction layer 的收益很低。

#### 5.6.3 独立 XGBoost 专家与 late fusion

feature-level 把所有交互拼在一起会让边际专家和交互专家互相干扰，于是单独训练两个 clean rooted-WL XGBoost 专家：`T+A` marginals 与 centered binding；各自用 12-trial Optuna 和 5 个 model seeds，预先固定 probability `50/50` late fusion。所有表示都不含 KSVD。

| candidate | train+valid refit test mean | test probability ensemble |
|---|---:|---:|
| `T+A` | `0.7610` | `0.7628` |
| centered | `0.7716` | `0.7732` |
| fixed 50/50 late fusion | **`0.7825`** | **`0.7839`** |

冻结前 official-valid 的五 seed 均值为 `T+A=0.7945`、`centered=0.7962`；late fusion 的 test 结果是一次受控 terminal evaluation。由于 MolHIV test 在更早路线已经查看过，不能称为 untouched test，也不能据此再改权重。这个控制说明独立专家的误差互补可能有价值，但它仍然不是 KSVD 贡献。

### 5.7 MolHIV 路线最终判断

| 结论 | 状态 |
|---|---|
| clean 局部 patch 与其跨中心分布有任务信息 | 已支持 |
| XGBoost 可以成为强的固定特征下游 | 已支持 |
| ordinary unsupervised K-SVD update 提供稳定增量 | 未支持 |
| KSVD-GINE valid 增益能稳定迁移到 test | 未支持 |
| 高分 `S+marginal` 可以归因给 KSVD | 不可以 |
| 继续扫 K/T、Beam、attention、router 能解决当前瓶颈 | 不支持，已停止 |

---

## 6. ZINC：真正使用 KSVD 的路线

### 6.1 长程与半径 factorial：感受野有效，K-SVD update 未被证明有效

**目的**：验证导师提出的“当前局部对象看不到长程信息”是否成立，并分离 radius、patch cap、raw object 和 KSVD update 的作用。

**协议**：

- PyG `ZINC(subset=True)`，`10000/1000/1000`；
- 每个原子都是中心，保留全部中心 patch；
- radius `1/2/3`；默认 `max_nodes=12`，另测 radius-3 `max_nodes=20`；
- train-only KSVD：`K=24`、OMP `T=3`、`6 iterations`；
- XGBoost 回归，objective `reg:absoluteerror`，指标 MAE；
- 只有冻结后才对 validation 晋级方案做 train+valid refit test。

**固定参数结果**：

| view | valid MAE | test MAE |
|---|---:|---:|
| global statistics + attributes | `0.6122` | `0.6382` |
| r1 global + raw + KSVD-final | `0.5436` | `0.5674` |
| r2 global + raw + KSVD-final | `0.5430` | `0.5599` |
| r3 global + raw + KSVD-final | **`0.5225`** | `0.5400` |
| r3-wide (`max_nodes=20`) | `0.5256` | **`0.5276`** |

后续对候选做有限 Optuna：

| view | valid MAE | train+valid refit test MAE |
|---|---:|---:|
| r1 | `0.5151` | `0.5475` |
| r2 | `0.5139` | `0.5478` |
| **r3** | **`0.4953`** | `0.5271` |
| r3-wide | `0.4967` | **`0.5080`** |

采样诊断：平均每图约 `23.17` 个中心，节点覆盖率 `1.0`；节点对覆盖率约为 r1 `0.2391`、r2 `0.4952`、r3 cap12 `0.6787`、r3 cap20 `0.6987`。cap 从 12 提到 20 几乎不改善 valid，因此主要收益来自 radius 带来的上下文/关系覆盖，而不是简单消除截断。

**KSVD 归因**：

| radius | typed INIT reconstruction error | typed FINAL |
|---|---:|---:|
| r1 | `0.1414` | `0.0592` |
| r2 | `0.2728` | `0.1760` |
| r3 | `0.4267` | `0.3220` |
| r3-wide | `0.4329` | `0.3306` |

重构误差都下降，但 r3 standalone typed code 的下游从 INIT `1.0330` 变为 FINAL `1.0613`，融合时 valid 与 test 的 INIT/FINAL 排名也不一致。

**决定**：ZINC 上保留 radius-3 raw typed object 作为主对象；KSVD 只作压缩/诊断，不再把 reconstruction error 下降当成任务收益。

### 6.2 结构—属性 binding、SVD residual、长程 relation

这些实验先在两个互不重叠切片上做 gate，避免直接消耗 full test。

#### 6.2.1 joint v1

固定 radius-2、全中心、`max_nodes=12`、XGBoost；切片 A/B 各为 `5000/500`；图内同步打乱 attribute columns 作为 matched shuffle。

| view | slice A MAE | slice B MAE |
|---|---:|---:|
| global | `0.69399` | `0.58892` |
| global + typed raw | `0.67419` | `0.55329` |
| global + joint v1 | `0.69203` | `0.55738` |
| global + relation | `0.66805` | `0.55286` |
| global + raw + relation | `0.64894` | `0.54751` |

joint true 相对 shuffle 的 gap 为 `+0.02195/+0.01914`，所以 structure–attribute binding 可被检测；但 joint v1 没有超过 typed raw。relation 对 raw 的降 MAE 只有 `+0.02525/+0.00578`，没有跨切片达到预设 `0.01`。

#### 6.2.2 conditional joint v2 与 centered residual SVD16

v2 用 permutation-invariant signature：`(n_nodes,n_edges,cycle_rank,center_degree/shell1,shell2_size)`，最多 64 类；统计 signature-conditioned atom/bond distribution。

| 大切片 | global+raw | conditional | conditional 相对 raw |
|---|---:|---:|---:|
| A | `0.67419` | `0.69608` | `-0.02189` |
| B | `0.55329` | `0.54532` | `+0.00797` |

两个切片的 true-shuffle gap 都为正（约 `0.01559/0.03609`），所以机制存在；但预测增量不稳定，不能进入 full/Optuna/K-SVD。

随后只提取

```text
R = P(signature, attribute) - P(signature)P(attribute)
```

并用 train-only SVD 压到 16D：

| slice | global+raw | + true residual SVD16 | 相对 raw | true-shuffle gap |
|---|---:|---:|---:|---:|
| A | `0.73813` | `0.71246` | `+0.02567` | `+0.04408` |
| B | `0.56485` | `0.56640` | `-0.00155` | `+0.02089` |

SVD 保留约 `71%` 方差，但 B 仍失败。说明“检测得到 binding”不等于“binding 提供稳定任务增量”。

#### 6.2.3 long-range local-object relation

**设置**：以 radius-2 typed raw 为 baseline，统计距离 `≥3` 的 atom-object pair；distance 截断为 `7+`；position-shuffle 保留 object bag 但打乱中心位置；正式切片各 `2000/200`。

| slice | global+raw | + long atom relation | 相对 raw | true-shuffle gap |
|---|---:|---:|---:|---:|
| A | `0.73813` | `0.73223` | `+0.00591` | `+0.01436` |
| B | `0.56485` | `0.59294` | `-0.02809` | `+0.00264` |

pair coverage 为 `100%`、valid unknown rate 约 `0.00017/0.00010`，所以失败不是词表覆盖不足。当前结果否定的是“radius-3 gain 只是远距离 pair bag”，不否定 radius-3 对单个局部对象上下文的价值。

#### 6.2.4 r2+r3 multiscale

固定 XGBoost、无 K-SVD、两个 `2000/200` 和两个 `5000/500` 切片：r2+r3 在四个切片方向都优于单尺度，但相对最佳单尺度改善为 `0.01066/0.00549/0.01501/0.00471` MAE，没有稳定达到预设 `0.01`。因此保留 radius-3 单尺度，multiscale 不晋级。

### 6.3 ZINC：XGBoost typed motif count（关闭 KSVD）

这是 ZINC 上最重要的非 KSVD 对照，因为它直接测试“结构 token 计数”是否比粗统计更有效。

**设置**：

- radius-3、每个原子为中心；
- typed rooted-WL rounds `0/1/2/3`；
- 每个 train fold 只保留 top-K=`2048` exact nested tuple token，加 OOV；
- readout 是 raw count 与 count/center 数；
- XGBoost `reg:absoluteerror`，20 trials，3 个 train-only folds；
- KSVD disabled；另做 2048-bit Morgan/ECFP count 对照。

**结果**：

| view | 维度 | valid MAE | train+valid refit test MAE |
|---|---:|---:|---:|
| `S` global | 62 | `0.560657` | `0.592425` |
| `wl_count` | 16392 | `0.410354` | `0.429296` |
| **`S+wl_count`** | 16454 | **`0.370918`** | `0.376450` |
| Morgan count | 4096 | `0.534471` | `0.525467` |
| `S+morgan_count` | 4158 | `0.419934` | `0.417120` |

因此 typed-WL count 明显优于 Morgan count。但这是**typed-WL/XGBoost 结果，不是 KSVD 结果**。

### 6.4 ZINC：radius-2/3 marginal XGBoost transfer

另一套较低维的 `S+marginal` transfer 使用：`S=62D` global structure+atom/bond composition，local mean/std+context；XGBoost 每 view `20-trial Optuna`、3-fold shuffled train-only tuning，仍关闭 KSVD。

| view | 维度 | valid MAE | test MAE |
|---|---:|---:|---:|
| S | 62 | `0.560937` | `0.595750` |
| S + radius-3 marginal | 323 | `0.544409` | `0.551926` |

相对 S，valid 降 `0.016527` MAE，test 降 `0.043824`。由于它与高维 exact token count 使用了不同的特征对象和调参协议，不与上一节绝对排名混在一起。

### 6.5 ZINC：hierarchical exact-token backoff——当前 XGBoost 主结果

**目的**：解决 radius-3 typed-WL exact token 长尾严重、低阶结构信息被 OOV 丢掉的问题。

**设置**：

- rounds `0/1/2/3` 的 exact typed-WL vocab；
- 每轮 top-K=`2048`；
- 若高阶 exact token 不在词表，则回退到同中心的低阶 token；
- valid vocabulary 只用 official-train；test vocabulary 用 train+valid；
- XGBoost `reg:absoluteerror`；20 trials；3 个 train-only folds；
- **K-SVD disabled**。

**结果**：

| view | 维度 | valid MAE | test MAE |
|---|---:|---:|---:|
| `s_wl_count` | 16454 | `0.372258` | `0.375721` |
| **hierarchical backoff** | 41048 | **`0.354605`** | **`0.345460`** |

相对 `s_wl_count`，valid 降 `0.017652`，test 降 `0.030261`。这是当前截至日期的 ZINC XGBoost 主结果，但不能归因给 KSVD；它来自层级 typed-WL 词表和回退读出。

### 6.6 ZINC：层级特征扩展与 prototype matching

#### 6.6.1 hierarchical composition

在 hierarchical backoff 上加入 cross-level、cross-centre、rarity 和 molecule-composition，维度升到 `61592D`。

| view | valid MAE | test MAE |
|---|---:|---:|
| hierarchical backoff | `0.354605` | `0.345460` |
| hierarchical composition | `0.378619` | `0.402539` |

新增组合统计使结果变差，停止继续堆叠。

#### 6.6.2 typed match

**设置**：K-SVD disabled；多层 typed-WL prototype bank，`1024` 个 prototypes；round 权重 `[0.1,0.2,0.3,0.4]`；每个 prototype 做 weighted prefix-match mean/max、response summaries、best-centre score 和 residual summaries；总维度 `2143D`；XGBoost 20 trials、3 train-only folds。

结果：valid `0.410045`，test `0.395111`，明显不如 hierarchical backoff。它是 candidate implementation，不是导师 exact typed-match replication。

#### 6.6.3 topology-conditioned match gate

**目的**：测试 topology-conditioned attribute prototype 的“同中心匹配”是否有额外价值。

**设置**：每折 train-only 建 hierarchical vocab 和 conditional prototype bank；另做图内属性行置乱；固定 XGBoost，不做 Optuna。

| 三折均值 | MAE |
|---|---:|
| hierarchical baseline | `0.373110` |
| true match | `0.401711` |
| shuffled match | `0.402432` |

true 相对 shuffle 只好 `0.000721`，且 `0/3` folds 胜 baseline，gate 失败，不进入 official valid/test。

### 6.7 直接 exact pair relation：显式关系仍明显弱于层级回退

**目的**：把“层级 typed-WL token 计数”的收益与“显式枚举中心对关系”的收益分开。该实验直接使用 exact rooted typed radius-2 patch 和所有无序中心对，不做消息传递，也不使用 KSVD。

**设置**：patch descriptor `840D`；以 `pynauty 2.8.8.1` 的 colored-incidence canonical certificate 消除节点顺序影响；关系包括中心距离、四种 overlap ratio、root atom 是否相同、exact patch 是否相同、相邻 bond type、cosine 和 shared descriptor bits；每个 patch/关系块取 sum/mean/max，XGBoost 只使用 train-only 特征，另以相同特征训练 MLP。

| head | official-valid MAE | train+valid refit test MAE |
|---|---:|---:|
| XGBoost | `0.515355` | `0.545977` |
| MLP | `0.456223` | `0.454443` |

XGBoost 的 `0.515355` 明显弱于 radius-3 typed-WL count 的 `0.370918` 和 hierarchical backoff 的 `0.354605`。因此，直接把 pair relation 全量拼接起来不是当前有效的长程表示；MLP 的数字不能与 XGBoost 直接排名，只能说明可学习 relation head 可能需要单独协议。

### 6.8 ZINC 的非 XGBoost 神经旁线：只作边界，不并入主表

为了判断“XGBoost 表示强，还是需要可学习交互”，还跑过 topology-only rooted-WL radius-3 的小型神经控制；这些模型不使用 KSVD，且训练预算/目标与 XGBoost 不同。

在 full `10000/1000/1000` 的 fast run 中：

| 模式 | valid MAE | test MAE |
|---|---:|---:|
| attribute-only | `0.494290` | `0.530752` |
| structure-only | `1.119067` | `1.158670` |
| center concat | `0.269540` | `0.272167` |
| conditional fusion | `0.238657` | `0.209998` |
| conditional relation | `0.219294` | `0.238209` |

另一次 100 epoch shuffle confirmation 中，conditional fusion 为 `0.216337/0.212565`，shuffle 为 `0.326837/0.318601`。这些数字不能与 XGBoost 的 `0.354605/0.345460` 直接比较：模型、训练目标、epoch 选择和特征融合方式不同；它们只能说明中心级条件交互值得另立协议研究。

在早期 `2000/200/200` 开发切片中，`center_concat`、`conditional_fusion`、`conditional_relation` 的 valid MAE 为 `0.471925/0.469370/0.482359`；这一步没有显示 relation propagation 的稳定收益。后续 full official fast run 的 conditional relation valid 为 `0.219294`，但 test 为 `0.238209`，仍说明必须按完整数据和冻结协议重新判断，不能用小切片结果选结构。

另有一个匹配的原始 GINE 训练控制：ZINC `10000/1000/1000`，batch 32、最多 2000 epochs、按 valid 选 epoch，单 seed 选中第 2 epoch，valid/test 为 `0.566864/0.604222`。它与上面的 conditional fusion 使用不同训练 budget 和模型，不能直接比较，只作为神经基线的协议记录。

### 6.9 ZINC 路线最终判断

已经成立：

- radius-3 局部上下文比 radius-1/2 更有用；
- typed-WL count 比粗 global statistics 和 Morgan count 更强；
- hierarchical backoff 能显著缓解 exact token 长尾；
- structure–attribute binding 可以被 shuffle control 检测。

没有成立：

- K-SVD `FINAL` 稳定优于 `INIT/raw`；
- 复杂 composition、粗 long-range pair、conditional match gate 能稳定继续降低 MAE；
- 仅凭重构误差下降就能声称 KSVD 产生了 ZINC 任务收益。

---

## 7. 统一结果表：如何正确引用

### 7.1 MolHIV

| 分支 | 关键设置 | valid | test | 应如何表述 |
|---|---|---:|---:|---|
| KSVD standalone rich | `524D` patch，`D=8,T=2`，LR | `0.71213±0.00988` | 本阶段不作为终端数字 | 最强 KSVD-specific 归因 |
| localized KSVD-GINE | radius-2 `848D`，`D=32,T=3`，GINE h64/l3 | `0.794569` | `0.751893` | valid 有结构信号，test 未超过 GINE |
| GINE h64 control | h64/l3 | `0.769955` | `0.752695` | matched neural baseline |
| clean XGBoost `S+marginal` | invariant local distribution，16-trial Optuna | **`0.841285`** | `0.788872` strict | 当前 valid 性能主线，非 KSVD |
| clean XGBoost `S+both` | cross-cov + binding，PCA-8 | `0.838222` | **`0.808749`** strict | 机制/迁移候选，不能据 test 反调 |
| real-prototype MIL | observed patch prototypes，非 KSVD | `0.802488` | `0.767771` | 真实 prototype 竞争力证据 |
| clean rooted-WL late fusion | `T+A` + centered，固定 50/50，非 KSVD | `0.7962`（centered） | **`0.7839` ensemble** | 独立专家互补控制 |

### 7.2 ZINC

| 分支 | 关键设置 | valid MAE | test MAE | 应如何表述 |
|---|---|---:|---:|---|
| KSVD r3 fixed | `K=24,T=3,6 iters`，global+raw+FINAL | `0.5225` | `0.5400` | radius-3 + raw 的结果，不能归因于 update |
| KSVD r3-wide Optuna | `max_nodes=20` | `0.4967` | **`0.5080`** | 冻结后的 KSVD proxy 终端检查 |
| typed-WL count | radius-3，top-K 2048，20 trials | `0.370918` | `0.376450` | 非 KSVD 强对照 |
| hierarchical backoff | four WL rounds + lower-order fallback | **`0.354605`** | **`0.345460`** | 当前 ZINC XGBoost 主结果，关闭 KSVD |
| hierarchical composition | backoff + extra composition | `0.378619` | `0.402539` | 失败扩展 |
| typed match | 1024 prototypes，K-SVD disabled | `0.410045` | `0.395111` | candidate，不是 exact mentor replication |
| exact radius-2 pair relation | `840D` patch + all centre pairs，K-SVD disabled | `0.515355` | `0.545977` | 直接关系读出，弱于层级回退 |

---

## 8. 已证明、未证明、已停止

### 8.1 已证明

- patch 必须置换不变；旧的 node-ID tie-break 结果不能继续使用；
- 跨图共享字典是 KSVD 稀疏码可比较的前提；
- RW 只在任务确实需要更大感受野时有价值；
- MolHIV standalone 中，KSVD rich 相对 matched random-patch 有稳定优势；
- MolHIV 的 task-aware dictionary 能恢复一部分标签信号，但 dense task-aware projection 略高，不能归因于稀疏 KSVD；
- MolHIV clean XGBoost 中，保留所有中心的 local population distribution 比简单 mean 更有用；
- MolHIV 中独立 marginal/interaction 专家做固定 late fusion 的结果高于任一单一专家，但这仍不是 KSVD 证据；
- MolHIV 的 cross-centre structure–attribute covariance 具有机制和一定迁移信号；
- ZINC radius-3、typed-WL count、hierarchical backoff 都有明确的预测价值；
- 结构—属性 binding 可以被 shuffle control 检测，但检测到机制不等于得到稳定 target increment；
- MolHIV 的 MLP/attention 控制没有稳定超过属性边际，ZINC 的 exact pair relation 也没有超过 typed-WL hierarchy。

### 8.2 尚未证明

- “无监督 K-SVD update 本身”在 MolHIV 或 ZINC 上稳定带来分类/回归增益；
- KSVD-GINE 在 official scaffold test 上超过 GINE；
- prototype/occurrence 的关系传播是 KSVD-specific；
- XGBoost 的 `0.84` MolHIV 或 ZINC `0.345` MAE 是 KSVD 的结果；
- 旧导师 `69D/624D` schema 已被 exact replicate；当前实现只能叫 proxy/clean reconstruction。

### 8.3 已停止的方向

- 图级结构向量直接 concat 到 GINE；
- 盲目增加 Coverage、RW 长度、字典大小、稀疏度、GINE 宽度/深度；
- SVM/HGB 等非线性 readout 在小 valid 上继续调参；
- naive positive/negative dictionary、label-augmented KSVD、监督 atom selection；
- exact orbit、typed-edge、patch relation、粗 long-range pair；
- ZINC conditional histogram、SVD16 residual、hierarchical composition、conditioned match gate；
- 用 official test 结果反向选择新的 view、权重或超参数。

### 8.4 当前最合理的路线定位

```text
KSVD：共享、可还原、稀疏的结构压缩器与诊断工具；在 standalone 中有独立正证据。

XGBoost：当前固定特征下游的主性能工具；主信号来自 invariant local object、
population distribution、typed-WL token 和 hierarchy，而不是 KSVD update。
```

如果下一步仍要坚持 KSVD 核心，优先级应是：

1. 先固定 clean invariant local object；
2. 将 KSVD 与 raw/INIT/PCA/random 做严格 matched attribution；
3. 只在新的 scaffold outer split 上验证；
4. 研究 KSVD-induced incidence/lifting 或外部无标签 dictionary coverage；
5. 不再用同一 official-valid/test 反复筛选模型。

---

## 9. 数据泄漏、无效结果与引用注意事项

### 9.1 MolHIV test 的状态

早期 feasibility 和部分路线已经查看过 official test。后续冻结实验虽然没有用 test 选参数，但只能称 controlled terminal evaluation，不能称 untouched test。当前 test 结果用于判断“已冻结方案能否迁移”，不能用于继续改路线。

### 9.2 不能引用的历史高分

`RADIUS2_ATOM_CHEM_20260830.md` 的 `S+R_raw≈0.8307` 及 `S+R_final test≈0.8022` 必须标为 legacy/invalid-for-attribution：对象不满足置换不变性，且存在截断和类别碰撞。正式引用应使用后续 `INVARIANT_PATCH_MECHANISM_SCREEN_20260831.md` 与 `READOUT_DIAGNOSIS_20260901.md`。

### 9.3 不要混用的数字

- MolHIV ROC-AUC 与 ZINC MAE 不可横比；
- official valid/test 与 train-only internal scaffold folds 不可横比；
- KSVD rich、real-prototype、PCA、raw typed-WL 不是同一模型；
- XGBoost、GINE、MIL、Transformer 的结果只在协议一致时比较；
- 小子集上只有很少 positive 或 valid molecule 的高 AUC 不应升级为正式结论；
- ZINC `0.354605` 的 hierarchical backoff 是非 KSVD XGBoost 结果；
- ZINC 神经 conditional-fusion 的 `0.21` 左右 MAE 使用不同模型和训练协议，不能与 XGBoost 数字直接排名。

---

## 10. 结果源索引

### MolHIV：KSVD 主线

- [MolHIV 阶段首轮结果](../../tracks/ksvd/results/molhiv/MOLHIV_PHASE_SUMMARY.md)
- [设计消融](../../tracks/ksvd/results/molhiv/DESIGN_VERDICT.md)
- [C4/C8 多跳机制](../../tracks/ksvd/results/C4_SUMMARY.md)
- [CoverageRW 图级闭环](../../tracks/ksvd/results/GRAPH_LEVEL_SOLID_SUMMARY.md)
- [KSVD standalone 最终路线](../../tracks/ksvd/results/molhiv/KSVD_FINAL_ROUTES.md)
- [KSVD MolHIV 冻结最终报告](../../tracks/ksvd/results/molhiv/KSVD_MOLHIV_FINAL_REPORT.md)
- [节点级 token 与容量审计](../../tracks/ksvd/results/molhiv/KSVD_BREAKTHROUGH_ROADMAP.md)
- [occurrence 二部模型](../../tracks/ksvd/results/molhiv/ATOM_OCCURRENCE_BIPARTITE_SCREEN_20260728.md)
- [feature-conditioned occurrence GNN](../../tracks/ksvd/results/molhiv/FEATURE_CONDITIONED_OCCURRENCE_GNN_20260729.md)
- [GNN-free real-prototype terminal](../../tracks/ksvd/results/molhiv/KSVD_STABLE_REALPROTOTYPE_FULL_OFFICIAL_20260728.md)
- [task-aware dictionary](../../tracks/ksvd/results/molhiv/KSVD_TASK_ADAPTED_DICTIONARY_PRIMARY_MIL_20260728.md)
- [MolHIV 槽位桥重构与分类](../../tracks/ksvd/results/molhiv/MOLHIV_SLOT_BRIDGE_RECONSTRUCTION_PILOT_20260817.md)
- [KSVD 初始化的可微 task-aware dictionary](../../tracks/ksvd/results/molhiv/DIFFERENTIABLE_TASK_DICTIONARY_MIL_PILOT_20260824.md)
- [task-aware KSVD 与 GNN gate](../../tracks/ksvd/results/molhiv/TASK_AWARE_KSVD_VS_GNN_GATE_20260824.md)
- [真实 prototype 二部模型 full official](../../tracks/ksvd/results/molhiv/REALPATCH_ATOM_OCCURRENCE_BIPARTITE_FULL_OFFICIAL_20260728.md)
- [真实 prototype 距离尺度消融](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_COMPACT_DISTANCE_ABLATION_20260728.md)
- [真实 prototype pair-PCA official-valid](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_PAIR_PCA_OFFICIAL_VALID_20260728.md)
- [真实 prototype compact official-test](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_COMPACT_OFFICIAL_TEST_20260728.md)
- [真实 prototype relation full-matrix 对照](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_FROZEN_LINEAR_DISTANCE_RELATIONS_20260728.md)
- [真实 prototype relation random-walk 消融](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_COMPACT_RANDOMWALK_ABLATION_20260728.md)
- [真实 prototype pairwise relation basis](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_PAIRWISE_RELATION_BASIS_20260728.md)
- [真实 prototype task-matched relation sidecar](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_TASKMATCHED_RELATION_SIDECAR_20260728.md)
- [真实 prototype selector 稳健性](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_TASKMATCHED_RELATION_ROBUSTNESS_20260728.md)
- [真实 prototype occurrence residual](../../tracks/ksvd/results/molhiv/REALPROTOTYPE_LOW_CAPACITY_OCCURRENCE_RESIDUAL_20260729.md)
- [8k GNN-free raw-patch terminal](../../tracks/ksvd/results/molhiv/KSVD_GNNFREE_OFFICIAL_SPLIT_TERMINAL_20260728.md)

### MolHIV：luyin16 XGBoost / clean statistics

- [导师固定特征 proxy 判定](../../tracks/ksvd/results/luyin16/MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md)
- [历史 typed-slot 融合 proxy](../../tracks/ksvd/results/luyin16/MENTOR_FUSED_PROXY_SEARCH_20260829.md)
- [无效对象审计](../../tracks/ksvd/results/luyin16/PATCH_OBJECT_AUDIT_20260831.md)
- [不变 patch mechanism screen](../../tracks/ksvd/results/luyin16/INVARIANT_PATCH_MECHANISM_SCREEN_20260831.md)
- [clean structural-role fusion](../../tracks/ksvd/results/luyin16/CLEAN_STRUCTURAL_ROLE_FUSION_20260901.md)
- [readout diagnosis](../../tracks/ksvd/results/luyin16/READOUT_DIAGNOSIS_20260901.md)
- [exact topology × conditional chemistry](../../tracks/ksvd/results/luyin16/EXACT_CONDITIONAL_FUSION_20260901.md)
- [cross-centre interaction route](../../tracks/ksvd/results/luyin16/CROSS_CENTER_INTERACTION_ROUTE_VERDICT_20260902.md)
- [official-valid frozen XGBoost](../../tracks/ksvd/results/luyin16/CROSS_CENTER_INTERACTION_OFFICIAL_VALID_20260902.md)
- [official-test controlled evaluation](../../tracks/ksvd/results/luyin16/CROSS_CENTER_INTERACTION_OFFICIAL_TEST_20260902.md)
- [PCA-16 sensitivity](../../tracks/ksvd/results/luyin16/CROSS_CENTER_INTERACTION_PCA16_CROSSCOV_SUMMARY_20260903.md)
- [PCA-independent interaction diagnostic](../../tracks/ksvd/results/luyin16/INVARIANT_CONDITIONAL_INTERACTION_20260902.md)
- [联合 patch reconstruction 与 K-SVD INIT/FINAL](../../tracks/ksvd/results/luyin16/INVARIANT_JOINT_RECONSTRUCTION_20260901.md)
- [纯结构 69D/624D proxy](../../tracks/ksvd/results/luyin16/PURE_STRUCTURAL_FUSION_20260829.md)
- [T readout 消融](../../tracks/ksvd/results/luyin16/T_READOUT_ABLATION_20260830.md)
- [可学习结构—属性融合架构快筛](../../tracks/ksvd/results/luyin16/ROOTED_CONDITIONAL_ARCHITECTURE_SCREEN_20260901.md)
- [任务对齐 interaction 调参审计](../../tracks/ksvd/results/luyin16/TASK_ALIGNED_INTERACTION_TUNING_20260901.md)
- [冻结 rooted-WL late fusion terminal test](../../tracks/ksvd/results/luyin16/STRUCTURAL_ROLE_TERMINAL_TEST_20260901.md)
- [MolHIV MLP 控制](../../tracks/ksvd/results/luyin16/MOLHIV_MLP_CONTROLS_20260903.md)
- [MolHIV balanced BCE 控制](../../tracks/ksvd/results/luyin16/MOLHIV_MLP_CONTROLS_BALANCED_BCE_20260903.md)
- [MolHIV MLP positive-weight search](../../tracks/ksvd/results/luyin16/MOLHIV_MLP_WEIGHT_SEARCH_20260903.md)
- [MolHIV center conditional fusion](../../tracks/ksvd/results/luyin16/MOLHIV_CENTER_CONDITIONAL_FUSION_20260903.md)

### ZINC

- [ZINC 长程与 KSVD factorial](../../tracks/ksvd/results/luyin16/ZINC_LONG_RANGE_FACTORIAL_20260830.md)
- [ZINC binding / relation mechanism screen](../../tracks/ksvd/results/luyin16/ZINC_MECHANISM_SCREEN_20260830.md)
- [conditional joint v2](../../tracks/ksvd/results/luyin16/ZINC_CONDITIONAL_JOINT_V2_20260830.md)
- [conditional residual SVD16](../../tracks/ksvd/results/luyin16/ZINC_CONDITIONAL_RESIDUAL_SVD16_20260830.md)
- [long-range object relation](../../tracks/ksvd/results/luyin16/ZINC_LONG_RANGE_OBJECT_RELATION_20260830.md)
- [radius-2/radius-3 multiscale](../../tracks/ksvd/results/luyin16/ZINC_MULTISCALE_RADIUS_20260830.md)
- [typed motif count](../../tracks/ksvd/results/luyin16/ZINC_STEP_A_MOTIF_COUNT_20260903.md)
- [radius-3 typed-WL marginal](../../tracks/ksvd/results/luyin16/ZINC_RADIUS3_TYPED_WL_L1_20260903.md)
- [hierarchical exact-token backoff](../../tracks/ksvd/results/luyin16/ZINC_HIERARCHICAL_BACKOFF_20260904.md)
- [hierarchical composition](../../tracks/ksvd/results/luyin16/ZINC_HIERARCHICAL_COMPOSITION_20260904.md)
- [typed match candidate](../../tracks/ksvd/results/luyin16/ZINC_TYPED_MATCH_20260904.md)
- [conditioned match gate](../../tracks/ksvd/results/luyin16/ZINC_HIERARCHICAL_CONDITIONED_MATCH_GATE_20260904.md)
- [ZINC exact pair relation](../../tracks/ksvd/results/luyin16/ZINC_EXACT_PATCH_RELATION_20260904.md)
- [ZINC center relation neural control](../../tracks/ksvd/results/luyin16/ZINC_CENTER_RELATION_NETWORK_OFFICIAL_FAST_20260903.md)
- [ZINC conditional fusion shuffle confirmation](../../tracks/ksvd/results/luyin16/ZINC_CONDITIONAL_FUSION_SHUFFLE_20260903.md)
- [ZINC matched GINE control](../../tracks/ksvd/results/luyin16/ZINC_MATCHED_GINE_SMOKE_20260903.md)
- [ZINC mechanism route synthesis](../../tracks/ksvd/results/luyin16/ZINC_MECHANISM_ROUTE_SYNTHESIS_20260830.md)

> 时间边界：本文只纳入截至 2026-09-04 已完成并可审计的记录；更晚生成的实验不纳入本文结论。
