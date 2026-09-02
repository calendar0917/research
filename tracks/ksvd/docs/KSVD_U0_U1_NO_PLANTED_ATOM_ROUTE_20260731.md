# KSVD U0/U1：无人工原子词表的从零路线

> 日期：2026-07-31
>
> 状态：**U0-R / U0-P / U0-D / U1A 已完成；Level 2 通过，Level 3 未通过**
> 主要依据：`docs/luyin/luyin10.txt`、`docs/luyin/luyin11.txt`，以及 E0/E1、G0、G0B 的已有审计结果。

## 0. 先回答两个根本问题

### 0.1 是否必须让每个 atom 都“有意义”？

**不必须。**需要把“有意义”拆成三个互不等价的层次：

1. **人类可解释性**：atom 能否被命名成 triangle、cycle、star 等 motif；
2. **统计有效性**：atom 是否被使用、是否与其他 atom 重复、是否跨训练数据保持一定稳定性；
3. **任务有效性**：atom 的图级 code 是否给下游预测带来信息。

普通 KSVD 优化的是稀疏重建：

\[
\min_{D,X}\|Y-DX\|_F^2,
\qquad \|x_i\|_0\le T.
\]

它没有直接优化分类损失，所以不能预先要求：

- 每个 atom 都对应一个完整合法 motif；
- 每个 atom 都能被人类命名；
- 每个 atom 都对下游有贡献；
- atom 与某个人工词表一一对应。

允许出现：

- 背景 atom；
- 多个 atom 联合表示一种局部结构；
- 一个 atom 混合多种可观察变化；
- 统计稳定、对下游有用，但无法简单画成 motif 的 latent structural basis。

只有在训练后通过 top-activating patches、leave-one-atom-out、code permutation 等分析，才能讨论哪些 atom 有用。**人类可解释性是可选的 post-hoc 性质，不是 U0/U1 的通过条件。**

### 0.2 使用可控图，是否仍是在人工选择 atom？

不等价。任何实验都必须选择数据分布；关键在于人工控制的是哪一层。

本路线允许人工控制：

- 图的节点数、度数、连通性；
- 一个全局随机生成过程；
- train/validation/test 划分；
- patch sampler、字典容量和稀疏度。

本路线禁止人工提供：

- 有限 motif vocabulary；
- `D_true`；
- motif cell boundary；
- patch 的 motif label；
- “恢复某四个 motif”这种 atom-level 成功标准。

换句话说：

> **我们选择图分布，但不选择该分布应当被分解成哪些 atom。**

U0 的图由全局 degree-preserving rewiring 产生，而不是把 triangle、cycle、star 等局部模板拼接起来。生成器本身当然会产生统计偏好，但字典如何概括这些偏好由训练数据决定。这正是受控环境中的 data-driven discovery，而不是 planted-dictionary recovery。

---

## 1. 已有实验在新路线中的位置

| 阶段 | 已回答的问题 | 不能外推的内容 |
|---|---|---|
| E0/E1 | 数值实现、稀疏恢复和初始化可辨识性 | 不代表大图 discovery；多次 restart 只用于诊断 |
| G0 | 在四种 clean canonical columns 上，单次随机列初始化可被 KSVD 更新到完整 vocabulary | oracle cell extractor；只有四种唯一列；人工 motif 集 |
| G0B | noisy untyped canonical adjacency 下，“固定 edge-mask motif atom”可能不可辨识 | 只否定该强语义，不否定 latent basis |
| **U0/U1** | 无 `D_true`、无 motif vocabulary 时，KSVD 是否形成健康字典，以及 codes 是否可用于图级任务 | 本轮仍是 synthetic pipeline gate，不是 MolHIV/真实数据结论 |

因此，原计划中的 G0C identifiable planted-motif family 最多保留为单元正控制，**不再作为主路线的下一步**。

---

## 2. 主研究命题

### 2.1 U0：无标签字典命题

> 在不给 KSVD motif 词表、motif label 或图 label 的条件下，从一个通用局部图 patch 分布中，单次可部署初始化后的 KSVD，能否得到比相同 INIT 更好的 held-out 稀疏重建，同时不发生严重字典坍缩？

### 2.2 U1A：最小图级效用命题

> 用非常简单、符号不变的 graph readout 汇总 patch codes 后，FINAL codes 是否包含可泛化的图结构信息；它们相对相同 INIT、固定随机字典和简单 patch controls 是否有增益？

U1A 是 **signal-exposure positive control**。它只确认“局部 patch → sparse codes → graph readout”链条可用，不要求立刻超过所有全局图统计。

### 2.3 暂不在本轮检验

- 每个 atom 是否为可命名 motif；
- patch 之间的 overlap/incidence/binding；
- GNN 融合；
- 节点/边属性；
- MolHIV、CIN 或 benchmark 排名；
- supervised/discriminative KSVD；
- 多 sampler、大规模超参搜索或多 restart。

---

## 3. 为什么首先选 degree-preserving rewiring

U0 使用 4-regular ring lattice 经过 double-edge swaps 形成图族。

这样做不是因为它“是真实世界”，而是因为它同时满足四个基础要求：

1. 所有图均有 `n=60`；
2. 所有图均有 `m=120`；
3. 每个节点的 degree 恒为 4；
4. 图仍可在 clustering、cycle closure、path length 和 mixing 等结构上发生变化。

因此，下游不能只靠图大小、边数或 degree distribution 完成任务。更重要的是，生成过程是全局变换，没有隐藏的局部 atom 列表。

### 3.1 初始图

节点为 `0,...,59`。每个节点连接环上的 `±1` 和 `±2`：

\[
E_0=\{(i,i\pm1),(i,i\pm2)\}\pmod {60}.
\]

该图是连通的 4-regular simple graph。

### 3.2 一次 accepted double-edge swap

随机选择两条无共同端点的边 `(a,b)` 与 `(c,d)`，随机尝试一种交叉连接：

- `(a,c),(b,d)`，或
- `(a,d),(b,c)`。

若出现 self-loop、重复边或导致图不连通，则拒绝并重采样；否则接受。每次 accepted swap 严格保持每个节点的 degree。

### 3.3 U1A 两个结构 regime

- `LOW`：每张图独立均匀采样 `20..40` 次 accepted swaps；
- `HIGH`：每张图独立均匀采样 `60..80` 次 accepted swaps。

最后对 60 个节点做全局随机置换。swap 次数和 LOW/HIGH label 不进入 patch learner。

这两个 regime 只是定义一个最小正控制任务，不是 atom 定义。若 U1A 通过，后续 U1B 再换成与生成参数分离的动态/连续目标。

---

## 4. 通用、非 oracle 的 patch extractor

### 4.1 固定配置

- 每张图恰好提取 `M=24` 个 patch；
- root 从 60 个节点中均匀、无放回抽取；
- 每个 patch 含 `s=6` 个不同节点；
- 从 root 开始普通随机游走；
- 记录首次访问新节点的顺序，直到累计 6 个不同节点；
- 若 80 步内不足 6 个不同节点，则丢弃该 walk 并从同一 root 重试；最多 20 次；
- 对这 6 个节点取 **induced subgraph**，而不是只保留 walk traversed edges。

必须保存但不提供给 learner 的审计信息：root、walk trace、所选节点和 graph ID。

### 4.2 为什么不是 oracle motif sampler

- root 与任何隐藏结构无关；
- 没有 cell boundary；
- sampler 不知道 LOW/HIGH label；
- sampler 不寻找 triangle、cycle 或其他预定义形状；
- 每张图 patch 数相同，不允许用 patch 数量泄漏 label。

随机游走只是第一版通用 receptive-field 机制，不被预设为最终最优 sampler。若表示 gate 失败，只能说明当前 sampler/representation 没暴露信号，不能直接否定 KSVD。

### 4.3 第一版候选：rooted exact canonicalization

最初候选把 root 固定为 canonical slot 0；只对其余 5 个节点的 `5!=120` 种排列穷举，选择 adjacency upper triangle 的 lexicographically minimum 15 维 binary vector。

这样：

- 保留 sampler 提供的可观察 root；
- 消除非 root 节点编号；
- 不使用 generator oracle anchor；
- 不保留 walk 达到 6 个不同节点所用的步数，避免 trace-length 泄漏。

必须做两类自测：

1. 任意重排 5 个非 root 节点，canonical vector 完全相同；
2. 对整张图全局重编号，并同步映射 root 与同一组已选节点，canonical vector 完全相同。

随机 sampler 的分布置换不变性另作统计审计，不要求两次有限随机采样得到逐项相同的 patch multiset。

### 4.4 重要修正：canonical 不等于 KSVD 坐标有意义

阶乘枚举只保证 rooted-isomorphic patches 得到同一个代表元。它不保证：

- canonical slot 1/2/... 在不同非同构 patch 中承担稳定结构角色；
- 一条边的小扰动只改变一个 canonical coordinate；
- canonical-vector Euclidean/Hamming distance 接近真实的 root-preserving graph edit distance；
- 线性组合 `D x` 在图结构上具有自然解释。

因此在 U0-P 和 U0-D 之前新增 **U0-R adjacency representation audit**，同时比较：

1. `rooted_canonical_adjacency`：root 固定，其余节点做 120 个排列；
2. `walk_order_adjacency`：节点槽位直接定义为随机游走的首次发现顺序。

第二种表示不把同一个 induced subgraph 的所有 walk ordering 强行合并，但每个坐标具有明确的 sampler 语义：`(i,j)` 表示第 `i`、`j` 个首次发现节点之间是否有边。对整张图全局重编号，只要同步映射同一条 walk trace，向量严格不变。

U0-R 检查 permutation invariance、one-edge perturbation amplification、与 exact rooted graph-edit distance 的相关性，以及相同 patch 在不同合法 ordering 下的变化。**在 U0-R 完成前不冻结 KSVD 主表示。**详细协议见：

`tracks/ksvd/docs/KSVD_U0R_ADJACENCY_REPRESENTATION_AUDIT_20260731.md`

### 4.5 U0-R 已完成后的表示决定

三个独立 audit seeds 的结果一致：

- rooted canonical permutation invariance = `1.0`；
- sampled rooted-isomorphism injectivity disagreement = `0`；
- 但一次真实 edge flip 平均造成约 `3.27` 个 canonical coordinates 改变；
- 约 `62.1%–62.4%` 的 one-edge flips 被放大为多于一个 coordinate；
- 约 `36.9%–37.5%` 被放大为至少四个 coordinates；
- canonical distance 与 exact rooted GED 的 Spearman 只有约 `0.58–0.64`；
- walk-order distance 的 Spearman 约为 `0.63–0.70`，且同一 walk ordering 下 one-edge flip 严格只改变一个 coordinate。

因此当前判定是：

> `SELECT_WALK_ORDER_PENDING_U0P`

U0-P 的主表示改为 `walk first-discovery-order adjacency`。rooted canonical adjacency 保留为不变性 baseline，不再因为“阶乘枚举保证同一性”就默认送入 KSVD。

结果见：

- `tracks/ksvd/results/from_scratch/U0R_ADJACENCY_AUDIT_20260731.md`
- `tracks/ksvd/results/from_scratch/u0r_adjacency_audit_20260731.json`

---

## 5. 数据、预处理和现实训练预算

### 5.1 五个 independent data replicates

每个 replicate 都重新生成图和随机游走样本：

- train：300 图，LOW/HIGH 各 150；
- validation：100 图，各 50；
- test：200 图，各 100；
- train patches：`300 × 24 = 7200`；
- validation/test 永不参与字典学习。

五个 master data seeds 固定为：`731101, 731102, 731103, 731104, 731105`。每个 master seed 通过 `SeedSequence` 派生 graph generation、global permutation、root sampling 和 walk streams，禁止共享可变全局 RNG。

五个 replicates 用于检查结论是否依赖某一批训练图，**不是在同一个数据集上运行五次初始化并挑最好结果**。

每个 replicate 只允许一个冻结初始化、一个 FINAL 字典和一个下游模型选择过程。禁止根据 test 结果挑 data seed、walk seed、initialization seed 或字典。

### 5.2 无标签字典训练

字典训练时：

- 混合 train 中 LOW/HIGH patches；
- 丢弃 graph label、swap count、graph ID 和 walk metadata；
- 只接收冻结的 15 维 patch representation；U0-R 后主表示为 walk first-discovery-order adjacency；
- 只使用 train feature mean 做中心化，不做 per-patch norm normalization；
- validation 只选择下游逻辑回归正则，不选择字典。

不做 per-patch normalization，是为了不先验删除 induced edge count 等真实局部结构强度。所有 baseline 使用相同 train mean 规则。

---

## 6. 冻结的 KSVD 配置

- patch dimension：`d=15`；
- atoms：`K=12`；
- sparsity：`T=2`，`T_min=1`；
- updates：`25`；
- atom/code threshold：`1e-10`；
- 不使用 anchor、coherence penalty 或 projected atom；
- 不搜索 `K/T/iterations`；
- 不做 restart。

`K=12` 是模型容量，不表示存在 12 个真实 motif。

### 6.1 单次 deterministic maximin initialization

在 centered train patch columns 中：

1. 选择范数最大的列作为第一个 atom；
2. 后续选择与当前 atom 集合的最大 absolute cosine 最小的训练列；
3. 所有 tie 确定性解决；
4. 列归一化后作为 `D_init`。

每个 replicate 从完全相同的 `D_init` 比较：

- `INIT`：`n_iter=0`，只运行 OMP；
- `FINAL`：`n_iter=25`，运行普通 KSVD 更新。

maximin 本身是强而可部署的 initializer。如果 INIT 已经足够好，这是重要结论，而不是需要用更差初始化掩盖的问题。

KSVD 中 dead-atom handling 的内部 seed 固定为 0；不据此生成候选模型。

---

## 7. U0：先只审计字典，不做分类结论

### 7.1 U0-G：generator invariants

每张图必须满足：

- adjacency 对称、对角为 0；
- simple graph；
- connected；
- `n=60`、`m=120`；
- 所有节点 degree 恰好为 4；
- 实际 accepted swaps 等于请求值。

任一失败先修生成器，不运行 KSVD。

### 7.2 U0-P：representation exposure gate

在不训练字典前，检查当前 patch pipeline 是否至少看到了 regime 信号：

1. induced-edge-count histogram：每图统计 6-node patch 中 5 到 15 条边的频率；
2. primary walk-order summary：15 维坐标的 graph-wise mean 和 standard deviation；
3. rooted-canonical summary baseline：同样的 graph-wise mean 和 standard deviation；
4. label shuffle control。

使用 train 拟合单个线性分类器、validation 选正则、test 冻结评估。

通过条件：

- 至少一个真实 patch control 在 5 个 replicate 中有 4 个 test balanced accuracy `>=0.65`；
- 其五次均值 `>=0.70`；
- label shuffle 均值位于 `[0.45,0.55]`。

U0-P 失败时停止：KSVD 不可能从当前输入恢复 extractor 没有暴露的信息。

### 7.3 U0-D：dictionary optimization / health

报告 train/validation/test：

- relative reconstruction error；
- NMSE；
- mean nonzeros per patch；
- 每个 atom 的 activation frequency；
- activation entropy 与 effective atom count `exp(H)`；
- dead atom count，dead 定义为 test activation `<0.5%`；
- maximum absolute off-diagonal coherence；
- 单 atom 最大 activation share；
- INIT→FINAL reconstruction curve。

U0-D 的基础通过条件：

1. FINAL 相对 INIT 的 held-out relative reconstruction error 降低至少 10%，且 5 个 replicate 中至少 4 个成立；
2. 五次平均相对降低至少 15%；
3. test 上至少 6/12 atoms 不是 dead；
4. effective atom count 至少为 4；
5. maximum absolute coherence `<0.95`。

这里明确允许部分 atom 是背景、低使用或不可解释的；门槛只排除整个字典严重坍缩，不要求 12 个 atom 全部“有意义”。

跨 replicate 的 Hungarian atom cosine 和 subspace similarity 只作描述性指标，不要求全部 atom 一一稳定，因为无监督字典本身可能存在等价旋转/替代基。

---

## 8. U1A：最简单的 graph-level code

### 8.1 符号不变 readout

对一张图的 patch code matrix `X_g ∈ R^(K×M)`，每个 atom 只汇总三项：

1. activation frequency：
   \[
   f_j=\frac1M\sum_i \mathbf{1}(|x_{ji}|>10^{-10});
   \]
2. mean absolute coefficient：
   \[
   a_j=\frac1M\sum_i |x_{ji}|;
   \]
3. coefficient RMS / energy：
   \[
   e_j=\sqrt{\frac1M\sum_i x_{ji}^2}.
   \]

得到 `3K=36` 维 graph vector。该 readout 对 atom sign ambiguity 不敏感，也对应录音中“把结构所包含的能量 read out”的最小版本。

本轮不使用 attention、MLP、GNN 或 patch relation graph。

### 8.2 下游模型

- 标准化参数只在 train graph features 上拟合；
- L2 logistic regression；
- 正则网格固定为 `{1e-4,1e-3,1e-2,1e-1,1,10}`；
- 只用 validation balanced accuracy 选一次；tie 选择更强正则；
- test 只评估一次；
- 报告 5 个 paired data replicates 的 mean、std、min 和逐 seed 结果。

### 8.3 最小 baseline 集

| Baseline | 回答的问题 |
|---|---|
| graph constants / simple stats | 固定 degree 后，常见全局统计能做到什么 |
| patch edge-count histogram | 是否只需一个局部密度标量 |
| raw canonical mean/std | 不学字典是否已足够 |
| fixed Gaussian dictionary, `K=12,T=2` | 任意投影加稀疏编码是否足够 |
| deterministic medoid bag, `K=12` | 代表性真实 patch 的硬聚类是否足够 |
| PCA-12 | 普通线性压缩是否足够 |
| **INIT codes** | initializer 本身提供多少价值 |
| **FINAL KSVD codes** | KSVD 更新后增加了什么 |
| graph-code shuffle | 分类器是否利用了真实 graph/code 对应 |
| label shuffle | 完整评估管线是否产生伪信号 |

其中 INIT 与 FINAL 必须使用同一个 `D_init`、相同 OMP 和相同 graph readout。

`simple stats` 第一版固定为：`n_nodes`、`n_edges`、density、degree mean/std/min/max、triangle count、transitivity、average shortest-path length、diameter、connected-component count。这里不加入根据 test 表现临时想到的新统计。常数特征由 train standardizer 安全删除。

`fixed Gaussian dictionary` 使用每个 replicate 固定 seed 0 生成的 15×12 Gaussian columns 并逐列归一化，不做字典更新。`deterministic medoid bag` 使用与 maximin 相同的 12 个真实训练列，但对每个 patch 只做 nearest-medoid one-hot assignment。PCA 只在 centered train patches 上拟合。所有 patch representation 都使用同一 24-patch graph 集，不允许 baseline 获得额外采样。

label shuffle 和 graph-code shuffle 各只使用由该 replicate master seed 派生的一次固定 permutation；它们是负对照，不生成多次候选后挑结果。

### 8.4 分层结论，而不是一个模糊的 PASS

#### Level 1：`PASS_KSVD_OPTIMIZATION`

满足 U0-D。说明 KSVD 在 held-out patches 上确实改善了稀疏重建且未坍缩。

#### Level 2：`PASS_GRAPH_CODE_SIGNAL`

- FINAL mean balanced accuracy `>=0.70`；
- 5 个 replicate 中至少 4 个 `>=0.65`；
- FINAL 比 fixed Gaussian dictionary 平均至少高 3 个百分点；
- shuffle controls 接近 chance。

说明 FINAL codes 包含可用于图级任务的结构信号。

#### Level 3：`PASS_KSVD_ADDED_VALUE`

除 Level 1/2 外，还要求：

- `FINAL - INIT` 平均至少 `+0.02` balanced accuracy；
- 至少 4/5 replicates 中 FINAL 高于 INIT。

这才支持“KSVD 更新本身有下游增益”。

### 8.5 四种关键解释

| 观察 | 允许的结论 |
|---|---|
| reconstruction 与 downstream 都改善 | 当前最强的 KSVD added-value 证据 |
| reconstruction 改善，FINAL≈INIT downstream | KSVD 学会更好重建，但无监督目标与当前任务不一致 |
| INIT 与 FINAL 都好 | patch representation / initializer 有价值，KSVD 更新可能不必要 |
| INIT 与 FINAL 都失败 | 当前 sampler/representation 未形成可用 graph code；不能靠 restart 掩盖 |

U1A 不强制 FINAL 超过 simple global graph stats。因为 LOW/HIGH 本来就是结构 regime 正控制，全局 clustering/path-length 等统计可能非常强。真正的“超越简单统计或提供增量信息”属于下一阶段 U1B。

---

## 8.6 正式执行结果：路线目前停在 Level 2

正式结果文件：

- `tracks/ksvd/results/from_scratch/U0P_SIGNAL_EXPOSURE_20260731.md`；
- `tracks/ksvd/results/from_scratch/U0D_DICTIONARY_AUDIT_20260731.md`；
- `tracks/ksvd/results/from_scratch/U1A_GRAPH_CODE_SIGNAL_20260731.md`。

### U0-P：输入信号通过

五个 replicate 的 test balanced accuracy：

- WALK mean/std：`0.815 ± 0.027`；
- rooted-canonical mean/std：`0.803 ± 0.027`；
- edge-count histogram：`0.826 ± 0.012`；
- WALK label shuffle：`0.501 ± 0.014`。

因此 `PASS_U0P_WALK_SIGNAL_EXPOSED`。这确认 first-discovery-order adjacency 不是任意坐标噪声：固定 sampler-slot 语义下，它确实暴露了生成 regime 的局部结构差异。

### U0-D：单次初始化的 KSVD 优化通过

INIT→FINAL 的 test relative reconstruction error：

- 五次平均相对下降 `34.55%`；
- 每次下降均超过 `31%`；
- 五个 replicate 的 FINAL test 均为 `12/12` non-dead atoms；
- effective atom count 为 `8.56–9.51`；
- maximum coherence 为 `0.556–0.731`。

因此 `PASS_KSVD_OPTIMIZATION`。这是真正的单次可部署初始化结果，不依赖多 restart。它支持“KSVD 能从无人工 atom 词表的 walk-induced patches 中学习健康的重建 basis”。

### U1A：graph code 有信号，但 KSVD added value 未通过

五次 test balanced accuracy 均值：

| control | mean |
|---|---:|
| simple graph statistics | `0.935` |
| edge-count histogram | `0.826` |
| raw WALK mean/std | `0.815` |
| PCA-12 | `0.809` |
| FINAL KSVD codes | `0.750` |
| INIT codes | `0.713` |
| fixed Gaussian codes | `0.709` |
| graph-code shuffle | `0.475` |
| label shuffle | `0.491` |

FINAL 比 fixed Gaussian 平均高 `+0.041`，所以 Level 2 `PASS_GRAPH_CODE_SIGNAL` 通过。

但 FINAL−INIT 虽平均为 `+0.037`，只在 `3/5` replicates 为正，未满足预注册的 `4/5` 一致性条件。因此 Level 3 `PASS_KSVD_ADDED_VALUE` **未通过**。此外 FINAL 明显低于 raw WALK、edge histogram 和 PCA，说明当前 sparse code/readout 丢失了一部分对该任务有用的低阶信息。

当前最准确的路线结论是：

> **无人工 atom 词表的 KSVD 可以学习稳定、非坍缩、显著改善 held-out 重建的 latent basis；这些 codes 也含图级信号。但在当前 rewiring-regime 任务和最小 readout 下，尚不能稳定地把下游增益归因于 KSVD update 本身。**

这不是路线完全失败，也不是 KSVD discovery 已被验证。它把问题清楚地缩小为“无监督重建目标与下游 readout/target 是否对齐”，而不再是随机初始化、人工 motif 或 canonical 同一性问题。

## 9. atom 分析只能放在训练完成之后

正式结论不做 atom-to-motif matching。可做以下 post-hoc 分析：

1. 展示每个 atom 的 top-20 activating real patches，而不是直接把连续 atom threshold 成一张图；
2. leave-one-atom-out：去掉 atom `j` 的三项 graph features 后重新评估；
3. atom-wise permutation：仅打乱 atom `j` 在图之间的 features；
4. 线性分类器权重与跨 replicate 高效用 atom 匹配；
5. top-q atoms 的 cumulative utility；
6. background / utility / interpretable 三类可重叠标签。

这些分析可能发现：只有 2–4 个 atom 对 LOW/HIGH 有用，其余 atom 主要服务重建。这是允许且有信息量的结果。

只有当某 atom 的 top-activating patches 在多个 replicate 中呈现稳定、清晰的同构结构族时，才可谨慎称其为 motif-like atom。否则使用 latent structural basis/component。

---

## 10. 本轮非注册设计探针：为什么 INIT 对照必须保留

在冻结正式样本 seeds 之前，做过一次小型 exploratory probe；它**只用于检验设计是否明显不可运行，不计入正式证据，也不能与后续 test 合并**。

配置近似为：`n=60`、degree 4、LOW 20–40 swaps、HIGH 60–80 swaps、6-node rooted walk patches、`K=12,T=2`、单次 deterministic maximin、25 iterations。

单个小 probe 中：

- train reconstruction relative error：INIT 约 `0.562`，FINAL 约 `0.344`；
- LOW/HIGH test accuracy：INIT code 约 `0.800`，FINAL code 约 `0.806`；
- patch edge-count histogram 约 `0.850`；
- simple graph stats 约 `0.938`。

这个结果不是正式结论，但已经说明路线中最重要的风险：

> KSVD 可能显著改善 patch 重建，却几乎不改善图级任务；而简单统计可能更强。

因此不能只报告 FINAL accuracy，也不能因为 atom 看起来有形状就宣布成功。正式实验必须同时保留 U0 reconstruction、INIT-vs-FINAL attribution 和 simple controls。

---

## 11. 通过 U1A 以后才允许进入的阶段

### U1B：与生成 label 分离的任务

只有 U1A 至少达到 Level 2，才允许**设计** U1B；正式执行前仍需承认当前 Level 3 未通过，U1B 不能被当作对 U1A 失败的事后救援。候选目标包括：

- 固定扩散过程的传播时间；
- 固定 SIR 参数下的平均 final size；
- 随机故障下的连通保持率；
- mixing / hitting-time 类连续目标。

U1B 必须：

- 在字典训练后、独立于 atom 定义计算 target；
- 使用固定模拟 seeds 或足够 Monte Carlo 降低 label noise；
- 比较 `stats`、`stats+INIT`、`stats+FINAL`，检查 FINAL 是否提供增量信息；
- 在方案冻结前不得看 test 结果挑 target。

### U2：patch 之间的关系

只有 content-only FINAL code 已被证明可用，才处理 `luyin11.txt` 的第四个问题：patch 是片段，片段之间的关联尚未保留。

顺序应是：

1. content bag；
2. relation-only；
3. content + relation；
4. true binding；
5. shuffled binding。

不能一开始加入 relation GNN，否则无法判断收益来自 KSVD 内容还是关系模块。

### R：真实图

只有 U1B/U2 中至少一个给出清晰 added value，才进入真实数据。第一步仍应选纯结构、规模可控的数据，而不是直接回到 MolHIV/CIN。

---

## 12. 明确的停止与修正规则

1. **U0-G 失败**：修 generator；不解释算法。
2. **U0-P 失败**：修改 sampler 或 representation；不增加 KSVD restart。
3. **U0-D 失败**：普通 KSVD 在当前 patch 分布上不具备基本优化/泛化能力；先查实现、中心化、容量，再决定是否停止。
4. **U0-D 通过、U1A Level 2 失败**：重建 basis 与 graph task 不匹配；不声称“atom 对下游有意义”。
5. **Level 2 通过、Level 3 失败**：表示链条可行，但证据指向 initializer/raw representation，而不是 KSVD update。
6. **Level 3 通过**：才进入 U1B，检查非生成 label 与 simple-stat incremental utility。
7. 任一阶段都禁止以“再跑更多随机初始化并挑最好”作为修复手段。

---

## 13. 最短执行顺序

该最短序列现已执行到 U1A：

1. generator + invariant tests：完成；
2. U0-R adjacency geometry audit：完成，选择 WALK-order signal；
3. U0-P representation exposure gate：通过；
4. 单次 `D_init`、INIT/FINAL 和 U0-D dictionary report：通过；
5. U1A readout 和 baselines：Level 2 通过，Level 3 未通过；
6. 当前停止正式外推，先冻结下一问题：是把目标限定为“无监督 sparse compressor”，还是继续要求“KSVD update 的稳定 downstream added value”。

若继续后一目标，下一实验不能靠增加 restart 或重复使用当前 test 调参。应另立 U1A-v2/U1B protocol 与新 seeds，明确只改变一个层次：target、sampler、dictionary objective 或 graph readout 四者之一。

这条顺序保留了 `luyin10` 的核心动机——更一般的 receptive field、数据驱动结构基、可还原 sparse representation——同时遵守 `luyin11` 的四个警告：重建不等于图表征、置换问题需单独处理、随机游走不是唯一答案、patch 关系不能被遗忘但必须后置。
