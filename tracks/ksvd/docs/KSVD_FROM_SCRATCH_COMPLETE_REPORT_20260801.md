# KSVD 图结构路线：从零探索完整报告

> 日期：2026-08-01  
> 范围：从最基础的 KSVD 恢复实验，到无人工 atom 词表、邻接表示、WALK/n-hop sampler、raw IMDB-BINARY、下游归因与 basis characterization。  
> 面向读者：不要求预先了解本项目、KSVD 或此前实验。  
> 核心结论：**KSVD 作为无人工 motif 词表的 sparse graph-patch reconstruction basis learner 已得到支持；但普通无监督 reconstruction updates 没有在 raw IMDB-BINARY 上稳定产生超出 INIT 和简单图统计的分类增量。**

---

## 1. 一页摘要

这轮探索试图回答的不是“能否把某个复杂模型调到更高分”，而是逐层回答：

1. KSVD 实现本身能否恢复已知稀疏字典？
2. 邻接矩阵能否作为有意义的线性字典信号？
3. 不人工指定 motif atoms，KSVD 能否自行学习健康 basis？
4. 真实图上的 reconstruction improvement 是否能转化为图分类价值？
5. 如果不能，问题来自初始化、采样、排序、readout，还是 objective 与 task 本身错位？

最终证据链为：

```text
数值 KSVD recovery                         PASS
理想固定槽位结构 recovery                  PASS（部分条件受初始化影响）
人工 motif vocabulary 的无歧义恢复          只能作 positive control
noisy canonical adjacency 固定 motif 解释    FAIL
无人工 atom 词表的 WALK substrate           PASS
单 deterministic INIT 的 KSVD optimization  PASS
真实 IMDB held-out patch reconstruction      PASS
简单 marginal graph-code downstream utility FAIL
fixed-7 direct n-hop sampler repair           FAIL
atom-pair readout repair                      FAIL
statistics-conditioned residual objective     reconstruction PASS / task utility FAIL
label-free continuous basis characterization  部分支持，但不是可命名 graphlet vocabulary
```

因此当前研究边界不是“KSVD 完全无效”，也不是“再加几次随机初始化就会成功”，而是：

> **局部 patch reconstruction basis 与 graph-level task representation 之间不存在自动等价关系。**

---

## 2. 原始问题与研究原则

### 2.1 最初的几个疑问

路线开始时存在四个容易混淆的问题：

1. 真实训练不可能依赖大量随机初始化，实验是否仍然现实？
2. 人工选定 triangle、cycle 等 atoms，是否违背“自发现”的初衷？
3. adjacency vector 即使同构唯一，坐标距离是否真的适合 KSVD？
4. patch reconstruction 更好，是否就意味着图分类更好？

本轮实验的设计就是依次拆开这些问题，而不是一次构造一个包含 sampler、KSVD、GNN 和 benchmark 的复杂系统。

### 2.2 “自发现 atom”的操作性定义

本轮最终采用的定义不是：

> 每个 atom 都必须是一张合法、可命名的 motif 图。

而是：

> 在不给 KSVD motif vocabulary、motif labels 或 graph labels 的情况下，由数据分布本身形成 sparse basis；basis 应当非坍缩、被数据使用，并改善 held-out sparse reconstruction。

这允许 learned atom 是连续方向。它可以代表一组相近 patch 的共同变化，而不必等于某一个离散 graphlet。

### 2.3 可控数据不等于人工选择 atoms

在后来的无人工词表实验中，我们只控制图生成分布，例如 degree-preserving rewiring；没有告诉模型“应当存在 triangle atom、cycle atom”。准确表述是：

> **选择图分布，但不选择该分布应被分解成哪些 atoms。**

### 2.4 全程采用的实验纪律

- 先写协议和 gate，再看正式结果；
- 一次只修改一个层次；
- 明确区分 INIT 与 FINAL；
- reconstruction 与 downstream 分开判断；
- 不从多个 outer-test seeds 或 restarts 中挑最好结果；
- raw IMDB 是主任务，cleaned 只作敏感性解释；
- negative controls 包括 label shuffle、graph-code shuffle；
- 失败后不在相同 test folds 上继续扫描 `K/T/iterations/readout`。

---

## 3. 基础概念和统一符号

### 3.1 Patch signal

每个固定大小 patch 被写成向量：

\[
y\in\mathbb{R}^{d}.
\]

对于 7-node 无向简单图，使用邻接矩阵上三角：

\[
d=\binom{7}{2}=21.
\]

### 3.2 Dictionary 与 sparse code

字典为：

\[
D=[d_1,\ldots,d_K]\in\mathbb{R}^{d\times K},
\]

每个 patch 用稀疏系数表示：

\[
y\approx Da,\qquad \|a\|_0\le T.
\]

主真实数据配置：

```text
K = 12 atoms
T = 2 active atoms/patch
T_min = 1
KSVD updates = 25
```

KSVD 交替执行：

1. 固定字典，用 OMP 求 sparse codes；
2. 固定其他 atoms，逐 atom 用 SVD 更新该 atom 及其激活系数。

### 3.3 INIT 与 FINAL

- `INIT`：KSVD 更新前的字典；
- `FINAL`：从完全相同 INIT 执行 25 次更新后的字典。

只有 paired `FINAL−INIT` 才能归因于 KSVD updates。只报告 FINAL 分数无法区分收益来自初始化器还是 KSVD。

### 3.4 Held-out reconstruction

训练图 patches 用于拟合字典；test 图 patches 只编码、不参与训练。真实数据主要使用 graph-balanced error：

1. 先计算每张图的相对 Frobenius reconstruction error；
2. 再对图等权平均，避免 patch 多的图权重更大。

### 3.5 Graph code

最小 marginal readout 对每个 atom 汇聚：

1. activation frequency；
2. mean absolute coefficient；
3. coefficient RMS。

`K=12` 时得到 `36-D` graph code。它对 atom coefficient 的符号翻转不敏感。

---

## 4. 当前 sampler 的确切实现

这一节专门说明 WALK/n-hop 遇到节点不足、重复访问、变长 ego 或坍缩时如何处理。

## 4.1 WALK：如何得到恰好 7 个不同节点

实现入口：

- `tracks/ksvd/code/from_scratch_unplanted_representation.py`
- `sample_walk_patch()`
- `sample_graph_walk_patches()`

对每个 root：

```text
current = root
seen = {root}
node_ids = [root]

重复：
    从 current 的邻居中均匀随机选下一个节点
    WALK 可以回到已经访问过的节点
    只有首次发现的新节点才加入 node_ids
    当 node_ids 达到 7 个不同节点时停止
```

关键细节：

- WALK trace 可以包含重复节点；
- patch node set 不包含重复节点；
- 不用重复节点、zero padding 或虚拟节点凑满 7；
- 选出 7 个节点后，取这 7 个节点的**完整 induced adjacency**，不是只保留 walk traversed edges；
- slot 0 是 root，后续 slots 是首次发现顺序。

### 4.1.1 如果一次 WALK 没发现 7 个节点

冻结实现：

```text
max_steps = 80
max_retries = 20
```

- 一次最多走 80 steps；
- 未发现 7 个不同节点则从同一 root 重新开始；
- 20 次仍失败，明确抛出 `RuntimeError`；
- 不静默丢弃 patch，也不 padding。

### 4.1.2 如果整张图少于 7 个节点

直接抛出 `ValueError`。当前 raw IMDB-BINARY：

- 1000/1000 图连通；
- 最小图 12 nodes；

因此正式 IMDB 运行没有触发该边界。

### 4.1.3 每图选多少 roots

```text
patches_per_graph = min(n_nodes, 24)
```

- `n_nodes <= 24`：所有节点各作一次 root；
- `n_nodes > 24`：无放回随机选择 24 个不同 roots；
- sampling seed 固定为 `20260731`。

### 4.1.4 WALK 的“不变性”到底保证了什么

给定同一条抽象 WALK trajectory，将整张图重编号并同步映射 trajectory，first-discovery-order adjacency 完全相同。

它不保证：

> 对重编号后的图重新独立随机采样，逐条 patch 一定相同。

因此 WALK 表示的是 **walk-induced rooted patch distribution**，不是每个 rooted subgraph 唯一对应一个向量。

## 4.2 Direct capped n-hop：如何固定到 7 个节点

实现入口：

- `tracks/ksvd/code/imdb_nhop_sampler.py`
- `select_fixed_nhop_patch()`

先计算 root 到整张图所有节点的 shortest-path distance，然后取：

```text
root + 排名最前的 6 个非 root 节点
```

由于先按 distance 排序：root degree 不少于 6 时所选节点都直接连接 root；degree 少于 6 时全部 1-hop neighbors 会先被纳入，再补 2-hop 节点，因此当前 connected-graph 条件下所得 induced patch 保持连通。

三种 exploratory ranking：

```text
n_id:
    (distance, node_id)

n_degree:
    (distance, -global_degree, node_id)

n_signature:
    (distance,
     -common_neighbors_with_root,
     -global_degree,
     -sum_neighbor_degrees,
     node_id)
```

### 4.2.1 1-hop 少于 7 个节点

不 padding。distance 排序会继续从 2-hop 或更远节点补到 root+6。由于 raw IMDB 全部连通且最少 12 nodes，始终可以得到 7 个不同节点。

### 4.2.2 1-hop 多于 7 个节点

只取排名最前的 6 个 neighbors，即截断。若第六与第七候选的 invariant key 相同，则标记 `cutoff_tied=True`。

node ID 只是确定性 tie-break，并不被声称为 permutation invariant。

### 4.2.3 2-hop 坍缩为整图

在 raw IMDB 中，radius-2 ego 对 100% roots 等于整张图。因此完整 2-hop ego 没有被送入当前 KSVD：它已经不是 local fixed-size patch。

### 4.2.4 邻接排序

选定 7 个节点后保存两种 21-D vector：

1. ranking order adjacency；
2. exact rooted canonical adjacency。

n-hop signal audit 使用第 2 种，避免把 node-ID coordinate noise 当成 signal。

### 4.2.5 n-hop 的结构坍缩如何处理

当前没有用后处理“修复”坍缩，而是事先定义 substrate gate，测量：

- clique fraction；
- dominant topology mass；
- effective topology count；
- within-graph unique fraction；
- cutoff tie fraction；
- relabel exact-match rate。

结果 n-hop 明显比 WALK 更 clique-dominated，因此在训练 n-hop KSVD 之前停止。换言之：

> 坍缩是实验失败条件，不是靠增加 atoms/restarts 掩盖的问题。

---

## 5. 邻接矩阵与 permutation 问题

### 5.1 Exact rooted canonical adjacency

固定 root 为 slot 0，枚举其余 6 个节点的 `6!=720` 个排列，取 lexicographically minimum upper-triangle vector。

它严格保证：

> rooted-isomorphic 7-node graphs 得到相同 vector。

### 5.2 为什么“同构唯一”仍不够

KSVD 依赖 Euclidean distance、inner product 和线性组合。Canonical labeling 可能在一条边变化后切换整套节点排列，使多个 coordinates 同时改变。

U0-R 中，真实只 flip 一条边时：

- exact rooted graph edit distance 恒为 1；
- canonical vector mean Hamming change 约 `3.26`；
- `distance>1` amplification 约 `62%`；
- pairwise GED Spearman 只有 `0.58–0.64`。

因此 canonical adjacency 是优秀的“同一性编码”，但未必是优秀的线性字典坐标。

### 5.3 为什么多种排序不能保证不变性

BFS、degree、local signature 都可能 tie。拼接多种排序只会得到多个依赖 tie-break 的视图，并不会产生数学保证。

IMDB n-hop relabel audit：

| Selector | selected set match | ranked vector match | rooted canonical output match |
|---|---:|---:|---:|
| n-ID | 0.174 | 0.639 | 0.840 |
| n-degree | 0.317 | 0.961 | 0.970 |
| n-signature | 0.344 | 0.990 | 0.995 |

结构排序与 canonicalization 能让最终输出接近 invariant，但不能证明 cutoff 时选中的抽象节点集合严格一致。

---

## 6. 实验阶梯总览

| 阶段 | 数据/设置 | 唯一问题 | 结论 |
|---|---|---|---|
| E0 | 数值稀疏字典 | KSVD 数值实现能否恢复真字典 | PASS |
| E1 | 固定槽位图 atoms | 邻接 basis 在理想条件能否恢复 | T1 PASS；T2 初始化敏感 |
| E1-init | 50 初始化诊断 | 是目标错位还是 local optimum | 可管理 local optimum |
| E1B | singleton 降至 0.2 | atoms 很少单独出现时能否恢复 | 5-start 不稳定 |
| G0 | 四种 exact motifs | 不给 motif labels 能否恢复 vocabulary | maximin INIT 已枚举；只作 positive control |
| G0B | motif variants/noise | 固定 canonical edge-mask 是否可辨识 | FAIL |
| U0-R | unplanted rewired graphs | adjacency 坐标是否适合 KSVD | 选择 WALK，canonical 仅作 baseline |
| U0-P | 同上 | WALK patches 是否含结构信号 | PASS |
| U0-D | 同上 | 单 deterministic INIT KSVD 是否改善重建 | PASS |
| U1A | 同上 | graph codes 是否有信号/updates 是否增值 | 有信号；added value 未通过 |
| IMDB R0-P | raw IMDB | 真实 substrate/signal 是否可用 | s=7 PASS |
| IMDB R0-D | raw stratified/grouped | 真实 held-out reconstruction | PASS |
| IMDB R0-A | raw stratified/grouped | marginal graph-code incremental utility | FAIL |
| n-hop | raw IMDB | sampler 是否是主要问题 | direct capped n-hop FAIL |
| R0-B | raw IMDB | atom-pair readout 是否修复 | FAIL |
| R0-X | 复用 R0-D | objective 与 STATS 是否错位 | 优先诊断 conditional objective |
| R0-C | raw IMDB | residual objective 是否修复 | recon PASS / utility FAIL |
| R1-A/B | raw IMDB | basis 是否稳定、贴近真实 patches | continuous latent basis；非统一可命名 motifs |

---

## 7. 第一阶段：先验证 KSVD 本身

## 7.1 E0：纯数值 sparse recovery

### 问题

当数据确实由少量共享 atoms 稀疏生成时，代码实现能否恢复字典和 codes？

### 设计

- 已知 `D_true`；
- train/test patches 1000/300；
- `T=1` 与 `T=2`；
- 25 updates；
- 比较 atom cosine、support F1、test reconstruction。

### 结果

`T1/T2` 均达到：

```text
test reconstruction = 0
atom cosine = 1
support F1 = 1
10/10 strict recovery
```

### 结论

基础 KSVD、OMP、matching 和指标实现可工作。

## 7.2 E1：固定槽位图结构 basis

将 atoms 定义为固定 adjacency slots 上的结构基。

- `T=1`：10/10 strict recovery，PASS；
- `T=2`：5/10 strict recovery，mean atom cosine `0.884`，FAIL。

重要发现：patch reconstruction 接近完美时，真实 atoms 仍可能没有恢复。这第一次明确了：

> reconstruction 成功不等于生成结构可辨识。

## 7.3 初始化诊断的角色

E1-T2 使用 50 个 learner seeds 是为了诊断局部最优，不是最终训练协议：

- single-start strict success `29/50`；
- train error 与 atom recovery 强相关；
- 5-start group 中正确 basin 出现概率约 `0.991`，最低 train error 能正确选择。

随后跨 10 个 data seeds 的五启动确认达到 10/10。

但 singleton probability 降到 0.2 后，只在 8/10 data seeds 的五候选中出现正确 basin。这说明依赖多启动的 synthetic recovery 不适合直接外推到真实部署。

因此后续无人工词表和真实数据路线改为：

> **每 fold 一个 deterministic maximin INIT，restart=0。**

---

## 8. 人工 motif positive controls 及其边界

## 8.1 G0：四种 hidden motifs

训练数据只有四种 exact canonical columns。

- deterministic maximin INIT 已经 10/10 枚举四种 prototypes；
- FINAL 没有额外贡献；
- fixed random-column INIT 为 0/10，KSVD FINAL 为 10/10。

因此 G0 证明优化流程能从较弱初始化恢复离散 vocabulary，但 primary 结果只能标记为：

> `PASS_INITIALIZER_DISCOVERY_ONLY`

因为强初始化器已经完成“发现”。

## 8.2 G0B：加入 within-motif variation

加入一条 non-core edge 或随机 edge flip 后，不同 latent motif families 可以生成相同 observed canonical graph。

代表结果：

```text
cross-family collision mass = 0.477
Bayes accuracy ceiling = 0.892
worst robust fixed-core F1 = 0.591
```

最大度 structural root 和 generator oracle root 都无法让所有 motif families 同时获得稳定 fixed edge support。

结论：

> 对一般 noisy untyped patch，“一个 atom 对应 canonical adjacency 中固定 motif edge mask”的强命题不可辨识。

这推动路线放弃“每个 atom 必须是人工命名 motif”的要求。

---

## 9. 无人工 atom vocabulary：U0/U1

## 9.1 数据生成

使用 60-node degree-4 ring lattice，并做保持 degree sequence 和连通性的 double-edge swaps：

- LOW：20–40 accepted swaps；
- HIGH：60–80 accepted swaps。

模型不知道 LOW/HIGH label，也不知道任何 atom vocabulary。

## 9.2 U0-R：表示审计

Exact canonical adjacency 虽严格同构不变，但 one-edge perturbation 被放大，linear geometry gate 失败。

WALK first-discovery order：

- 一条固定 edge flip 只影响一个 coordinate；
- mapped-trajectory relabel invariance 为 1；
- 保留 sampler-slot 语义。

决定：`SELECT_WALK_ORDER_PENDING_U0P`。

## 9.3 U0-P：训练 KSVD 前先确认输入有信号

五个 independent data replicates：

| Feature | Mean test BA |
|---|---:|
| WALK mean/std | 0.815 |
| Canonical mean/std | 0.803 |
| Edge histogram | 0.826 |
| WALK label shuffle | 0.501 |

结论：`PASS_U0P_WALK_SIGNAL_EXPOSED`。

## 9.4 U0-D：单 deterministic INIT dictionary audit

配置：

```text
d=15, K=12, T=2, updates=25
one deterministic maximin INIT
restart=0
```

五个 replicates：

- held-out relative reconstruction reduction mean `0.3455`；
- `5/5` 为正且均超过 10%；
- 每次 `12/12` nondead；
- max activation share `0.230–0.294`；
- FINAL cross-replicate matched atom cosine `0.862`，高于 INIT `0.551`。

结论：`PASS_KSVD_OPTIMIZATION`。

这是第一项直接支持以下命题的证据：

> 不给人工 atom vocabulary、只用一次可部署初始化，KSVD 仍能学习健康且改善 unseen-patch reconstruction 的 basis。

## 9.5 U1A：下游 signal 与 added value

| Control | Mean BA |
|---|---:|
| Simple graph statistics | 0.935 |
| WALK raw | 0.815 |
| Edge histogram | 0.826 |
| PCA-12 | 0.809 |
| Gaussian code | 0.709 |
| INIT code | 0.713 |
| FINAL code | 0.750 |
| Code shuffle | 0.475 |
| Label shuffle | 0.491 |

FINAL codes 有真实信号，但：

```text
FINAL - INIT mean = +0.037
positive replicates = 3/5
registered requirement = 4/5
```

所以：

- graph-code signal：PASS；
- KSVD update added value：FAIL。

同时简单统计明显更强，说明 rewiring target 主要由低阶统计决定。

---

## 10. 迁移到完整 raw IMDB-BINARY

## 10.1 为什么不用 cleaned 替代 raw

Raw 数据审计：

```text
graphs = 1000, labels = 500/500
exact structure groups = 537
duplicate groups = 116
graphs in duplicate groups = 579
label-conflict groups = 44
graphs in conflict groups = 318
```

严格 isomorphism-invariant、structure-only deterministic classifier 在 raw 样本上的经验 accuracy ceiling 为 `0.886`，至少 114 个样本无法同时判对。

Cleaned 493 图恰好对应去除冲突结构并去重后的 label-consistent representatives。因此 cleaned 改变了任务，不可替代 raw benchmark。

冻结三种视图：

1. raw/stratified：外部参考；
2. raw/exact-isomorphism-grouped：同构组不跨 test/train；
3. cleaned/stratified：只作后续 sensitivity，本轮不用于主结论。

## 10.2 R0-P：真实 substrate

### s=6 首次迁移

- clique mass `0.442`；
- canonical effective topology count `9.763`；
- 原 gate 要求 `>=10`，形式失败。

没有事后把 9.763 改判为通过，而是只改变一个轴：`s=6 -> 7`。

### s=7 修复探针

```text
patch count = 17,526
clique mass = 0.3359
effective topology count = 18.929
within-graph unique median = 0.500
WALK BA = 0.619
canonical BA = 0.630
edge histogram BA = 0.648
STATS BA = 0.700
STATS+WALK BA = 0.696
label shuffle = 0.504
```

结论：substrate 与 standalone signal 足以进入字典审计，但 patch summaries 尚未补充 STATS。

## 10.3 R0-D：真实 dictionary optimization

冻结：

```text
s=7, d=21, K=12, T=2, T_min=1
updates=25
one deterministic maximin INIT/fold
restart=0
split seed=731301
```

两个 5-fold views：

| View | Positive folds | Mean INIT→FINAL graph-balanced reduction | FINAL test error |
|---|---:|---:|---:|
| raw/stratified | 5/5 | 0.4787 | 0.3657 |
| raw/grouped | 5/5 | 0.4789 | 0.3551 |

所有 folds：

- `12/12` nondead；
- max usage share `<0.29`；
- train/test reconstruction gap 约 0；
- FINAL cross-fold matched cosine `0.753/0.746`。

结论：`PASS_R0D_REAL_DICTIONARY_OPTIMIZATION`。

PCA-12 reconstruction 更低，是因为它使用全部 12 个 dense coefficients；KSVD 被限制为 `T=2`。PCA 是非稀疏 floor，不是 KSVD 必须击败的 gate。

---

## 11. IMDB 下游归因与连续修复

## 11.1 R0-A：36-D marginal readout

新 split seed `731401`，不复用 R0-P/R0-D test replication。主比较：

\[
BA(STATS+FINAL)-BA(STATS+INIT).
\]

| View | STATS | STATS+INIT | STATS+FINAL | Mean update | Mean beyond STATS |
|---|---:|---:|---:|---:|---:|
| stratified | 0.704 | 0.680 | 0.678 | -0.002 | -0.026 |
| grouped | 0.623 | 0.624 | 0.630 | +0.006 | +0.007 |

Stratified 只有 2/5 folds 为正。正确对齐 FINAL 在 stratified 平均还低于 shuffled FINAL：`0.678 vs 0.685`。Permutation implementation audit 排除了 shuffle 未真正执行。

结论：没有建立稳定 task utility。

## 11.2 Direct n-hop sampler audit

### Ego 大小

| Radius | <7 | =7 | >7 | Equals whole graph |
|---:|---:|---:|---:|---:|
| 1 | 0.316 | 0.138 | 0.546 | 0.183 |
| 2 | 0 | 0 | 1.000 | 1.000 |

### Substrate

| Sampler | Clique mass | Effective topology count | Unique median |
|---|---:|---:|---:|
| WALK | 0.336 | 18.929 | 0.500 |
| n-ID | 0.559 | 6.755 | 0.214 |
| n-degree | 0.622 | 4.828 | 0.167 |
| n-signature | 0.642 | 4.700 | 0.154 |

n-hop standalone BA 最高 `0.647`，只比 WALK canonical 高 `0.017`，且 `STATS+n-hop` 仍低于 STATS。

结论：`FAIL_DIRECT_CAPPED_NHOP_SAMPLER`。当前失败不能简化为“随机游走采样不好”。

## 11.3 R0-B：within-patch atom-pair readout

只在 36-D marginal 后加入 `C(12,2)=66` 个同 patch support co-activation features；新 outer seed `731501`。

| View | Pair update mean | Positive direction | Pair added over marginal FINAL |
|---|---:|---:|---:|
| stratified | +0.015 | 3/5 | -0.018 |
| grouped | +0.001 | 2/5 | -0.021 |

Train score 上升但 test score 下降，说明更多 bag statistics 主要增加拟合容量。

结论：`FAIL_R0B_PAIR_READOUT_UTILITY`。

## 11.4 R0-X：objective-alignment diagnosis

复用 R0-D frozen dictionaries，不重新训练。

主要发现：

- STATS 对 reconstruction gain 的 held-out explained fraction 为 `0.737/0.672`；
- STATS 对 FINAL codes 的 explained fraction高于 INIT；
- 控制 STATS 后 FINAL−INIT label projection 方向不稳定；
- reconstruction gain 与 patch edge-count frequency 高相关。

解释：KSVD update 主要增强了与全局统计共同变化的 reconstruction directions，而不是稳定的 label-residual direction。

路由：`PRIORITIZE_STATS_CONDITIONAL_OBJECTIVE`。

## 11.5 R0-C：statistics-conditioned residual objective

只修改无监督 dictionary target：

1. outer-train graph statistics 预测每图 patch mean；
2. 原始 patches 减去该预测；
3. 对 residual patches 学 KSVD；
4. 不使用 labels。

### Reconstruction

| View | Positive folds | Mean residual reconstruction reduction | Nondead |
|---|---:|---:|---:|
| stratified | 5/5 | 0.362 | 12/12 |
| grouped | 5/5 | 0.372 | 12/12 |

### Task utility

| View | Residual FINAL−INIT | Direction | Residual FINAL−STATS | Residual vs standard FINAL |
|---|---:|---:|---:|---:|
| stratified | -0.016 | 2/5 | -0.023 | +0.011 |
| grouped | +0.036 | 3/5 | +0.011 | -0.015 |

Residual basis 学习成功，但 task utility 仍不稳定。

结论：`FAIL_R0C_STATS_CONDITIONAL_UTILITY`。按照冻结停止规则，普通无监督 reconstruction-KSVD 的 raw IMDB task branch 到此结束。

---

## 12. Basis-discovery/compressor 分支

## 12.1 R1-A：label-free characterization

复用 R0-D dictionaries，不重新训练。结果：

```text
nondead = 12/12
FINAL cross-fold matched cosine = 0.753 / 0.746
INIT cross-fold matched cosine = 0.536 / 0.561
FINAL nearest-real cosine = 0.887 / 0.882
Gaussian nearest-real cosine = 0.636 / 0.638
```

正式 gate 仍标记 FAIL，因为预注册条件要求 FINAL nearest-real 不低于 INIT；但 INIT atoms 本身就是归一化真实 training patches，nearest-real cosine 按构造恒为 1。该条件结构性失配，不能事后删除并改判 PASS。

因此 R1-A 应解释为：

> protocol-misspecified / inconclusive，而不是所有 basis evidence 失败。

## 12.2 R1-B：grouped consensus basis atlas

无 PASS/FAIL gate，只构造描述性 atlas：

- 12 个 latent components；
- reference-matched cosine mean `0.717`；
- worst atom/fold match `0.102`；
- 每 atom 25 个 exemplars 中平均 `9.17` 个 unique canonical signatures；
- dominant canonical mass mean `0.327`。

这支持：

> learned atoms 是与多种真实 patches 对齐的 continuous latent basis components。

但不支持：

> 12 个 atoms 全部稳定对应 12 个清晰、可命名 graph motifs。

---

## 13. 数据隔离、分类器与 controls

## 13.1 Outer folds

真实 IMDB 使用：

- raw stratified 5-fold；
- raw exact-isomorphism-grouped 5-fold。

Grouped view 保持含冲突 labels 的整个同构组完整，group leakage 为 0。

## 13.2 Inner validation

Outer train 再固定划分 inner train/validation：

- stratified view 按 label stratified；
- grouped view 继续保持 exact-isomorphism groups 完整；
- 所有 feature controls 共用同一 inner split。

## 13.3 Classifier

使用 package-independent L2 logistic regression：

```text
lambda grid = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
standardization = inner train only
selection = validation balanced accuracy
tie = choose larger lambda
outer test = evaluate once
```

没有搜索神经网络深度、dropout、hidden dimension 或 learning rate。

## 13.4 Controls

- `STATS`：12-D simple graph statistics；
- raw WALK mean/std；
- edge-count histogram；
- fixed Gaussian dictionary；
- real-patch medoid；
- PCA-12；
- INIT codes；
- graph-code shuffle；
- label shuffle。

主要归因始终保留：

```text
FINAL vs INIT
STATS+FINAL vs STATS
correctly aligned code vs shuffled code
```

---

## 14. 当前已经证明和没有证明的内容

## 14.1 已有证据支持

1. KSVD/OMP 基础实现能恢复已知 sparse dictionary；
2. 单 deterministic maximin INIT 不依赖随机 restart selection；
3. WALK-order adjacency 在固定 sampler-slot 语义下具有可用结构信号；
4. 无人工 atom vocabulary 的 KSVD 能形成非坍缩 basis；
5. FINAL 在 synthetic unplanted 和 raw IMDB 上都稳定改善 unseen-patch reconstruction；
6. 该 reconstruction gain 在 exact-isomorphism-grouped view 仍成立；
7. FINAL basis 整体具有跨 fold 对齐性，且比 Gaussian 更接近真实 patches。

## 14.2 当前没有证据支持

1. 每个 learned atom 都是合法 adjacency；
2. 每个 atom 都有唯一、可命名的人类 motif 语义；
3. reconstruction improvement 必然产生分类 improvement；
4. FINAL graph codes 稳定超过 INIT；
5. FINAL codes 稳定补充 simple graph statistics；
6. direct capped n-hop 优于 WALK；
7. within-patch pair statistics 足以解决 patch relation；
8. statistics-conditioned residual reconstruction 足以产生 label-residual utility；
9. cleaned IMDB 分数可以代替 raw benchmark；
10. 当前路线达到或超过公开 SOTA。

---

## 15. 当前瓶颈

### 15.1 不是 optimizer 瓶颈

R0-D 和 R0-C 都显示：

- 5/5 held-out reconstruction improvement；
- 12/12 nondead；
- train/test gap 很小。

因此不能把失败归因于 KSVD 没有收敛或需要更多 restarts。

### 15.2 IMDB 的局部尺度不适配

- 1-hop 高度 clique-dominated；
- 2-hop 等于整图；
- fixed small adjacency patch 必须截断；
- selector ties 导致严格 invariance 困难。

### 15.3 Objective 与 task 错位

无监督 KSVD 优先改善高频、能量大的 patch variation。它不使用 labels，因此没有理由保证这些方向是分类最重要方向。

### 15.4 Patch 到 graph 的关系不足

Marginal readout 和同 patch atom pairs 都没有表达 occurrence 在原图中的空间关系。但继续加入 relation graph、MIL 或 attention 会明显扩张方法，并稀释 KSVD 的独立贡献。

---

## 16. 建议的路线定位

### 16.1 已成立的核心定位

> **KSVD 是一个无人工 motif vocabulary 的 sparse graph-patch compressor / continuous basis learner。**

后续可继续研究：

- compression/reconstruction trade-off；
- cross-dataset basis transfer；
- OOD patch reconstruction；
- atom usage 与真实 exemplar atlas；
- typed/multi-view patch signals；
- 计算和存储成本。

### 16.2 如果分类是必须目标

需要明确进入新的研究命题：

- label-conditioned/discriminative dictionary learning；或
- sparse basis + explicit occurrence relation model。

此时普通 KSVD 应被定位为初始化器、regularizer 或 compressor，而不能把全部下游收益归因于无监督 atom discovery。

### 16.3 不建议继续的操作

- 增加 restart 后挑最好；
- 在已看过的 outer tests 上扫描 K/T/iterations；
- 继续堆 bag-level statistics；
- 用 cleaned 掩盖 raw 结果；
- threshold continuous atoms 后直接命名 motif；
- 同时修改 sampler、objective、readout 和 classifier。

---

## 17. 代码结构与复现入口

### 17.1 核心实现

| 文件 | 作用 |
|---|---|
| `tracks/ksvd/code/ksvd.py` | KSVD/OMP 核心实现；本轮不覆盖用户已有修改 |
| `from_scratch_unplanted_representation.py` | 图生成、WALK patch、canonicalization、表示审计 |
| `from_scratch_unplanted_signal.py` | 图统计、标准化、L2 logistic、balanced accuracy |
| `from_scratch_unplanted_dictionary.py` | deterministic maximin、dictionary health/stability |
| `from_scratch_unplanted_downstream.py` | 基础 graph-code readout 与 controls |
| `imdb_walk_substrate.py` | TU raw loader、isomorphism groups、IMDB WALK cache |
| `imdb_walk_dictionary.py` | stratified/grouped folds、fold-local dictionary audit |
| `imdb_walk_downstream.py` | variable-patch graph readout、inner split、R0-A attribution |
| `imdb_nhop_sampler.py` | fixed-7 n-hop selector、ego feasibility、relabel audit |
| `imdb_walk_basis_characterization.py` | label-free basis metrics 与 exemplar characterization |

### 17.2 主要 runners

```text
run_from_scratch_e0_e1.py
run_from_scratch_e1_t2_initialization_audit.py
run_from_scratch_e1_t2_multidata_confirmation.py
run_from_scratch_e1b_s20.py
run_from_scratch_g0_hidden_motif.py
run_from_scratch_g0b_representation_gate.py
run_from_scratch_u0r_adjacency_audit.py
run_from_scratch_u0p_signal_exposure.py
run_from_scratch_u0d_dictionary_audit.py
run_from_scratch_u1a_graph_code_signal.py
run_imdb_binary_r0p_audit.py
run_imdb_binary_r0d_dictionary_audit.py
run_imdb_binary_r0a_downstream_attribution.py
run_imdb_binary_nhop_sampler_audit.py
run_imdb_binary_r0b_pair_readout_audit.py
run_imdb_binary_r0x_alignment_diagnosis.py
run_imdb_binary_r0c_stats_conditional_audit.py
run_imdb_binary_r1a_basis_characterization.py
run_imdb_binary_r1b_consensus_basis_atlas.py
```

### 17.3 主要协议与结果

总状态：

- `tracks/ksvd/docs/KSVD_FROM_SCRATCH_ROUTE_STATUS_20260731.md`
- `tracks/ksvd/docs/KSVD_ROUTE_BOTTLENECK_DISCUSSION_20260801.md`

无人工词表路线：

- `tracks/ksvd/docs/KSVD_U0_U1_NO_PLANTED_ATOM_ROUTE_20260731.md`
- `tracks/ksvd/results/from_scratch/U0R_ADJACENCY_AUDIT_20260731.md`
- `tracks/ksvd/results/from_scratch/U0P_SIGNAL_EXPOSURE_20260731.md`
- `tracks/ksvd/results/from_scratch/U0D_DICTIONARY_AUDIT_20260731.md`
- `tracks/ksvd/results/from_scratch/U1A_GRAPH_CODE_SIGNAL_20260731.md`

IMDB 路线：

- `tracks/ksvd/docs/KSVD_IMDB_BINARY_RAW_TRANSFER_ROUTE_20260731.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0P_WALK_S7_PROBE_20260731.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0D_DICTIONARY_AUDIT_20260731.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0A_DOWNSTREAM_ATTRIBUTION_20260731.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_NHOP_SAMPLER_AUDIT_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0B_PAIR_READOUT_AUDIT_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0X_ALIGNMENT_DIAGNOSIS_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0C_STATS_CONDITIONAL_AUDIT_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R1A_BASIS_CHARACTERIZATION_20260801.md`
- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R1B_CONSENSUS_BASIS_ATLAS_20260801.md`

---

## 18. 最终结论

本轮从零探索没有得到“KSVD 可以自动发现一组 task-optimal、可命名图 motifs”的强结论，但建立了一个更可靠的分层答案：

1. **算法层**：KSVD 实现和 sparse recovery 可工作；
2. **表示层**：WALK-order adjacency 比 exact canonical adjacency 更适合当前线性 geometry，但它表示 walk-induced signals，而不是唯一 graphlets；
3. **字典层**：无人工 atom vocabulary、单 deterministic INIT 的 KSVD 能在 synthetic unplanted 和 raw IMDB 上学习健康、稳定、改善 unseen reconstruction 的 continuous basis；
4. **任务层**：standard marginal、atom-pair 和 statistics-conditioned residual routes 都没有建立稳定的 raw IMDB downstream added value；
5. **解释层**：learned components 可以由真实 exemplar families 描述，但不能要求全部对应清晰、稳定、可命名 motifs。

最准确的一句话是：

> **KSVD 能够自发现有统计意义的 sparse graph-patch reconstruction basis；但“自发现 reconstruction basis”不等于“自发现对图分类最有用的结构原子”。**

## 19. 补充：导师式 per-graph dictionary 是否能绕开瓶颈

在回顾 Phase 2/3 与导师脚本后，另开了严格独立的 G0 支线：每张 IMDB 图单独拟合 `K=8,T=2,updates=10` 的字典，并使用 atom-permutation-invariant dictionary/code/spectral readout。该实验仍固定当前 WALK `s=7` substrate，避免同时改变 sampler、动态 patch size 与 dictionary scope。

结果：

- invariant readout permutation error `1.24e-14`；
- mean reconstruction `INIT 0.1370 -> FINAL 0.0935`，但 PCA 为 `0.0663`；
- stratified `STATS+RAW+INIT/FINAL/PCA = 0.693/0.681/0.689`；
- grouped `STATS+RAW+INIT/FINAL/PCA = 0.675/0.667/0.627`；
- FINAL−INIT 分别为 `-0.012`（2/5 positive）和 `-0.008`（1/5 positive）；
- legacy ordered readout 对 atom permutation 有可见敏感性。

正式分类：`RECON_ONLY_PERGRAPH_DICTIONARY`。

这表明 per-graph KSVD 可以作为图内 patch cloud 的 sparse factorization，但当前可预测信息主要已经存在于 raw distribution、simple statistics 或 deterministic INIT 中。把字典本身 readout 并没有使 KSVD updates 获得稳定的 downstream 归因。
