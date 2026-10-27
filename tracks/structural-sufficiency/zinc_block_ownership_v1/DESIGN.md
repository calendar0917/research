# DESIGN — `zinc_block_ownership_v1`（物理键的真实块归属）

**日期：** 2026-10-27
**状态：** 设计 + Phase 0 CPU 研究门控；本文件在正式 GPU 训练之前提交。
**边界：** official test 全程不构造、不加载、不预测、不评分。official valid 只用于
每 epoch 评估与严格最小选点，不进入特征、归一化、SSL 或结构统计。

---

## 0. 一句话

在 BOSI-B 的共享局部编码（atom / bond / half-edge / side / compose）之上，增加一条
**非递归、无 learned junction token、无 attention** 的块级通道：把每条物理键的
`c_e` 按真实 biconnected component（桥边为单键块）归属到唯一块，先做共享块级非线性
`phi(u_B)`，再对所有块求和，作为 32 维结构组合通道加入原 BOSI reader。用严格匹配的
**R（真实归属）/ N（全图均场）/ S（保留单键块真实内容，多键块均场）** 三臂，
把“真实块归属是否提供原有汇总看不到的信息，并在标签预测中有益”与“块级非线性是否
提高组合泛化效率”分开检验。

本轮优先回答 A/B，不预设原理成立；只有 B 成立后才讨论 C。

---

## 1. 研究命题与三层区分

- **A（信息）**：真实 biconnected 块归属是否提供了 BOSI 现有全局汇总
  `[Σc, Σc², Σa, Σa², global33]` 与 BOSI v0 的 per-half-edge 关系分组（T/Q/R/N）
  看不到的信息？
- **B（预测价值）**：在参数量、输入接口、初始化、训练配方完全相同的条件下，
  这种信息是否改善未见局部环境/未见块组合上的标签预测？
- **C（组合泛化效率）**：共享块级非线性 `phi` 本身是否比等参数量的逐块独立表达
  更有效率？**本轮不做 C 的独立控制**；R/N/S 三臂都使用同一个 `phi`，因此
  R−N 或 R−S 的正收益不能直接归因于“非线性共享”，只能归因于归属/内容。

必须区分的事实：

1. R 与 N 的输入总量（一阶、二阶键统计）在归一化前完全相同；两者只差“内容如何
   分到块”。因此 R−N 不是“更多统计量”或“更多参数”的对照。
2. N 丢失真实归属信息，因此 R−N **不是严格的同信息归纳偏置实验**。本报告会明确
   这一点，不把 R−N 包装成“纯归纳偏置”结论。
3. S 保留所有单键桥块的真实内容，只对多键块使用块间守恒的均场分配。R−S 才把
   差异集中到“多个非平凡块之间的真实内容归属”；S−N 则主要度量单键块上的逐键
   内容/非线性是否已经足够。
4. 不允许把“R 优于 N、但不优于 S”解释为块级组合泛化得到支持。那最多是
   `SINGLETON_NONLINEARITY_ONLY`。

---

## 2. 与历史工作的关系（必须公开）

### 2.1 BOSI v0 / partner-content v1

- `zinc_bosi_architecture_v0` 的 R/B 对照检验的是 **per-half-edge 伙伴矩如何分配到
  T/Q/R/N 四个关系组**。其分配只在 19.50% 的半键上有定义，Locked 上 R−B 为
  −0.00206 / +0.00155（符号翻转），判决 `NO_GO`。本轮不是它的重跑：本轮的块是
  **全局 biconnected component**，每个物理键归属唯一块；块级汇总包含不在同一中心
  原子上的键；R/S 的结构覆盖率为分子级 88.3%（Phase 0 实测），不是 19.5%。
- `zinc_bosi_partner_content_v1` 的 P/B 对照证明“其他邻键的 learned 内容进入矩块”
  有价值（+0.0124，两 seed）。本轮不重复这个结论；本轮固定使用 BOSI-B 的
  `M`（包含伙伴内容），只问 **这些内容按真实块归属聚合是否比全图均场更有预测价值**。
- `zinc_adaptive_arch_search_v1/ERRATA.md` E4 证明 BOSI-B 的 384 维矩块在代数上是
  `(96 维总矩, 4 维计数)` 的广播，不是 384 个独立统计量。本轮不再往这个 384 槽位
  里加东西，而是新增一个 **块划分上的 Deep Sets 通道**；它与矩 bank 的代数对象不同。

### 2.2 AAS ERRATA E3：方向 C 未被检验

E3 指出：S1/S2 的差异同时混合了槽位语义、局部性、归一化和计数信息，因此
“图级池化”本身没有被检验；要检验它，需要图级/块级统计量、保留局部通道、
等参数等初值的 matched 对照。本轮的 R/N/S 正是为此设计的一个**受限版本**：
块级统计量来自真实块对象，局部 BOSI 通道全部保留，三臂等参数、同初值。
但本轮只检验“块归属”这一条，不外推到所有图级池化。

### 2.3 旧 BlockModel（`zinc_arch_discovery_v1` 的 B 臂）

旧 B 是**另一个模型**，不是本轮的前身：

| 维度 | 旧 BlockModel | 本轮 |
|---|---|---|
| 块特征 | 693 维 atom-pair×bond joint histogram + 21 atom + 3 bond + 11 scalars | 每条物理键的 learned `c_e` 的块内一阶/二阶矩（2×96）+ 4 个结构标量 |
| 块间计算 | 1 层 global attention，带 block-cut-tree distance / overlap bias | **无 attention**；共享 `phi` 后直接求和 |
| 读出 | 独立的块 token 读出（sum/mean/2nd moment + composition + global33） | 32 维通道追加到**原有 BOSI reader**，旧 reader 参数逐位保留 |
| 对照 | 无 | R/N/S 三臂严格匹配 |
| 参数 | 338,805 | 497,161（Phase 0 实测） |
| valid | 0.165418（seed 0，远差于 P3 0.0791） | 本轮待测 |

旧 B 的 `block_joint`、`block_atom_type`、`block_bond_type`、`block_dist`、
`block_overlap`、`GlobalAttention` **一条都没有进入本轮 forward**。本轮复用的只是
“biconnected component + 桥边”的确定性划分，并且会在 CPU gate 中与 NetworkX 独立
参考实现、以及一个独立 Tarjan 实现三方核对。

### 2.4 其他已失败的近邻

- `zinc_rooted_branch_operator_v0`、`zinc_structural_conditional_moments_v1`：
  根内无条件矩/旁路修正，不是块归属。
- `zinc_branching_fragment_v1`：rooted branching fragment 的 198 维统计，零初始化
  首层权重、早停；对象是 root 邻域分叉模式，不是全局块，且没有 R/N/S 归属对照。
- `zinc_hierarchical_relation_v1`：atom→bond→junction→patch 层级 + support gate +
  一次低秩关系交互；不是块级内容归属，且其 support gate 在训练后塌缩。
- `zinc_local_factorization_v1`、`zinc_shared_factor_v1`、`zinc_exact_structure_residual_v1`：
  低参数加性/因子化读出、精确结构残差；它们改变的是读出或残差，本轮改变的是
  **结构内容的聚合划分**，且不把块通道作为主预测器（零初始化 reader 列）。
- `zinc_support_weighted_objective_v1`：标签加权；本轮不改 loss，只改结构输入。
- `zinc_dccl_bottleneck_v1`：DCC-L 的误差与 radius-2 支撑剂量成正比，低支持是
  方差受限；本轮的块通道是一种跨块共享统计的候选，但本轮不声称能解决低支持，
  只按预注册分组报告。

### 2.5 性能参照

- DCC-L（B-R0，494,193 参数）4 seed official valid：**0.075990 ± 0.002137**；
  one-shot test 0.067319 ± 0.000709。历史最好单模型 test 0.066267。
- BOSI-B 在 official 10k/1k 协议上：**0.082405**（`zinc_bosi_joint_moment_v1`，
  473,889 参数，seed 0）。BOSI 主干在官方协议上并不与 DCC-L 打平，因此本轮
  “R 的绝对 best-valid ≤ 0.082” 是一个**高于 BOSI-B 历史点估计**的门槛，不是
  自动可达的。若 R 达不到，必须如实报告，不得用“相对 N 有改善”掩盖。

---

## 3. 对象：真实块归属

对每个分子，用原始无向物理键图构造 **biconnected components**：

- 桥边是单键块（`n_edges=1, n_atoms=2, cyclomatic_rank=0`）；
- 每个非桥 biconnected component 是一个多键块（环系/双连通子图）；
- 每个无向物理键**必须且只能**归属一个块；
- 孤立原子不属于任何键块（ZINC subset 中实测没有孤立原子；若出现，块通道忽略
  无键块，不构造虚假内容）。

对块 `B`：

```
n_edges(B)        = |E(B)|
n_atoms(B)        = |V(B)|
cyclomatic_rank   = n_edges - n_atoms + 1          # 桥块为 0
n_articulations   = |{a in V(B): a 是原分子的 articulation point}|
```

`n_articulations` 使用**原分子**的 articulation points（不是块子图内部的割点；
双连通块内部没有割点，桥块则看两端是否把分子切开）。这是 label-free 结构量。

块的确定性构造以独立 Tarjan 边栈算法为主实现，`networkx.biconnected_component_edges`
为独立参考；两者在每个分子上必须给出相同的边划分（作为集合比较）。旧 BlockModel
的 `block_*` 缓存只用于三方交叉核对，不进入 forward。

---

## 4. 模型

### 4.1 共享局部基础（不改）

完全复用 BOSI-B 的 atom / bond / half-edge / side / compose：

```
a_v      = atom_encoder(atom_feat_v)                         108 -> 128 -> 96
b_e      = bond_encoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bf_e])  295 -> 160 -> 96
x[v,e]   = tanh(half_encoder([a_v,a_u,b_e,bf_e]))             295 -> 128 -> 32
M,n      = BOSI-B 的 per-half-edge 伙伴统计（arm="B"）
s[v,e]   = side([a_v,b_e,x,M,n,closure,log1p(deg)])           617 -> 128 -> 96
c_e      = compose([b_e,s[u,e]+s[v,e],|s[u,e]-s[v,e]|,
                    s[u,e]*s[v,e],bf_e])                      391 -> 160 -> 96
z_old    = [Σ_e c_e, Σ_e c_e², Σ_v a_v, Σ_v a_v², global33]   417
```

`c_e` 与 `z_old` 的公式、形状、参数全部保持不变。`M` 始终走 BOSI-B（不是 R 的
per-relation 分配），因此本轮的 R/N/S 与 BOSI v0 的 R/B 不是同一开关。

### 4.2 块级通道（新增）

对每个块 `B`：

```
u_B = [ mean_{e in B} c_e , mean_{e in B} c_e² ,
        log1p(n_edges), log1p(n_atoms), cyclomatic_rank, log1p(n_articulations) ]
      ∈ R^{2*96+4} = R^196
phi(u_B) = Linear(64<-196) -> SiLU -> Linear(32<-64)           # 共享，per-block LayerNorm(196)
h_block[G] = Σ_{B in G} phi(u_B)                               # 32 维
z_new = h_block[G]
yhat = reader_ext(z_old, z_new)                                # 449 维输入的等价拆分
```

- 归一化 `mean = content / n_edges` 是三个臂**完全相同的公式**；CPU gate 另行证明
  归一化前的 `Σ_B content_first` 与 `Σ_B content_second` 在 R/N/S 之间逐分子守恒。
- 每块使用同一个 `LayerNorm(196)`（affine），这是三臂共享的。
- 无 attention、无块间消息传递、无 block-cut-tree 距离、无块类型 embedding。
- `phi` 是共享的，因此“块级非线性”本身在所有臂中都存在；R/N/S 只切换内容归属。

### 4.3 reader 的零初始化扩展

先按历史顺序构造完整 BOSI-B（六个 MLP，RNG 流与历史 `BOSI(arm="B")` 相同），
再添加块编码器与 32 个新 reader 输入列。实现上，新列用一个
`Linear(256<-32, bias=False)` 与旧 `reader.lins[0]` 相加：

```
h1 = reader.lins[0](z_old) + reader_ext(z_new)
h2 = SiLU(h1); h3 = reader.lins[1](h2); h4 = SiLU(h3); y = reader.lins[2](h4)
```

这与“把 reader 第一层从 417 扩到 449、新列权重为 0”在数学上完全等价，但能保证
初始化时 `reader_ext(z_new) ≡ 0` 的逐位相等（不依赖 BLAS 对 449 维归约的顺序）。
`reader_ext.weight` 零初始化；旧 reader 的全部权重/偏置逐位保留。

参数（`named_parameters()` 实测，Phase 0 会重算）：

| 模块 | 参数 |
|---|---:|
| BOSI-B 原有 | 473,889 |
| block LayerNorm(196, affine) | 392 |
| block MLP 196→64→32 | 14,688 |
| reader_ext 32→256（无 bias） | 8,192 |
| **合计** | **497,161**（≤500,000） |

### 4.4 初始化纪律

- 同一 seed：先 `torch.manual_seed(seed)`，构造完整 BOSI-B，再构造 block norm/MLP，
  最后零初始化 `reader_ext`；然后只从 train-only masked-SSL checkpoint 加载
  `atom_encoder` 与 `bond_encoder`。
- 三臂 R/N/S 的完整初始 state（含块编码器与 reader_ext）逐张量相同。
- 与同 seed 的原始 BOSI-B 相比：所有公共参数逐张量相同；在真实 batch 上
  `max |y_new - y_ref| == 0.0`（零初始化 reader_ext）。
- 不加载任何监督训练过的 BOSI/DCC checkpoint；不加载 junction encoder。

---

## 5. R / N / S 三臂

记 `first_B = Σ_{e in B} c_e`，`second_B = Σ_{e in B} c_e²`。

### R — Real ownership

```
content_first_B  = first_B
content_second_B = second_B
```

按真实块收集每条物理键的 `c_e`，先算 `u_B` 与共享 `phi(u_B)`，再 `Σ_B`。

### N — Mean-field null

保留块个数、每块 `n_edges/n_atoms/rank/n_articulations`、原始 BOSI 通道与同一个
`phi`；把块内内容替换为按块边数分配的全图键统计均场：

```
S1[G] = Σ_{e in G} c_e
S2[G] = Σ_{e in G} c_e²
E[G]  = Σ_{e in G} 1 = 分子 G 的键数
content_first_B  = (n_edges(B) / E[G]) * S1[G]
content_second_B = (n_edges(B) / E[G]) * S2[G]
```

于是 `Σ_B content_first_B = S1[G]`、`Σ_B content_second_B = S2[G]`，与 R 在归一化前
逐分子相同。N 丢失真实归属，但保留全部结构标量与总统计量。

### S — Singleton-preserving null（条件性）

保留所有单键桥块的真实内容；只对多键块做**在这些多键块之间**守恒的均场分配：

```
MULTI[G] = {B in G : n_edges(B) > 1}
E_multi[G] = Σ_{B in MULTI[G]} n_edges(B)
S1_multi[G] = Σ_{B in MULTI[G]} first_B
S2_multi[G] = Σ_{B in MULTI[G]} second_B

B 单键: content_first_B = first_B,          content_second_B = second_B
B 多键: content_first_B  = (n_edges(B)/E_multi[G]) * S1_multi[G]
        content_second_B = (n_edges(B)/E_multi[G]) * S2_multi[G]
```

于是 `Σ_B content_first_B` 与 `Σ_B content_second_B` 仍与 R 逐分子相同。
R−S 只差“多个非平凡块之间的真实内容归属”；S−N 主要差“单键块上的真实逐键内容”。

---

## 6. Phase 0：CPU-only 研究门控（先不占 GPU）

使用官方 train 10k 结构构造块缓存，完成：

1. **块划分正确性**：独立 Tarjan 与 NetworkX 参考逐分子一致；每条真实键出现一次；
   所有键完整覆盖；旧 BlockModel 的块缓存作为第三方交叉核对。
2. **不变性**：原子重编号、边反向、边顺序、batch 重组下块特征/前向不变。
3. **总量守恒**：R/N/S 的归一化前 `Σ_B content_first` 与 `Σ_B content_second`
   逐分子相等；N/S 的分配比例和为 1；S 的单键块内容精确等于 R。
4. **严格合成反例**：相同键状态多重集、相同块大小，但真实块归属不同；R 可区分，
   N 不可区分（算子级；不是化学标签证明）。
5. **真实统计**：单键块/多键块比例、块尺寸分布、含 ≥2 个非平凡块的分子比例、
   R/S 输入差异的结构覆盖率与数值覆盖率。
6. **与旧 BlockModel 的区分**：静态代码审计 + 特征/参数对照，证明没有复用旧
   `block_joint` / `block_dist` / `block_overlap` / `GlobalAttention`。
7. **工程门控**：三臂梯度、初始化哈希、标签隔离、数值稳定性、参数量、前向形状、
   checkpoint 保存/加载、缓存可复现。

### 6.1 GPU 停止规则（Phase 0 之后）

若以下任一成立，**停止**，不进入 GPU，不通过新增对象/超参数/加宽来救援：

- 官方 train 10k 中含 **≥2 个非平凡块（n_edges≥2）** 的分子比例 **< 20%**；
- R/S 的真实输入差异极弱：在含 ≥2 个多键块的分子中，多键块的 R−S 相对内容差
  `< 1%` 的比例超过 50%，或平均相对差 `< 1%`；
- 块缓存无法与 NetworkX/Tarjan 对齐，或三臂初始化不逐位相同，或参数 >500k；
- 机制在训练第二步后仍没有梯度（此时负结果不能作为科学否定，只能报告
  `MECHANISM_INACTIVE`）。

Phase 0 实测（本文件提交后写入 `analysis/cpu_checks.json`）：train 10k 中
98,286 个单键块 / 23,522 个多键块；**88.32%** 的分子含 ≥2 个多键块；
块尺寸（边数）分布见 REPORT。该结构覆盖率远高于 20% 门槛。

---

## 7. Phase 1：匹配实验（通过 Phase 0 后）

### 7.1 统一协议

- 官方 PyG ZINC `subset=True`；train 10k、valid 1k；raw-y MAE；
  **official test 不加载、不预测、不评分**。
- 600 epochs，47,400 updates，batch 128（79 steps/epoch）。
- 与 BOSI/DCC R0 相同的 AdamW、warmup 500、per-step cosine 1e-3→1e-5、
  weight decay 1e-5、grad clip 5、dropout 0、FP32、L1 raw-y。
- full valid 每 epoch 评估，strict minimum 选 best（平局保留更早 epoch）。
- 同 seed 的 R/N/S 使用相同 SSL、相同 batch 顺序、全部公共初始张量逐位相同。
- 不改标签、不加额外训练数据、不做 label decomposition / RDKit 性质 oracle、
  不做蒸馏、ensemble、TTA、test 选择。
- 不加载任何监督 BOSI/DCC checkpoint；只加载 train-only masked-SSL atom/bond
  编码器（`pretrain_encoder_s{seed}.pt`；本轮的 seed 0/1 都有独立 SSL 文件）。

### 7.2 批次与停止规则

**第一批（2 GPU 并行）：** `R_seed0`（res-gpu0）与 `N_seed0`（res-gpu1）。
正式部署前 commit、`rr doctor res`、`rr deploy res`；用 `rr run`，不手工 SSH
后台作业。先做少量更新的 GPU smoke，核对初值、形状、梯度、显存。

seed-0 判决（预注册，看完 seed 0 不得改设计）：

```
d_best = N_best_valid - R_best_valid          # 正 = R 更好
d_W20  = N_W20        - R_W20
d_T100 = N_T100       - R_T100

PASS  iff
  d_best >= 0.005
  and R_best_valid <= 0.082
  and d_W20 >= -0.002 and d_T100 >= -0.002
  and at least one of (d_W20, d_T100) > 0
  and ordinary (y > -3) paired ΔMAE >= 0
  and fixed-worst-10-removed paired ΔMAE > 0
  and mechanism_active
```

`mechanism_active` 的定义（在 best checkpoint 上）：
- step 2 的 block encoder 梯度范数 > 1e-8（数值非零）；
- `h_block` 在 valid 上的标准差 > 1e-6；
- `reader_ext.weight` 的 Frobenius 范数 > 1e-6；
- 把 R 权重切换到 N 内容后，valid 预测不是逐位不变（mean|Δpred| > 1e-6）。

若失败：**停止**，不调宽度、层数、学习率、归一化、块类型；不买 seed 1。

**第二批（条件）：** `R_seed1` / `N_seed1`（2 GPU 并行）。两 seed 的 `d_best`
都 ≥0.005 且方向一致，才进入第三批。

**第三批（条件）：** `S_seed0` / `S_seed1`（2 GPU 并行）。重点看 R 是否稳定优于 S，
而不是只优于 N。

### 7.3 预算

- 软预算 **4 A100 GPU-hours**，硬上限 **6**。含 smoke、失败作业与正式训练。
- 单 run 预计 0.45–0.55 GPU-h（BOSI-B 0.376 h + 块通道开销）。
- 最多 6 条正式轨迹（R/N/S × 2 seeds）+ smoke；预计 3.0–3.5 GPU-h。
- 超过硬上限立即停止并报告，不挪用预算继续超参搜索。

---

## 8. 指标与科学判断

对每条轨迹报告：

- best-valid、W20、T100、final-valid、train@best/final；
- 逐 seed 同分子配对的 ΔMAE 与分子 bootstrap CI；
- ordinary（y>-3）、tail（y≤-3）、radius-2 all-supported/low-support/any-unseen、
  rings≥4；
- median AE、P90 AE、固定 worst-10 / 其余 990、drop-row172 敏感性；
- 不同块数量分组、块尺寸/块组合新颖性分组（只用结构预定义）；
- 参数量、GPU 时间、初始状态 hash、代码 commit、checkpoint hash、执行环境。

“在未见组合上更好”的主张必须控制块尺寸、原子数、环数量和极端标签的混杂；
不能仅凭一个 post-hoc 子群下结论。若 R 的改善主要来自 ZINC 人工 cycle penalty
的极端样本（大环/受罚子集），而不在一般分子/真实结构组合上出现，不宣称通用
组合泛化。

分子 bootstrap 只说明固定模型的样本不确定性，不代替训练 seed/SSL/执行非确定性。
两个 seed 共用同一对 SSL checkpoint（seed 0/1），不是完全独立的 SSL 初始化族；
若需确认，应另行设计 SSL encoder × supervised seed 的平衡交叉重复。

---

## 9. 决策标签

按预注册顺序：

1. **`OWNERSHIP_NOT_USEFUL`**：R−N 没有正向实用收益（两 seed 均 <0.005 或方向相反），
   或 R 在两个 seed 上都不优于 N。
2. **`SINGLETON_NONLINEARITY_ONLY`**：R−N 正向且可重复，但 R−S 不稳定/不正向；
   即收益主要来自单键块上的逐键内容与非线性，而不是多块之间的真实归属。
3. **`STRUCTURAL_GAIN_BUT_NOT_COMPETITIVE`**：R−S 方向稳定、平均改善 ≥0.003，
   但 R 的绝对 valid 性能不接近 DCC-L 均值（>0.077）。
4. **`PROMISING_COMPOSITION_MECHANISM`**：R−S 两 seed 同向、平均改善 ≥0.003，
   且 R 的 seed-mean valid ≤0.077（接近或优于 DCC-L 0.075990）。
5. **`UNRESOLVED_DUE_TO_VARIANCE`**：符号/CI 不能区分，或 seed 间冲突，
   无法归入上述任一；不追加 seed、不改设计。
6. 若机制未激活：**`MECHANISM_INACTIVE`**，明确这不是科学否定，停止且不救援。

---

## 10. 本轮不声称

- 不声称“块归属一定有用”或“原理成立”；R/N/S 是检验，不是证明。
- 不声称 R−N 是严格同信息对照（N 丢归属信息）。
- 不声称 C（共享块级非线性）已被检验；三臂共享 `phi`，C 需要额外控制。
- 不声称旧 BlockModel 的负结果被推翻；本轮对象/特征/计算/读出都不同。
- 不声称 ZINC 上的收益可外推到真实性质任务；若 R 有机制收益，后续应设计
  train-only 结构组合留出与 Lipophilicity scaffold-disjoint 验证。
- 不读 official test；不用 valid 反复调参后仍称预注册。
- 不把代码正确性、机制活跃性或局部显著性包装成泛化成功。
