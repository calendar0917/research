# BOSI v0 — Bond-Oriented Structural Incidence（研究设计，不是已验证模型）

**日期：2026-10-26。** 设计基于 `calendar0917/research@d27358b` 与当前 main；本轮只读源码、历史报告、公开文献并做独立 CPU 数学见证，**没有任何新 ZINC 训练/评分，未读取 official valid/test**。

## 0. 决策

**`DESIGN_READY_CPU_FIRST`**：唯一值得先核查的架构命题是：*以物理键为预测主体，用原子为统计锚点，把“目标键的其他邻键”按真实结构关系分组，读取其有限阶对称统计；这样能否在不实例化 trainable junction token、也不递归消息传递的条件下提供足够的局部组合能力和更好的样本效率？*

**不是**证明 BOSI 新颖、比 DCC-L 好，也不是“多加三阶矩就会赢”。如果在 label-free 信息门禁或首轮匹配实验中失败，应停止。

## 1. 设计选择：哪些历史证据真正约束我们

- DCC-L：`zinc_dual_channel_composer_v1/code/dcc_models.py` 原始 forward 为 `a,b,w=P3Encoders`，`Wg,Vg`（全图 junction 均值×本键 junction 计数），`Wl,Vl`（真实每键 junction 归属），`c_glob/c_loc`，最终 `[Σc_glob,Σa,Σa²,global33,Σc_loc,Σc_loc²] → Reader`。494,193 参数。**junction 只是共享中心的无序二键组合 wedge，不是化学天然独立对象**。
- 旧 BJOC 去 learned junction、等总参数的对照 Locked **+0.019825 MAE**（`tracks/structure-contribution/results/bjoc_representation_optimization_v1/REPORT.md`）：说明当时的 learned wedge 内容有价值，**不证明必须显式创建 wedge token**；删掉它且不提供等价替代本来就不公平。
- 旧 DCC-L Local vs Null ownership 单 seed valid 改善 **0.00383**，但 Local 仍未超过 P3B；这支持研究“归属怎样进入预测”，**不支持复制 Local/Global 双通道即最优**。
- 已失败的近邻：`zinc_rooted_branch_operator_v0` 的中心无条件 e1/e2/e3+rel2 对 r2-unseen Q 反而 +0.0041；`zinc_structural_conditional_moments_v1` 的 DCC-L 旁路修正 +0.00064/+0.00328；`results/zinc_bond_conditioned_context_attention_v1` 的 junction learned attention 比 P3 差 +0.00745；`results/zinc_structural_token_attention_v1` 更差；DBT、多层 bond attention 未建立优势。**故不做根内无条件 moments、不做附加残差、不在第一阶段加 attention。**
- 文献 ESA+、GRIT、CSA、MoSE 已覆盖 edge-attention、pair attention、subgraph statistics。BOSI 的候选贡献只能来自 *目标键条件化的显式统计与可检验的信息/样本效率分解*；从未据此宣称原创。

## 2. 主干：原子是上下文，物理键是对象，统计是明确的关系，注意力暂不参与

### 2.1 不变的输入

对每个原子 `v`，复用原来的 `atom_feat(108)→a_v(96)`；每个无向物理键 `e={u,v}` 只编码一次，复用 `bond_feat(7)` 和端点对称 `b_e(96)`。**只从相同 Fit-only masked-SSL checkpoint 初始化 atom/bond encoder**；不加载曾用 ZINC 监督训练过的 DCC-L 参数，不实例化 junction encoder；其余模块从新 seed 初始化。

### 2.2 有方向但不递归的半键描述

原子 `v` 的关联物理键 `e=(v,u)`：`x_{v,e}=tanh(MLP([a_v,a_u,b_e,bond_feat_e])) ∈ (-1,1)^32`。这里方向只表示 *以 v 为中心*，反转无向键的存储方向不应改变最终预测；该表征一次性由静态初始原子/键状态计算，之后**不更新 atom/bond token**。

### 2.3 目标键条件化的邻键统计（本方法的核心）

对 `(v,e)` 定义伙伴集 `P(v,e)={f: f incident to v, f!=e}`。结构分组 `r(v,e,f)` **只由原始图**计算，优先级如下：

1. `T`：`e/f` 的另两个端点直接相连（三角闭合）；
2. `Q`：否则另两个端点共享除 `v` 外的其他邻居（四环/共享分支）；
3. `R`：否则两键同属以 `v` 为节点的同一个 biconnected ring block；
4. `N`：其余无上述闭合关系。

四组互斥、覆盖所有伙伴；额外保留每 `(v,e)` 的 `Σ_f log1p(number_of_additional_common_neighbors)` 等原始闭合计数，避免用优先级丢弃原 junction 中的计数证据；这些量只用结构不读标签。环归属需核对 block 算法与历史缓存一致。

对每个组 `r` 和通道 `d`：

```text
T1(z)=z, T2(z)=2z²−1, T3(z)=4z³−3z
n[v,e,r] = |{f in P(v,e): r(v,e,f)=r}|
M[v,e,r,k] = Σ_{f in P(v,e), r(v,e,f)=r} Tk(x[v,f]),  k=1,2,3
```

`M` 每个查询共 `4×3×32=384` 维，另存 4 组计数和原始闭合统计。空组全零，保留确切 count；不用 learned motif ID 或 r2 key embedding。每个中心度数上限须在 **Fit8k-only** 审计：若 deg≤4，则一个查询的伙伴至多 3 个。

**数学依据及边界：** 对固定的**一个标量投影**与至多 3 个伙伴，`n, ΣT1, ΣT2, ΣT3` 与普通 p1/p2/p3 有可逆三角变换，Newton 恒等式由其恢复投影值的无序多重集。因此它不是任意拍脑袋的三阶上限。**但多维坐标各自的矩并不自动保留跨通道配对**；使用先混合原子/键特征的 learnable 32 投影，必须做多维配对反例及真实分子 label-free 检查。更不意味着任意高维伙伴 multiset 全局无损。关系分组保留了无条件根内矩完全看不到的 `目标键—伙伴键` 真实对应。

**CPU 数学见证（已独立运行）：** 三伙伴 `A=[−.5,0,.5]`、`B=[−1/(2√3),−1/(2√3),1/√3]`，二者 `n=3,p1=0,p2=.5` 完全一样，但 `ΣT3(A)=0`、`ΣT3(B)=0.5773502692`；三阶确实增加可区别的伙伴形态。二维伙伴集合 `[(0,0),(1,1)]` 与 `[(0,1),(1,0)]` 每坐标所有阶矩完全相同；投影 `x+y` 后的二阶矩为 4 vs 2，证明必须处理跨通道配对。它们是**代数反例，不是合法分子标签反例**。

### 2.4 以物理键为唯一主读出对象

```text
s[v,e] = MLP([a_v,b_e,x[v,e],flatten(M[v,e,:,:]),n[v,e,:],closure,log(1+degree(v))]) -> 96
c[e=(u,v)] = MLP([b_e, s[u,e]+s[v,e], abs(s[u,e]-s[v,e]), s[u,e]*s[v,e], bond_feat_e]) -> 96
z[G] = [Σ_e c[e], Σ_e c[e]^2, Σ_v a[v], Σ_v a[v]^2, global33]     # 417 dims
yhat = MLP(417->256->256->1)(z)
```

不使用原来的 `w_j`、`group_composer`、`local_composer`、全图广播隐 junction；**新架构主体独立**。沿用 global33 是为了不无谓删除 DCC-L 的有效图尺度/环统计，保留所有 10k 标签不额外监督。这里的 `c[e]` 允许端点分支非加性交互，因此不是已失败的标量贡献 additive reader。

**实际参数审计**（以 `atom 108→128→96`, `bond 295→160→96`, `halfedge 295→128→32`, `side 617→128→96`, `compose 391→160→96`, `reader 417→256→256→1`，SiLU 为例，PyTorch Linear 含 bias）：`26,336 + 62,816 + 42,016 + 91,488 + 78,176 + 173,057 = 473,889`，低于 500k；实际实现若额外加入 closure 列则须重算 `side` 输入并守住 cap，不能未经核对继续使用 473,889。

## 3. 唯一同参数机制对照：R（真实条件统计） vs B（广播条件统计）

**R：** 使用上面真实 `M[v,e,r,k]`。

**B：** 对同一中心 `v` 用全部关联键的 `Sv[k]=Σ_{f∈Inc(v)}Tk(x[v,f])`，按 *相同的目标键结构关系计数* 广播：`M_B[v,e,r,k] = (n[v,e,r]/max(deg(v),1))*Sv[k]`。保留 `a_v,b_e,x[v,e],n[v,e,r],closure,global33` 完全相同，**全部参数形状、初始化、训练批次顺序、readout 也完全相同**。两臂只差 `M` 的伙伴值是否遵循真实目标键/结构分组。

注意：B 从 `Sv` 与本键 `x[v,e]` 原理上可重构**未分组**的 leave-one-out 总矩，因此 R−B 不能直接解释为“多了全部伙伴信息”。我们检验的是 *正确的伙伴身份/结构归属是否带来统计效率*，以及 R 对 B 的差距能否变成原版 DCC-L 的实际性能收益。先报告多少查询具备有区分度的伙伴/关系分布，若结构分组几乎退化，不能归因为环关系。

## 4. 注意力的位置：有条件的第二阶段，而非第一轮拼接

如果且仅如果 R 对 B 呈可重复的实用收益，且 R 与历史 DCC-L 的 Held-out 性能接近或更好，才考虑 `BOSI-A`：以当前键/中心的 `q_{v,e}` 作**单次 query**，读取四个由 `M_{v,e,r,:}` 得到的关系统计 token；softmax 只在非空的 relation groups 间做**类型选择**，不在 atom/bond tokens 间反复更新隐藏状态。保留固定矩路径，避免把结构信息变成易塌缩的注意力门。必须有**同输入、同参数量级的 FFN stat-fusion 对照**，否则只是添加容量。

这仍然是 cross-attention / gating 的既有数学形式，**不称为新的 attention 算法**。旧 junction-context learned attention、object-token attention、DBT 的负结果是风险先验；若首轮 R 本身远逊 DCC-L，绝不启动这一阶段去“救援”。

## 5. 执行/停止顺序（下一次 Agent 任务）

**A. CPU-only（强制，0 GPU）：** 复现现有 BJOC/P3 数据构建路径，不加载任何官方 test/valid；核对全部 Fit8k degree、R/B shape/parameter、`Σ_r n = deg−1`、batch/atom/bond-order 不变性、闭环与 ring-block 标签、empty/degree1、梯度、上述数学反例、真实 Fit8k 的分组覆盖率以及 `R−B` label-free 数值差；必要时抽样检查当前结构特征能否反推出 group 条件。若真实 R/B 数值几乎恒等或参数/输入路径失败，`NO_INFORMATION_SEPARATION` 并停止。

**B. GPU Stage1（条件）：** 固定 `zinc_dccl_bottleneck_v1` 的 Fit8k/Monitor/Locked 行号及 Fit-only normalization，原版 DCC-L 两个 seed **只复用 checkpoint/历史预测，不重训**。R/B 在相同 SSL atom/bond 初始化、同 seed 相同公共参数初值、相同 batch 顺序/47,400 updates/batch128/L1/AdamW 下分别训练 seed0；Monitor 只用于各自 best checkpoint。若 R0 比历史 DCC-L s0 Monitor 差 >0.012 且 R0 对 B0 未达 0.003 改善，则停止不加 seed；否则按预先锁定的规则买 R1/B1。所有新训练 ≤500k 参数；总预算先设 2.5 A100 GPU·h（含 smoke，超预算停止而不缩短更新数）。

**C. Locked 一次性报告：** 只有决策/预测 hash 冻结后，对已使用过、**不是新的盲测**的内部 Locked 一次性评估。报告 R−B、R−历史 DCC-L、B−历史 DCC-L 的 overall / ordinary / tail / top10-excluded / low-support / any-unseen 的双 seed 配对结果；CI 只描述分子抽样，不当作 seed 置信度。官方 valid/test 都不碰。

**决策：** `STAT_RELATION_PROMISING` 需要 R 在两 seed 的 Locked 上优于 B，差值具有 >=0.003 的稳定意义，且 R 至少接近 DCC-L、未见结构方向非负；`PERFORMANCE_PROMISING` 进一步要求 R 相对 DCC-L overall 改善 >=0.005、且不限于极端尾部、any-unseen 有稳健正向；`NO_GO` 是其余情况。只有前两者才谈 attention Stage2；不得因 single seed、单一尾部或表面 0.001 重新调模型。

## 6. 论文/创新边界

本研究目标是 **检验“junc token 是不是必要”的架构命题**，不是宣称“显式统计 + attention”原创。相关先例必须在论文端认真比较 ESA/GRIT/CSA/MoSE、Janossy/Deep Sets、统计矩与线性/核注意力。该设计的潜在方法学价值仅在：物理键条件下带闭合/环关系的低阶统计压缩，及其信息/泛化效率的受控分离；若 R 不胜 B 或 DCC-L，停止，不包装成性能方法。

参考外部：[ESA+ (2025)](https://www.nature.com/articles/s41467-025-60252-z)、[MoSE (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f11b71773294e68ffc9ec6792dd9d163-Abstract-Conference.html)、[2026 atom/bond dual graph](https://www.nature.com/articles/s41467-026-75005-9)。