# 架构审计 — BOSI v0 与 RBO / SCM / DCC-L / ESOC / SPS 的实际 forward 差异

**日期 2026-10-26。CPU only。** 基准 commit `5950260`（含 `DESIGN.md`），代码事实取自
`5950260` 工作树中的实际文件，历史数字只引用当时冻结的报告；本文件没有训练、没有使用
official valid/test。

审计先回答一件事：**BOSI 是否在数学或计算上等价于某个已失败的架构？** 结论：
**不等价**，但它有两个必须写清楚的"已存在"成分——

1. **BOSI-B 的无条件矩与 RBO 的根内矩是仿射同构的**（下面给出代数证明）。也就是说
   BOSI 的"矩压缩"本身不是新机制；新的是**按目标键—伙伴键结构关系分组**这一项。
2. **"真实归属 vs 广播"这一对照在 DCC-L 自身的 junction 世界里已经做过**（`L−C`），
   single seed official-valid **−0.003830**（CI 上界 +0.00007，几乎贴 0）；本轮的 R/B
   是同一假设在"键中心 + 无 learned junction"主体上的复测，不是新假设。

因此本轮的可检验命题被收窄为：
*（i）* 去 learned-junction 的键中心主干能否接近 DCC-L；*（ii）* 在**伙伴分配只在
19.5% 半键上才有定义**的真实数据上，正确归属是否带来 ≥0.003 的实用差异。

---

## 1. 四个模型的实际 forward（逐行核对代码）

### 1.1 DCC-L（`zinc_dual_channel_composer_v1/code/dcc_models.py`，494,193 参数）

```
a,b,w      = P3Encoders(atom_feat, bond_feat, junc_feat)         # 26,336 / 62,816 / 112,896
w_j        = Psi([a_i+a_k, |a_i-a_k|, a_i*a_k, a_j, b1+b2, |b1-b2|, junc_feat(32)])   # 608->160->96
n_b        = #junction incidences touching bond b
Wg_b       = n_b * mean_G(w)            Vg_b = n_b * mean_G(w^2)      # 全图广播（C 臂）
Wl_b       = sum_{j touches b} w_j       Vl_b = sum_{j touches b} w_j^2   # 真实归属（L 臂）
c_glob     = Phi_g([b, Wg, Vg, log1p(n_b), bond_feat])   # 296->160->96
c_loc      = Phi_l([b, Wl, Vl, log1p(n_b), bond_feat])   # 296-> 80->96
yhat       = Reader([sum_b c_glob, sum_v a, sum_v a^2, global33,
                     sum_b c_loc, sum_b c_loc^2])        # 513->256->256->1
```

关键事实（不是模块名，而是计算性质）：

* `w_j` 是**每个 wedge 一个 learned 非线性向量**，输入含**两个键**的 96 维嵌入与两侧
  far-endpoint 原子，因此它同时看见 `(i,j,k,b1,b2)`。
* 归属通道是 `sum_{j∋b} w_j`：**非线性在求和之前**（per-partner encoder → pool）。
* 已经有 `sum w` 与 `sum w^2`，即 learned token 的一、二阶矩。
* 一阶/二阶矩是对**学习到的 96 维向量**求的，不是对固定投影求的。
* 无递归、无 attention。

### 1.2 RBO（`zinc_rooted_branch_operator_v0/code/rbo_model.py`，331,521 参数）

```
z_{v->u} = [a_v, a_u, b_{vu}, SiLU(hist_proj(neighbour typed incidence histogram after deleting uv))]
x_{v->u} = SiLU(W2 SiLU(W1 z))                      # 320->128->32
p1 = sum_i x_i,  e2 = sum_{i<j} x_i*x_j,  e3 = sum_{i<j<l} x_i*x_j*x_l   # 逐元素（逐通道）
rel2 = sum_{i<j} g(rel_ij) * (x_i*x_j)              # 学习到的 relation embedding × 成对乘积
h_v  = SiLU(Root([p1, e2, e3, rel2, kfeat(5)]))     # 4*32+5 -> 128
yhat = Reader([sum_v h, max_v h, mean_v h^2, global33, 11 counts])   # 428->256->256->1
```

* 对象是**原子**：一个原子一个 `h_v`；目标键不存在，`(v,e)` 的"被哪个键查询"没有进入统计。
* 关系进入方式是**成对乘积的门**（`rel_ij` 是分支对的关系），不是**对统计量的划分**。
* 三体项 `e3` 是**不同伙伴之间的逐元素乘积**，不是单通道的幂和；对单通道两者由 Newton
  恒等式互相可逆（见 §3.1）。

### 1.3 SCM（`zinc_structural_conditional_moments_v1`）

```
c_loc'[e] = c_loc[e] + Wout([S1[e], A1[e], S2[e], A2[e]])       # Wout 零初始化，5,496 新参数
S_t[e]    = sum_f mu_t[e,f] * v_f  -  n_t[e] * mean_G(v),      v_f = h_e = c_loc[f]
A_t[e]    = n_t[e] * (cov_t[e] - cov_G[e])                     # rank-8 内容注意力
mu_t      : 物理键 line graph 上的最短路距离带 t1 in [1,2], t2 in [3,8]
```

* 关系是**键—键 line-graph 跳数带**，不是 T/Q/R/N 闭环/环块分类。
* 值是**模型自己的 `c_loc`**（监督训练过的隐状态），不是静态键描述子。
* 形态是**对已冻结 reader 的加性残差**（零初始化 → step-0 功能恒等），不是新的前置统计通路。
* 结论：SCM 的失败（S-only +0.00064，S+A +0.00328）测的是"在 DCC-L 现有输入上再叠加一个
  很小的带内聚合"，与本轮"伙伴统计从一开始就是唯一的伙伴信息入口"不同。

### 1.4 ESOC（`zinc_explicit_bond_object_v1`，最近的同族前作）

```
P_b        = bond b 的半径-1 诱导子图（局部节点 = 邻原子/端点）
loc_et     = 局部节点对之间的**真实键类**（0 无边，1/2/3 单/双/三键）
z_i        = Emb(atom type) + Emb(role = dist to (u,v))
phi_p      = MLP([z_i+z_j, |z_i-z_j|, z_i*z_j, onehot(loc_et)])       # 逐对 learned encoder
token_b    = DeepSet(phi_n(z), phi_p(pair))  →  L=4 层 sL2 attention + RRWP bias  →  64 维
yhat       = Reader([sum_b token, sum_b token^2, ...])
NULL 臂     : onehot(loc_et) 被替换成该对象内的**类频率向量**广播到每一对
```

* 关系对象是"键对象内**逐对**的真实键类"，粒度比 T/Q/R/N 细（它区分单/双/三键），
  聚合方式是 **learned pair encoder + attention**，不是固定矩。
* ESOC 的 NULL 与本轮 BOSI-B 是**同一假设的两种实现**：都把"哪个伙伴带哪条关系"
  替换成"关系分布"。ESOC REAL−NULL = **−0.02816**（official valid，1 seed，187,200 updates，
  绝对 0.12801 vs DCC-L 0.07599）→ 归属/对应在键中心主体上**确实有可用信号**，但那个主体
  本身离 DCC-L 太远。这是本轮 R/B 的先验：方向可能是正的，绝对水平是主要风险。

### 1.5 SPS（`zinc_statistical_program_search_v1`，两个显式统计主体）

* Q-fixed：以**原子**为 root，把"原子记录"和"键记录"按 `dist(r,·)` 分 6 组，用 4 个 learned
  query 做 log-sum-exp 读取；键记录被当作 (b_e, g_ru, g_rv) 的独立元素，**不保留目标键与
  邻键的结构关系**。
* F-content：图 Laplacian/森林边际（`Q_τ` 对角与边差），完全不同的数学对象。
* 两者都是 248k / 205k 参数、同配方，official-valid 0.110569 / 0.101948（DCC-L 0.078537）。

---

## 2. BOSI v0 的实际 forward（`code/bosi_models.py`，473,889 参数）

```
a_v   = AtomEnc(atom_feat)                                     108->128->96
b_e   = BondEnc([a_u+a_v, |a_u-a_v|, a_u*a_v, bond_feat])      295->160->96
x_v,e = tanh(HalfEnc([a_v, a_u, b_e, bond_feat]))              295->128->32
P(v,e)= {f != e incident to v},  r(v,e,f) in {T,Q,R,N}         纯图算法、无标签
M[v,e,r,k] = sum_{f in P, r(f)=r} T_k(x_v,f),  k=1,2,3, T=(z, 2z^2-1, 4z^3-3z)
n[v,e,r]   = |{f in P : r(f)=r}|
s_v,e = Side([a_v, b_e, x, M(4x3x32), n(4), closure(4), log1p(deg)])   617->128->96
c_e   = Compose([b_e, s_u+s_v, |s_u-s_v|, s_u*s_v, bond_feat])         391->160->96
yhat  = Reader([sum_e c, sum_e c^2, sum_v a, sum_v a^2, global33])     417->256->256->1
```

**与上述每一个的实际差别（不是名字差别）：**

| 维度 | DCC-L | RBO | SCM | ESOC | **BOSI-R** |
|---|---|---|---|---|---|
| 被查询对象 | 物理键 | 原子 | 物理键（残差） | 物理键 | 物理键（唯一主读出对象）|
| partner 编码 | learned wedge 向量 `w_j`（608→96，含两端键 + 两侧端点 + 32-d 闭环特征）| learned 分支向量 32-d（含邻居删边后 typed incidence 直方图）| 模型自身 `c_loc[f]` | learned pair encoder（含真实键类 one-hot）+ attention | 固定 Chebyshev 矩（1/2/3 阶）of learned 32-d half-edge 向量 |
| 非线性位置 | **求和前**（逐 partner） | 求和前（z→x）| 求和后 | 求和前 + attention | **求和后**（`M` 是 x 的固定多项式，再由 `Side` 非线性读取）|
| 关系进入方式 | learned junction 编码 + 归属 `sum_{j∋b} w_j` | 成对乘积门 `g(rel_ij)·x_i*x_j` | line-graph 跳数带加权 | pair-edge 标签进 learned encoder | **对统计量的划分**（4 个互斥组，各组独立矩 + 计数）|
| 是否新增 learned junction token | 是 | 否 | 否 | 否（但有 learned pair token）| **否** |
| 递归 / attention | 无 | 无 | 无 | 有（sL2 + RRWP）| **无** |
| 读出 | 513-d | 428-d | 513-d（+残差）| sum/sum² | **417-d** |

---

## 3. 五个必须区分的概念

### 3.1 原子中心的无条件统计 vs 目标键条件化统计

设中心 `v` 的伙伴分支向量为 `{x_f}`，目标键为 `e`。

* 无条件根统计：`p_k(v) = Σ_{f∈Inc(v)} x_f^k`（RBO 的 p1/p2/p3，SPS-Q 的组内和）。
* 目标键条件统计：`Sloo(v,e) = Σ_{f≠e} x_f^k = p_k(v) − x_e^k`。

**结论（代数事实）：给定 `p_k(v)` 与 `x_e`，`Sloo(v,e)` 完全确定；反之亦然。**
所以"对目标键条件化"本身**不增加任何信息**——它只把同一个根矩重新分配到不同查询上。
BOSI 里真正新增的是 `r(v,e,f)`：这个标签**依赖目标键的另一端 `u`**，从而把
`Sloo` 切分成"与这条目标键闭环/同环块"和"不闭环"两部分。没有 `r`，BOSI-R 就是
RBO 的矩换一个池化对象。

### 3.2 一阶/二阶矩 vs 三阶矩

对**单个通道**、伙伴数 `n ≤ 3`：

```
(n, ΣT1, ΣT2, ΣT3)  ⟷  (n, p1, p2, p3)  ⟷  (n, e1, e2, e3)  ⟷ 伙伴 multiset
```

第一、二个箭头是线性可逆（`ΣT2 = 2p2 − n`、`ΣT3 = 4p3 − 3p1`，Newton 恒等式），第三个是
三次多项式的根。因此三阶矩是 `n≤3` 时**逐通道完整的 multiset 不变量**；少一阶就丢信息。
`G1` 实测 Fit8k 最大度 4、最大伙伴数 3，所以这个"上限 3 阶"不是拍脑袋，而是与数据界紧的。
**但**这只对单通道成立（见 3.3），也不意味着任意高维伙伴 multiset 无损。

### 3.3 单通道矩无法保留跨通道配对

`T_k` 逐坐标作用，因此 `M` 只含 `Σ_j x_{j,c}^k`。反例（`G4` 实测）：
伙伴集合 `P={(0,0),(1,1)}` 与 `Q={(0,1),(1,0)}` 在**每个通道、每一阶**的矩完全相同，
但投影 `x_0+x_1` 的二阶矩是 `4` vs `2`。所以 BOSI 的 `M` 看不见伙伴内部的**通道间
协方差**；这部分信息只能通过 `Side` 对 `M` 的后续非线性、`Compose` 的 `s_u*s_v`
端点交互、以及 `bond_encoder` 的对称乘积间接表达。这是本设计的**已知表达力上限**，
不是 bug，也不是本轮能靠加宽网络绕过的问题（加宽不改变张量结构）。

### 3.4 邻键类型分布 vs 实际闭环归属

* **分布**（BOSI-B 保留）：`n[v,e,r]`，即"这条目标键周围有几条 T/Q/R/N 邻键"。
* **归属**（BOSI-R 保留）：哪个具体伙伴（带哪条 `x_f`）落进哪个组。

`G9` 的直方图保持置换实验给出了代码级证明：把每个半键内部的 relation 标签在伙伴之间
随机置换（每组计数完全不变），**B 臂输出逐位不变（|Δy| = 0.0）**，R 臂输出改变
（`|Δy| = 4.2e-4` at init）。B 在信息上是 R 的**严格粗化**：`M_B = f(M_R 的组和, n_r)`。

### 3.5 junction 有预测价值 vs junction token 必要性

`bjoc_representation_optimization_v1` 在**等容量**下去掉 junction reader：fit 0.051586
（对照 0.041495）、Locked 0.115888（对照 0.096063），**+0.019825**，CI 不含 0
（Split A = 同一 8k/1k/1k 划分）。这证明**该 junction 内容**在那个主体里有用；
它**不能**证明必须存在一个 per-wedge learned token，因为被删掉的是
"每个 wedge 一个 96 维 learned 向量 + 其和/平方和"，而不是"邻域内容"。
BOSI 用**同一条邻域内容的结构化矩**替代 token，是这条逻辑差异的直接实验：
如果 R 能接近 DCC-L，就说明"内容重要、token 不必要"。

---

## 4. 结论：是否等价于已失败架构

**不等价，可以进入 CPU gate。** 三项代码级差异：

1. **对象与读出**：BOSI 的伙伴信息**只**经 `M → Side → Compose → bond token` 进入；没有
   DCC-L 的 `c_glob/c_loc` 旁路，没有并行键通道，没有 supervision-trained 权重初始化。
   SCM 的残差形态（零初始化叠加到已训练 reader）在结构上无法出现。
2. **关系语义**：SCM 是 line-graph 跳数带；RBO 是分支对关系门；ESOC 是逐对真实键类进
   learned encoder；DCC-L 是 learned wedge。BOSI-R 是**目标键—伙伴键的闭环/环块互斥划分**
   加上**划分后的独立矩**。没有既有实现把"划分"和"固定矩"组合在键中心主体上。
3. **对照**：R/B 是**同一参数张量、同初始化、同步长**下，唯一差别为 `M` 的分配方式
   （PROTOCOL 修订 A1 给出与 DESIGN §3 字面公式的差异及理由）。

**但必须记录两条"非新"事实**（它们限定了论文主张的范围）：
(a) 无条件幂和/初等对称多项式在 `n≤3` 时互为仿射变换 → **B 臂的矩压缩不是新机制**；
(b) "真实归属 vs 广播"在 DCC-L 的 junction 世界已测得 **L−C = −0.0038**（single seed,
exposed valid），在 ESOC 已测得 REAL−NULL = **−0.028**（1 seed, 187k updates）。
本轮是这两条先验在**新主体**上的检验，而不是新假设的首次检验。

## 5. 审计得出的风险清单（写进 PROTOCOL 的 pre-registration）

| 风险 | 量化 | 处理 |
|---|---|---|
| 归属干预的覆盖面 | 只有 **19.5%** 的半键有 ≥2 个非空关系组；单组时 R 与 LOO 广播**恒等** | 预注册报告该覆盖率；决策时同时给"分配有定义的半键占比最高四分位"子组结果 |
| T/Q 极稀疏 | T 占伙伴行 0.53%、Q 0.21%；含 T 的分子 6.0%、含 Q 1.8% | 不把"闭环组"当作主要机制来源；单独报告其使用率 |
| 键中心主体绝对水平 | ESOC 0.128、SPS 0.102–0.111 vs DCC-L 0.076 | B 臂就是该风险的直接测量；若 B 已远差于 DCC-L，则 R−B 的价值不能救绝对水平 |
| 固定三阶矩 | 逐通道完整、跨通道不完整 | 不做加宽/加阶规避；在报告里限定主张 |
