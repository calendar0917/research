# MODEL — 本轮模型的实际信息流与回溯边界

对象：本轮从头训练的 6 个单元 `{C1, FULL} × seeds{0,1,2}`，数据为本轮内部的
Fit(2016)/Select(336) 划分。计算图与参数逐项沿用冻结的上游实现
（`tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py`，
其 A0 与本轮 `lcrc_model.LCRC` 在本机 CPU 上逐位一致，见
`analysis/correctness_gates.json → frozen_equivalence`）。

本文件只描述**模型在算什么**，以及本轮允许/禁止的回溯解读。实验判定在 `REPORT.md`。

参照点：`analysis/correctness_gates.json` 记录 `raw_unit_formula` 的恒等式残差
（`Σ_v f(h_v) − L`、`L+ρ−ŷ`、原始单位换算三者均为 **0.0**，32 行 batch）。

## 1. 输入与预处理

| 字段 | 维度 | 来源 | 是否学习 |
|---|---|---|---|
| `atom_feat` | 51 | RDKit 原子类/电荷/氢 + 物理键图（环、度、半径 1/2 类型直方图等） | 否 |
| `bond_feat` | 13 | 原始 edge_class/stereo + 环/端点标志 | 否 |
| `atom_scalar` | 9 | 图分析 + 二跳枚举（`log1p_n_d2`、`log1p_n_paths2`、`closure_ratio`、`log1p_degree`、`in_ring`、`min_ring_size/8`、`ring_bond_count/4`、`is_terminal`、`is_fused_or_spiro`） | 否（Fit 标准化） |
| `global` (`g`) | 42 | 全图统计 | 否（Fit 标准化） |
| `nbr_code` | int | `atom_type(u)*5+(bt−1)` | 否 |
| `path_cat` / `path_end63` / `path_closure` | int | 二跳枚举 | 否 |

目标单位：原始 logD。训练在 Fit 标准化的目标上做 MSE；**报告一律换算回原始单位**。
`mu`（Fit 目标均值）在分子对差分中抵消。

## 2. 一条前向（逐行）

```
a_v  = AtomEncoder(atom_feat_v)                          51 -> 128 -> 96
b_e  = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bf_e])  301 -> 160 -> 96

S1  b_sum_v  = Σ_{e∋v} b_e                               (96)
    b_mean_v = b_sum_v / deg(v)                          (96)   <- 仅 FULL 有；C1 删去
    nbr_v    = Σ_{(v,u)} Emb_nbr[code(u,bt)]             (32)
S2  T_v = Σ_p Emb_type[path_cat_p]                       (32)
    E_v = Σ_p Emb_end[path_end63_p]                      (32)
    J_v = Σ_p W_j (U_p ⊙ V_p)                            (32)
    K_v = Σ_p Emb_close[closure_p]                       (8)
    C_v = Σ_p gate[path_cat_p] · a_{end(p)}              (96)
x_v = [a_v, b_sum_v, (b_mean_v,) nbr_v, T_v, E_v, J_v, K_v, C_v, scalars_v]
h_v = LayerNorm(LocalMLP(x_v))                           -> 64
h_v = Dropout_0.10(h_v)                                   # 仅训练，每次 forward 一次
L   = Σ_v f(h_v)                                          # 加性读出
ρ   = rho([g, mean_v h_v, mean_v h_v², log1p n_atoms])     # 池化读出
ŷ   = L + ρ                (标准化)      pred = mu + sigma·ŷ   (原始 logD)
```

`C1` 与 `FULL` 的差别只有一个 96 维输入块：`b_mean`（`b_sum/deg`）。因此
`x_v^{C1}` 是 `x_v^{FULL}` 的固定列子集，`LocalMLP` 第一层权重是同一权重矩阵的
对应列子集（`gate_c1_column_deletion`：列/权重差 0.0；把 FULL 的 `b_mean` 块置零后
重跑同一个头，与 C1 的输出差 ≤1.4e-6，差值是 529 宽与 433 宽 GEMM 的 fp32 归约
顺序）。

参数量：FULL 247 106，C1 228 674（`gate_c1_column_deletion` 逐项核对）。

## 3. 分子对差分的真实计算分解

对 P0 定义的 A→B（同一条共同核心、不同变量片段），两分子**都重新完整编码**：

```
dL_var  = Σ_{v∈B 变量片段} f(h_v) − Σ_{v∈A 变量片段} f(h_v)          (标准化)
dL_core = Σ_{v∈B 核心}     f(h_v) − Σ_{v∈A 核心}     f(h_v)          (标准化)
dR      = ρ(B) − ρ(A)                                              (标准化)
d_pred  = sigma · (dL_var + dL_core + dR)                           (原始 logD)
```

三条恒等式在 `analysis/trace_verification.json` 中对 **509 对 × 6 单元** 全部核对：

* `pred = mu + sigma·(L+ρ)`（`raw_unit_formula`，残差 0.0）；
* `d_pred = sigma·(dL_var+dL_core+dR)`（全部 |残差| ≤ 1e-5 原始 logD）；
* 反向查询严格变号（最大 |偏差| ~3e-7）。

### 3.1 为什么 `dL_core` 不依赖映射

`dL_core` 是"B 核心原子 `f` 之和 − A 核心原子 `f` 之和"，两个求和都是对**整个核心
原子集合**取和，所以与具体同构映射无关。逐原子的 `Δf` 只有在映射唯一时才有意义；
本轮 442/509 对的核心自同构群非平凡（对位取代苯环一类），因此逐原子结果一律按
自同构轨道合并或标 `mapping_ambiguous`，不作为群体统计的输入。

### 3.2 核验过的对称性

| 性质 | 结果（`analysis/trace_verification.json`） |
|---|---|
| 反向变号（全 509 对 × 6 单元） | 失败 0，最大偏差 ~3e-7 原始 logD |
| batch 拆分/合并 | 最大差 ~1.8e-7 标准化单位 |
| 原子重编号（502 个配对分子） | `Δh` ≤ 1.1e-6、`Δf` ≤ 7e-8、`Δpred` ≤ 2.4e-7 |
| 签名相同的核心原子 | `Δf` 与 `Δh` 均为 0（逐对记录 `max_abs_df_unchanged`） |
| 原子序 = RDKit 原子序 | 3360 个分子全部一致（`cache_alignment`） |

## 4. 全局统计 `g` 与池化读出

`ρ` 的输入是 `g`、`mean h`、`mean h²`、`log1p n`。**因此任何局部片段替换都会通过
`mean h`/`mean h²` 改变 `ρ`**，即使所有单个核心原子的 `f` 不变也不能保证 `ρ` 不变。
本轮把这一部分单独记成 `dR`，并在每对里报告 `g` 的变化（`g_n_changed`、
`g_delta_l1`、`g_delta_linf`）与 `mean_h`/`mean_h2` 的变化。`g` 由两个合法分子的
结构重算，不是从 A 搬运。

## 5. 规约（gauge）与绝对零点

对一个分子，`pred = mu + sigma·(Σ_v c_v + ρ)`。令 `c'_v = c_v + k`（所有原子）且
`ρ' = ρ − k·n_atoms`，则预测逐个分子不变。因此

* 逐原子贡献 `c_v` 的**绝对零点**不是物理量；
* 分子对层面 `dL_var` 与 `dR` 会随 `k·(n_var_B − n_var_A)` 互相移动，
  只有 `d_pred`（与 `e_delta`）被锚定。

`analysis/gauge_shift.json` 对每个模型/seed 在全部 84 个 Diagnose 对上展示该变换，
预测改变量为 0。这也是为什么 §6 的 TRACE 探针在"变量片段重原子数不同"的对上
（113/509）对规约选择敏感，报告中按 `same_var_heavy` 分层给出。

## 6. 本轮支持 / 不支持的解读

**支持**：在指定模型与指定数据分组上，真实局部编辑的响应 `d_pred` 可沿显式统计
（变量片段和 / 核心和 / 池化读出）逐项核查；恒等式与映射/重编号/批量对称性可核验；
某些变换的 `e_delta` 在 Diagnose 中存在方向性。

**不支持**：

* `dL_var`/`dL_core`/`dR` 不是误差的唯一分配，也不是化学因果；规约变换可以任意
  移动三者的比例；
* 删除一条二跳记录（上一轮的输入通道干预）**不是**本轮所说的分子编辑；
* 逐原子热点在多数对上因核心自同构而不唯一；
* `rho` 的分支响应通过 `mean h` 的矩耦合到全图，局部性只在 `f` 上严格成立；
* 本轮不修正模型，不做可积校正；任何"闭环残差"结论都不构成"已修复单分子预测器"。
