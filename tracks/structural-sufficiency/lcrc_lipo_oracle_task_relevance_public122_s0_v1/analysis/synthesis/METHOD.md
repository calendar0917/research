# METHOD — LCRC 的结构追溯方法（计算定义、充分签名、两条性质）

本轮性质：**方法定义 + 已有结果的统计口径统一 + 论文/导师材料**。
新增训练 = 0，新增干预 = 0，新增模型推理 = 0，GPU = 0。所有数字均从已提交产物读出，
不重跑、不改旧协议、不覆盖旧文件。

- round: `tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/`
- 报告 revision `b1688975`（训练 `7bf4774f`，后续修复 `ce7fdc3d`）
- 机制来源 round: `lcrc_lipo_multi_motif_mechanism_public122_v1`（协议 `0273cad6`，报告 `a5febe96`）
- 模型定义来源 round: `lcrc_lipo_model_dissection_v1`（`code/dissect_model.py`、`code/dissect_common.py`、`MODEL.md`）
- 签名定义来源 round: `lcrc_lipo_s3_locality_public122_s0_v1`（`PROTOCOL.md` §3、`analysis/alias_audit.json`）

本文件描述**模型在算什么**。凡是「模型依赖某个输入」都不等于「该输入有化学因果作用」；
凡是「改线视图的预测变了」都不等于「真实分子的 logD 应该这样变」。

---

## 1. 被解释的对象：一个已经冻结的模型

| 项 | 值 | 来源 |
|---|---|---|
| 变体 | LCRC-C1 = `NO_BOND_MEAN`（去掉 `b_mean` 这一 96 维块） | `dissect_common.py::VARIANT_DEFS["C1"]` |
| 参数量 | 228,674（FULL/A0 为 247,106） | `dissect_common.py::EXPECTED_PARAMS` |
| 训练配方 | AdamW，per-step cosine 3e-4→1e-5，wd 1e-3，dropout 0.10，clip 5，batch 128，360 epoch × 27 step = 9,720 update | `dissect_common.py::RECIPE` |
| 目标 | 实验 logD（ChemBench Lipophilicity `exp`），在标准化空间做 MSE，报告换算回原始 logD | `lcrc_model.py` + `train_task.py` |
| 归一化 | `global42` / `atom_scalar` 用 **Train 结构** 的 mean/scale；目标用 **Train 标签** 的 mean/std，均冻结 | `round_common.py::frozen_norm`，`train_task.py:176` |
| 数据 | ChemBench `Lipop`，commit `0b10d4c1…`，4200 分子，官方 `random122` = 3360/420/420 | `PROTOCOL.md` §1 |

口径提醒：本证据链里 logD 侧的数字全部来自 **`A_s0` 这个 checkpoint**（plain MSE，
SHA256 `bdd670e8…`，state `7d32221e…`，best epoch 244，Valid RMSE 0.7025358558，
Test RMSE 0.5761674772）。旧的 FULL/A0 pilot 参照（Valid 0.7108803968 / Test 0.5727470871）
只用于「计算图等价复播」，不要与 `A_s0` 的数字混用。

---

## 2. 计算式（逐行对应实现）

记原子 $v$、物理键 $e=(u,v)$、以 $v$ 为根的跳数 2 路径 occurrence $p=(v\to w\to u)$。

```
a_v  = AtomEncoder(atom_feat_v)                          51  → 128 → 96
b_e  = BondEncoder([a_u+a_v, |a_u−a_v|, a_u·a_v, bf_e])  301 → 160 → 96

b_sum_v = Σ_{e∋v} b_e                                    (96)
nbr_v   = Σ_{(v,u)} Emb_nbr[atom_type(u)·5 + (bt_uv−1)]  (32)
T_v     = Σ_{p: root=v} Emb_type[path_cat_p]             (32)
E_v     = Σ_{p: root=v} Emb_end [path_end63_p]           (32)
J_v     = Σ_{p: root=v} W_j( U[path_cat_p] ⊙ V[path_end63_p] )   (32)
K_v     = Σ_{p: root=v} Emb_close[path_closure_p]        (8)
C_v     = Σ_{p: root=v} gate[path_cat_p] · a_{end(p)}    (96)

x_v = [a_v, b_sum_v, nbr_v, T_v, E_v, J_v, K_v, C_v, atom_scalar_v]     (433)
h_v = LayerNorm( LocalMLP(433 → 192 → 64)(x_v) )                       (64)
q_v = f(h_v)                                                           (64 → 64 → 1)

mean_h  = (1/n) Σ_v h_v
mean_h2 = (1/n) Σ_v (h_v ⊙ h_v)          ← 逐分量平方后再取平均，不是 (mean_h)²
p(G)    = μ_y + s_y · [ Σ_v q_v + ρ( ḡ , mean_h , mean_h2 , log1p n ) ]
```

四点必须同时写对，否则复现不了已提交的数字：

1. **`s_y` 不能漏。** 训练在标准化空间做 MSE，`Σ_v q_v` 与 `ρ` 都是标准化单位，
   原始 logD 的预测是 `μ_y + s_y·(…)`。`μ_y, s_y` 是该任务 Train 原始标签的
   mean/std（WC_LOGP: 3.2733 / 1.3238；MolWt: 383.30 / 108.294；logD 见 checkpoint 的
   `target_norm`）。
2. **`mean_h2` 是二阶池化。** 它是 `mean_v (h_v⊙h_v)`，逐分量平方后取平均。
   写成 `(mean h)²` 会改变 rho 的输入（C3 变体 `NO_M2` 就是把这一项清零）。
3. **`ρ` 不能按原子分摊。** `ρ` 读的是全图池化量与 `global42`，它对预测的贡献
   无法唯一地分给某个原子/官能团（见 §4 性质 2）。
4. **`C_v` 的 gate 是学习参数。** `path_gate` 初始化为 1，但会训练；所以「S2 内容项」
   不是固定的 1.0 权重求和。

**FULL 与 C1 的差别只有一项**：C1 不把 `b_mean_v = b_sum_v/deg(v)` 送进 `x_v`。
因此 C1 的 `x_v` 是 433 维，FULL 是 529 维。本文件其余部分都以 C1 为准。

---

## 3. 「可验证的局部计算充分签名」 $\Sigma_v$

### 3.1 定义

对根原子 $v$，签名 $\Sigma_v$ 是下面六块**多重集**（重复 occurrence 保留，不做集合化；
不含原子 id、路径终点下标、分子 id、SMILES、split、标签）：

| 块 | 内容 | 它固定了 forward 里的哪一段 |
|---|---|---|
| A | `atom_feat[v]`(51) + `atom_scalar[v]`(9) | `a_v` 与标量输入 |
| B | 入射多重集 `(atom_feat[u], bond_feat[b])` | `b_e`（因为 `b_e` 读 `a_u,a_v,bf_e`）→ `b_sum_v`，并给出 `deg(v)` |
| C | 一跳 `nbr_code` 计数 | `nbr_v` |
| D | `(path_cat, path_end63)` 计数 | `T_v`、`E_v`、`J_v`（三者都是这对码的函数之和） |
| E | `(path_cat, atom_feat[path_end])` 计数 | `C_v`（content 项读端点原子特征，不是端点下标） |
| F | `path_closure` 边际计数 | `K_v` |

**为什么第一项（入射签名 B）是必要的。** 键编码器读 `a_u+a_v`、`|a_u−a_v|`、
`a_u·a_v` 和 `BF13`。因此**逐原子 AF51 完全不变**时，只要端点配对方式改变，
`b_e` 与 `b_sum_v` 仍会改变。只有「每个原子的 AF51 集合」不足以固定 `b_sum_v`；
必须是入射多重集（含键特征）。这一点在 `lcrc_lipo_amide_connected_rewire_public122_v1`
里被实测：保持连通的两键交换让**每个原子 AF51 差 0.0**，模型预测仍然移动
（Test `D`=0.2571、`J`=0.3206）。

### 3.2 「充分」的含义与不含义

- **是**：$\Sigma_v$ 相等 ⇒ 局部输入行 `x_v` 相同 ⇒ 在**任意**共享权重下 `h_v` 相同
  ⇒ `q_v` 相同。这是逐块对照 forward 读的字段得到的，不是从化学直觉猜的。
- **不是**最小签名。块之间可以有冗余（例如 B 与 C 部分重叠），本文不研究最小性。
- **不是**完整化学表示。签名丢掉了 id 级信息（哪个终点是哪个原子），因此
  **非同构分子可以共享签名**——这不是 bug，而是 s3_locality 轮已经记录的
  input-aliasing 现象（87 个局部组，加 `global42` 后 86 个 full-input 组，
  其中 5 对是非同构的**不同**分子）。
- **不是**「对分子性质充分」。它只是「对这个模型的计算输入充分」。
- 本文**不使用**模型没有消费的更高阶全联合表（例如 3-hop 直方图、全 pair 联合表）。
  用它们能区分更多分子，但那不再是对**这个模型**的追溯。

签名与 `global42` 分开记录，`global42` 永不折叠进局部签名。

---

## 4. 两条由现有计算式直接推出的性质

### 性质 1（签名决定预测）

> 若两个图 $G, G'$ 有 $\{\Sigma_v\}_G = \{\Sigma_v\}_{G'}$（多重集相等）且
> $\bar g_G = \bar g_{G'}$、原子数相同，则在**同一冻结参数与同一冻结归一化**下
> $p(G) = p(G')$，在精确实数意义下成立；实现按既有数值容差比较。

*证明.* 由 §3.2，$\Sigma_v$ 相等 ⇒ 对应根的 `x_v` 相同 ⇒ `h_v` 相同 ⇒ `q_v` 相同。
多重集相等意味着把根一一配对后 `Σ_v q_v` 相同。同时 `mean_h`、`mean_h2` 是 `h` 的多重集
的函数，`log1p n` 由原子数决定，`ḡ` 由假设相等，所以 `ρ` 的输入相同 ⇒ `ρ` 相同。
两项相加再乘同一 `s_y` 加同一 `μ_y`，故 $p$ 相同。∎

**已有数值证据（不新跑）**：`lcrc_lipo_s3_locality_public122_s0_v1/analysis/alias_audit.json`
记录 5 对**非同构**分子，其 $\Sigma$ 多重集与 `global42` 全等，复用 A 的预测差
$|\Delta p| \le 3\text{e-}7$（容差 2e-5）；逐根局部输入与 `h` 的多重集最大差 1.2e-6
（float32 求和顺序）。这就是性质 1 的实现级验证。

**这条性质不说什么。** 它不说「所有非同构分子都可分辨」，也不说签名捕获了化学。
反方向更弱：签名**不相等**并不能证明模型能数值区分（连续 MLP 可以把不同输入映到同一输出）。
s3_locality 的 `1467/2674` 正是「输入仍完全相同」的反例，`294/1665` 等是「输入可分」。

### 性质 2（改线的端点差分解）

> 对带原子映射的既有改线 $O \to V$，令 $U$ 为局部签名被改变的根原子集合，
> $U^c$ 为签名未变的根。则
> $$\Delta p \;=\; p(V)-p(O) \;=\; s_y\Big[\sum_{v\in U}\Delta q_v \;+\; \Delta\rho\Big],
> \qquad \Delta q_v = 0 \ \ (v \in U^c).$$

*证明.* 由性质 1 的逐根论证，$v\in U^c$ ⇒ `x_v` 不变 ⇒ `h_v`、`q_v` 不变 ⇒ $\Delta q_v=0$。
所以 $\sum_v \Delta q_v = \sum_{v\in U}\Delta q_v$。$p = \mu_y + s_y(\sum_v q_v + \rho)$，
$\mu_y,s_y$ 与图无关，相减即得。$\Delta\rho$ 一般不为 0，因为 $\rho$ 读
`mean_h`、`mean_h2`、`log1p n`，任何根的 `h` 变化都会通过池化进入 `ρ`。∎

**两点写作纪律：**

1. **$\Delta\rho$ 不是小量。** 已提交的 dissection 记录（`lcrc_lipo_model_dissection_v1/analysis/attribution/TABLES.md`，
   leave-one-S2-occurrence-out，原始单位、FULL seed0、rank 0 / row 2087）：
   $\Delta L=+0.5203$、$\Delta\rho=+0.6393$、$\Delta p=+1.1596$，且「其他原子 $\Delta f$」
   在整张表里恒为 0（因为只有根的局部输入被改）。$\Delta\rho$ 与 $\Delta q$ 同量级，
   所以**不能**把 $\Delta p$ 当成「被改那个原子的贡献」，更不能事后按比例分摊后
   称「唯一原子/官能团效应」。
2. **本文只引用已有残差，不新造。** 同一文件里有若干 `identity gap`（0 ~ 5.8e-7），
   属于 float32 求和顺序；Part A 的七项恒等残差最大 4.4e-16。

---

## 5. 图干预：八格 Möbius 是什么、不是什么

机制轮的「八格」是**预定义输入分组**上的计算干预，不是化学反事实：

- 三个二值因素：S1（一跳聚合，含 `b_sum`/`nbr`）、S2（二跳 typed 路径块）、
  scalar（`atom_scalar` 的 9 列）。$2^3 = 8$ 个格 = 「哪些块被送入 local MLP」。
- 在每个格上用**同一组冻结权重**前向，得到 `value(123)`、`value(12)`、`value(13)`、
  `value(23)`、`value(1)`、`value(2)`、`value(3)`、`value(0)`。
- 七个 Möbius 系数：$g_1,g_2,g_3,g_{12},g_{13},g_{23},g_{123}$，恒等式
  「七项之和 = 端点差」已提交（最大残差 `4.4e-16`，
  `partA_clarifications.json::A3_conditional_recovery.arithmetic_residual_max`）。

三条边界必须同时写：

1. **混合输入不是合法新分子。** 「保留 S1、去掉 S2」的输入在图空间里没有对应分子，
   它只说明当前函数对这条通道的依赖。
2. **算术完备 ≠ 化学因果正确。** 七项之和精确等于端点差，只保证分解是代数恒等式。
3. **固定权重干预 ≠ 训练必要性。** 它回答「这个已训练好的函数用不用这条通道」，
   不回答「训练时没有这条通道会怎样」（那需要重训，本任务不做）。

术语纠正（Part A 的核心更正之一）：

- $R_3 = \text{value}(123)-\text{value}(12)$ 是**条件标量恢复效应**，它等于
  $g_3+g_{13}+g_{23}+g_{123}$，包含所有含 scalar 的交互项。
  因此 **$R_3$ 不是 standalone main effect**；真正的单独项是 $g_3$ 另列。
- $R_2 = \text{value}(123)-\text{value}(13) = g_2+g_{12}+g_{23}+g_{123}$，是**条件 S2 恢复效应**。
- `R2_T_H`（目标 T 相对原分子 O 的恢复）与 `R2_TminusC_H`（同一批焦点 T 相对**匹配对照 C**）
  是**不同的量**，前者衡量「T 相对 O 移动了多少」，后者才对应「相对对照的 J 恢复」。
  混写会直接改变主张的适用范围。

已提交的算术核对（本轮只做减法验证，不重跑）：

| 量（amide / Test / RETAINED_C，H 空间） | 值 | 恒等式核对 |
|---|---|---|
| $g_2$ | +0.10372 | — |
| $g_{12}$ | −0.01142 | — |
| $g_{23}$ | +0.00854 | — |
| $g_{123}$ | +0.03470 | — |
| $g_3$ | −0.00172 | 真正的 standalone 标量项 |
| $R_2 = g_2+g_{12}+g_{23}+g_{123}$ | +0.135547 | = `R2_TminusC_H` +0.1355470 ✓（差 <1e-9） |
| $R_3 = g_3+g_{13}+g_{23}+g_{123}$ | +0.030665 | = `R3_TminusC_H` +0.0306653 ✓（差 <1e-9） |
| $J_C = \sum$ 七项 | +0.146024 | = `mechanism.json::summary["test/calibration/amide"].H.sum_of_terms` ✓ |

**由此得到的写作约束**：$R_2/J_C = 0.1355/0.1460 = 92.8\%$。这个比例的分子
$R_2$ 自身已经包含 $g_{12},g_{23},g_{123}$ 三个交互项，所以只能说
「条件 S2 恢复占该子群端点差的约 93%」，**不能说「S2 独占贡献 93%」**。
同理不能说「对所有分子必要」、「去掉 S2 重训必然失败」。

---

## 6. 剂量对照（C）在方法里的位置

- 剂量/caliper 是**统计剂量匹配**（一跳/二跳统计与键数），不是物理键数匹配。
- 每个家族里只有一部分焦点 T 找到合法 C：Test 保留率 amide 0.206（21% 对 / 38% 分子）、
  sulfonamide 0.259、arylamine 0.098、ester 0.152、aryl_ether 0.089；磺酸酯 0
  （`partA_clarifications.json::A1_coverage_and_claim_matrix`）。
- 539 条 C 链接是**结构链接**，对应 364 个唯一控制视图，**不是 539 个独立分子**。
  所有效应仍用原分子→实例→对照→donor 的权重与原来的 scaffold-cluster bootstrap。
- 三张表必须**分开**报：`ALL_K0`（全部合格 T/K0）、`RETAINED_K0`（有 C 的焦点 T 仍对原 K0）、
  `RETAINED_C`（同一批焦点 T 对匹配 C）。`ALL→RETAINED` 是**选人群/改线**变化，
  `K0→C` 是**同人群换对照**变化，二者永不合并成一句话。

amide / Test 三表（`G_FIXED`，原始 logD 单位）：

| 表 | n 对 | B | D | J |
|---|---|---|---|---|
| `ALL_K0` | 533 | 0.280 | 0.112（对 K0） | 0.187 |
| `RETAINED_K0` | 110 | 0.492 | 0.263（对 K0） | 0.273 |
| `RETAINED_C` | 110 | 0.492 | 0.173（对 C） | **0.146** |

---

## 7. 内部干预（已有，不新增）

已提交的模型内部干预只有三类，都在冻结权重、eval 模式下：

1. **块级/通道级置零**（dissection）：`g_zero`、`contrib_zero`、`mean_h2_zero`、
   `typed_s2_zero`、`agg_b5`，给出 mean/max |Δpred| 与 ΔRMSE
   （`lcrc_lipo_model_dissection_v1/analysis/attribution/interventions.json`）。
2. **leave-one-S2-occurrence-out**（dissection）：只从某根的 typed-S2 输入里删一条
   occurrence，图/S1/scalar/`g` 全不变，记录 root/middle/end 与两条键型、
   Δpred、ΔL、Δρ 与恒等 gap（`analysis/attribution/TABLES.md`）。
3. **八格 Möbius**（multi-motif）：叠加式通道开关，见 §5。

这三类都不产生「唯一原子贡献」，也不产生化学因果结论。它们的价值在于
**把「预测依赖哪个输入通道」变成可复算的算术**。

---

## 8. 任务参照：三种任务不能混成一个真值

| 任务 | 参照是什么 | 改写后的分子有标签吗 | 能主张什么 |
|---|---|---|---|
| 实验 logD | 实验测量（ChemBench `exp`） | **没有**（改线图无实验值） | 原图上的偏好、路径、人群覆盖 |
| WC_LOGP | RDKit `CalcCrippenDescriptors(...,force=True)[0]`，**计算函数** | 有（可对任意图算） | 改线的 Δpred 是否跟随 Δref 的方向/幅度 |
| MolWt | RDKit `Descriptors.MolWt`，**计算函数** | 有 | 组成不变改线下模型是否近似不变 |

关键区别，写论文时不要混淆：

- WC_LOGP **不是**新的 logP/logD 测量，它是同一个分子上的计算估计函数值。
- MolWt 的 $|\Delta\text{ref}| = 0$ 是**构造性质**（两键交换保持组成），不是模型性质；
  所以 MolWt 的 gain 恒为 $-$delta_mae，**不能**当质量指标。
- 计算任务上的正确性**不能**追认实验 logD 模型的广泛化学正确性：两者标签来源不同，
  且 logD 侧仍是「一个 checkpoint、一个 seed、一个结构库」。
- 本轮的 logD 强主张只能来自 Part A 的校准人群；WC_LOGP 的跨家族结论属于**另一个**
  （新训练的）模型。不得拼接为「同一个 logD 模型已在四个家族验证了化学机制」。

---

## 9. 参数/配置检查（已有记录，不重跑）

| 检查 | 结果 | 出处 |
|---|---|---|
| 参数量 | C1 228,674；FULL 247,106；18 变体逐一核对 | `dissect_common.py::EXPECTED_PARAMS`，`dissection/analysis/audit_cpu.json` |
| 两个计算任务共享的初始 state | `init_state_hash aac1e061…`，`full_init_state_hash 666b88d7…` 相同 | `analysis_checks.json::shared_recipe.consistency` |
| batch 顺序 | 两任务相同 `b10278f9…` | 同上 `per_task.*.batch_order_sha256` |
| 特征归一化 | 指纹相同 `6c56f9bc…` | 同上 `same_feature_norm_fingerprint` |
| 无改线标签/跨任务标签进入训练 | `no_rewire_labels_in_training=true`、`no_cross_task_labels=true` | 同上 |
| 原图性能（排除「模型本身不行」） | WC_LOGP Test R² 0.99643 / RMSE 0.0781；MolWt R² 0.99058 / RMSE 10.08 | `partB_*_metrics.json::original_graph_metrics.test` |
| oracle 无直接泄漏 | `direct_leak=false`（两个任务） | `oracle_manifest.json::checks.direct_leak` |
| 独立重解析 | 20 个视图，max W-C 差 8.9e-16，max MW 差 2.3e-13 | `oracle_manifest.json::checks.independent_reparse` |
| MolWt 改线不变量 | `molwt_invariant_pass=true`，max |Δ| = 0.0 | `oracle_manifest.json::checks.molwt_invariant_max_abs_delta` |
| WC_LOGP 改线确实变化 | 3617 检查中 2516 变化（69.6%），mean |Δ| 0.191、p95 0.553、max 1.060 | `oracle_manifest.json::checks.wc_logp_rewire_abs_delta` |

---

## 10. 本方法的边界（写论文时逐条对应）

1. **单 checkpoint / 单 seed。** logD 侧是一个 `A_s0`（seed 0）；两个计算任务各一个 seed-0
   checkpoint。没有 seed 1 复核，因此「稳定性」不在主张范围内。
2. **单结构库。** 4200 个 Lipophilicity 分子、6 个连接家族、一个冻结改线目录。
3. **计算目标。** WC_LOGP/MolWt 是计算参照，不是新测量。
4. **校准覆盖有限。** C 只覆盖每个队列的一部分；校准外的分子只报告不外推。
5. **签名充分但不最小、不完整。** 非同构分子可以共享签名（5 对已有记录）；
   反过来，签名不同也不保证模型能数值区分。
6. **不改旧数字。** 旧协议、旧 REPORT、旧 checkpoint 的量值一个都不改；
   本轮只追加口径说明与新汇总表（`analysis/weighted_response.csv`）。
