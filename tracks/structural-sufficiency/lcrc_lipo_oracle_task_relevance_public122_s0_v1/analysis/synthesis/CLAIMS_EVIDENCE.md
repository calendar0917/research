# CLAIMS_EVIDENCE — 主张 × 直接证据 × 适用范围 × 尚未支持

本文件把论文里**能写**的主张逐条钉在已有的表/JSON 字段/代码上。每一行的
「尚未支持」是**证据缺口**，不是待办清单；本任务不自动执行任何缺口（不新增训练、
不新增性质/基团/seed/数据集/模型对比/probe）。

禁止的拼接（本文件全程遵守）：

- 不得把「实验 logD 上的校准酰胺子群偏好」与「新训练 WC_LOGP 模型的四家族改线正确性」
  拼成「同一个 logD 模型已在四个家族验证了化学机制」。
- 不得把八格 Möbius 的算术完备性写成化学因果。
- 不得把「一个 checkpoint、一个 seed、一个结构库」的证据写成稳定性结论。
- 无对应对照时，能力类主张（如"优于其他解释模型"、"LCRC 独有"）记**证据缺口**，
  而不是补训练。

---

## 0. 证据文件索引（全部已提交，本轮只读）

| 代号 | 文件 | 本轮用途 |
|---|---|---|
| E1 | `analysis/partA_clarifications.json` | 覆盖/主张矩阵、同人群三表、条件恢复 |
| E2 | `analysis/summary_tables.json` | 头部数字（Part A + 两任务） |
| E3 | `analysis/partB_wc_logp_{metrics,pairs}.json/csv` | WC_LOGP 逐对预测与家族统计 |
| E4 | `analysis/partB_molwt_{metrics,pairs}.json/csv` | MolWt 不变量对照 |
| E5 | `analysis/analysis_checks.json` | 共享配方一致性、overview |
| E6 | `analysis/oracle_{labels,manifest}.json`、`freeze_oracle.json` | 参照值构造与检查 |
| E7 | `analysis/cf_isolation.csv` | main CF 定义与 1810 计数 |
| E8 | `analysis/weighted_response.csv` + `.json` | **本轮新增**：统一加权口径 |
| E9 | `../lcrc_lipo_multi_motif_mechanism_public122_v1/analysis/mechanism.json` | 八格七项（冻结） |
| E10 | `../lcrc_lipo_model_dissection_v1/analysis/attribution/{interventions.json,TABLES.md}` | 固定权重干预、Δp 分解记录 |
| E11 | `../lcrc_lipo_s3_locality_public122_s0_v1/analysis/alias_audit.json` | 签名充分性/别名对的实测 |
| E12 | `../lcrc_lipo_model_dissection_v1/code/{dissect_model.py,dissect_common.py}`、`MODEL.md` | 计算式、变体、参数量 |
| E13 | `../lcrc_lipo_amide_connected_rewire_public122_v1/REPORT.md` | AF51 差 0.0 仍移动的实测 |
| E14 | `../lcrc_lipo_amide_dependency_public122_v1/REPORT.md` | 断键 → 越界（反例背景） |
| E15 | `../lcrc_lipo_edit_supervision_public122_s0_v1/REPORT.md` | 真实编辑监督无联合收益（边界） |

---

## 1. 固定主张矩阵（五行）

| # | 主张 | 直接证据 | 适用范围 | 尚未支持（证据缺口） |
|---|---|---|---|---|
| 1 | **显式局部结构统计可计算追溯**：C1 的预测完全由逐原子的显式统计签名 $\Sigma_v$ 与 `global42` 决定，改线可分解到「改变根集合 $U$」与池化项 | 计算式与签名定义见 `METHOD.md` §2–§3；`E12` `dissect_model.py::DissectLCRC.forward`（433 列 = a/b_sum/nbr/T/E/J/K/C/scal）；性质 1 的实测：`E11` `alias_pair_summary.{n_signature_multiset_equal,n_global42_equal,n_prediction_equal_within_tol}=5/5/5`，预测差 ≤3e-7；性质 2 的实测：`E10` TABLES.md（ΔL/Δρ/Δpred + `max Δ other f ≡ 0`，identity gap ≤5.8e-7） | 该冻结模型、该 `lipo_v1` 特征、该 checkpoint；用于**计算**追溯 | 不是**唯一化学归因**：$\Delta\rho$ 与 $\Delta q$ 同量级（+0.6393 vs +0.5203），无法分摊成原子贡献；签名不最小、不完整；签名不同的输入不保证模型能数值区分 |
| 2 | **实验 logD 上的训练相关结构偏好**：在等剂量可匹配的酰胺子群内，模型相对匹配对照更偏向目标连接，且该偏好随训练出现 | `E1` A1：amide/Test 校准后 62 分子 / 110 对；`E1` A1 `claim_matrix`：$rC$=+0.164 [0.052,0.278]，$E$=+0.321；`E2` `three_tables_amide_test.RETAINED_C`（B 0.492 / D_C 0.173 / **J_C 0.146**）；`E1` A3：`R2_TminusC_H`=+0.13555 [0.03348,0.23318]，`R3_TminusC_H`=+0.03067 [0.00377,0.05634] | **仅**该子群（Test 21% 对 / 38% 分子覆盖）；仅 `A_s0` 这一个 checkpoint/seed；`G_FIXED` 与 `FULL` 要分开写 | 其他家族异质：磺酰胺仅 $E$ 明确、芳醚**反向**（−0.433）、芳胺 n=4 且 $E$ 为负、酯未成立、磺酸酯无 C 未检验；**不是**全部酰胺、**不是**全部官能团；改线分子**没有实验 logD 标签** |
| 3 | **当前模型响应的内部路径**：该子群的端点差主要经由 typed-S2 通道及其与 S1 的交互，独立标量项很小 | `E9` `summary["test/calibration/amide"].H`：$g_2$=+0.1037、$g_3$=−0.0017、$\sum$=0.14602；`E1` A3 的 $R_2,R_3$ 与 `arithmetic_residual_max`=4.4e-16；`E10` `interventions.json`（`typed_s2_zero` mean|Δpred| 1.6774、`g_zero` 0.4736、`mean_h2_zero` 0.3353） | 同一冻结权重下的**条件性依赖**（八格是预定义输入分组） | 不是架构必要性（需重训）；不是 joint 算子独有优势（`B4/B5` 只是代数对照）；混合输入不是合法分子；$R_2/J_C$=92.8% **不能**写成「S2 独占 93%」 |
| 4 | **跨家族的计算任务相关响应**：在 WC_LOGP 参照下，模型对同批改线的响应跟随目标的方向与幅度 | `E3` `family_stats.wc_logp/test/*`：gain amide +0.108 [0.094,0.124]、sulfonamide +0.116 [0.074,0.158]、aryl_ether +0.119 [0.094,0.144]、arylamine +0.314 [0.258,0.363]，四者 6 项 Bonferroni 区间不含 0；sign 0.952–0.990；$q$ 一致率 0.819–0.892；新图对自身 oracle 的 MAE 0.067–0.119；原图 Test R² 0.99643 | **新训练的** WC_LOGP 模型（seed 0）、main CF（1810 对，Test 844 / Valid 966）、`FULL` scope | 酯未解决（8 分子/46 对，+0.063 [−0.007,0.138]）；磺酸酯未检验；不是 logP/logD 测量；不能追认实验 logD 的化学机制；单 seed |
| 5 | **不变量对照**：组成不变改线下参照变化严格为 0，而模型残留响应小且非零 | `E6` `checks.molwt_invariant_{pass,max_abs_delta}` = true / 0.0；`E4` `family_stats.molwt/test/*`；`E8` `weighted_response.csv`：分子权重 mean\|Δp\| 0.9045–1.9214 g/mol（0.0084–0.0177 σ，σ_train=108.294），Test POOLED 1.3293 g/mol = 0.01228 σ，且 **delta-MAE ≡ mean\|Δp\|（差 0.0）** | MolWt 这一个不变量任务；`FULL`；同一 1810 对 | **不是**精确不变性（最大个例 14.235 g/mol = 0.1315 σ，`sulfonamide:1275:1:0:d9-14` T 视图）；不从零均值/不显著推等价；不是真实化学机制 |

### 1.1 逐条细节与可引用字段

**主张 1（方法层）**

- 计算式：`p(G)=μ_y+s_y[Σ_v q_v+ρ(ḡ,mean_h,mean_h2,log1p n)]`，`mean_h2` 为逐分量平方后取均值。
  实现在 `E12` `dissect_model.py:434-435`（`mean_h = index_add(h)/n`、
  `mean_h2 = index_add(h*h)/n`）与 `dissect_model.py:470-473`（`y = contrib + rho`）。
- 签名六块 A–F：定义见 `METHOD.md` §3，与 `E11` 所在轮的 `PROTOCOL.md` §3 一致。
- 第一条必要性的实测反例：`E13` 报「每个原子的 AF51 特征差 0.0」而 Test `D`=0.2571、`J`=0.3206，
  说明逐原子 AF 不变不足以固定 `b_e`。
- 追溯**不是**归因：`E10` 的 `max Δ other f = 0.00e+00` 与 `Δρ ≠ 0` 同时成立。

**主张 2（logD 偏好）**

- 三表必须分开引用；`RETAINED_C` 的 `J_C`=0.1460 只在「有 C 的 110 对」上成立。
- $E>0$ 只表示「相对这一个初始化偏好上升」，与「$rC>0$」是两个命题（`E1` A1 分列）。
- 残余均值接近 0 不构成等价结论（不改写为「模型对酰胺无响应」）。

**主张 3（内部路径）**

- 七项恒等式已核：`g2+g12+g23+g123` = +0.135547 = `R2_TminusC_H`（差 <1e-9）；
  `g3+g13+g23+g123` = +0.030665 = `R3_TminusC_H`（差 <1e-9）。
  这是本轮的**算术核对**，不是新实验。
- $R_3$ 是条件标量恢复（含所有含 scalar 的交互），`standalone` 项是 $g_3$=−0.0017。
- `R2_T_H`(+0.4585) 与 `R2_TminusC_H`(+0.13555) 是不同量，禁止混写。

**主张 4（WC_LOGP）**

- `gain = mean|Δref| − mean|Δpred−Δref|`（分子权重，见 `E8`）；baseline 逐家族
  0.136–0.378 logP。
- Bonferroni 是 6 项（5 个家族 × 1 个 gain 项，磺酸酯无数据 → NA），不缩减乘子；
  `E3` `bonferroni.items.*.gain_bonferroni_excludes_zero`。
- 方向一致率仅对 `|Δref|>1e-8` 计算；零变化视图另计（`n_views_zero_ref`）。
  $q$ 一致率另列，且 $q_{ref}$ 允许为负，不要求任何家族 T>K0。

**主张 5（MolWt）**

- `Δref = 0` 是构造性质；因此 gain = −delta_mae 恒成立，**不是**质量指标。
- 分子权重口径下 `delta-MAE` 与 `mean|Δp|` 必须**精确相等**（`E8` 24 个 cell 全部通过，
  最大差 0.0）。这条算术核对使「小响应」这句话在两种口径下都可复算。
- 尾部必须保留：家族内分子权重最大 8.616 g/mol（sulfonamide），
  单视图最大 14.235 g/mol（0.1315 σ）。

---

## 2. 四个推广边界

| 边界 | 具体含义 | 后果 |
|---|---|---|
| **单 checkpoint / 单 seed** | logD 侧是 `A_s0`（seed 0，best epoch 244）；WC_LOGP_s0 / MOLWT_s0 各一个 seed-0 checkpoint，共享初始 state `aac1e061…` 与 batch 顺序 `b10278f9…` | 不能声称「稳定性」或「换 seed 仍成立」；seed 间差异未被测量 |
| **单结构库** | ChemBench Lipophilicity，4200 分子，官方 `random122` 3360/420/420，6 个连接家族，一个冻结改线目录（539 C 链接 / 364 唯一视图） | 不能外推到其他分子集或其他改线族 |
| **计算目标** | WC_LOGP = Crippen；MolWt = 平均分子量；两者都是**计算函数** | 不能写成新测量；不能追认实验 logD 的化学正确性 |
| **校准覆盖** | Test 保留率 amide 0.206 / sulfonamide 0.259 / ester 0.152 / arylamine 0.098 / aryl_ether 0.089（对）；磺酸酯 0 | logD 侧主张限定在被 C 覆盖的人群；校准外不外推 |

---

## 3. 已核对的旧口径问题（不覆盖旧文件，只追加说明）

| 位置 | 原文 | 核对结果 | 本轮处理 |
|---|---|---|---|
| `REPORT.md` §3.3「Pooled over Test: mean\|Δpred\|=0.246 vs mean\|Δref\|=0.241, mean\|Δpred−Δref\|=0.072」 | 未标注权重 | 复算确认是 **T 视图、pair 加权**（0.24620 / 0.24094 / 0.07204）。T+K 平均为 0.19136 / 0.18450 / 0.05573；分子权重 0.18456 / 0.17225 / 0.05573 | `E8` 并列三套口径；正文改用分子权重并标注 |
| `REPORT.md` §3.4「Pooled Test mean \|Δpred\| = 1.81 g/mol」 | 未标注权重 | 同为 **T 视图、pair 加权**（1.8127）。T+K pair/view 1.3644；分子权重 **1.3293** | 同上 |
| `REPORT.md` §1 Q4「≤0.054·std at p95 for every family」 | p95 归属 | 不复现：0.0536 是 aryl_ether 的 **max**/σ；逐家族 **p95**/σ 最大为 0.0425（sulfonamide）。分子权重下 p95/σ 最大 0.0269 | 以 `E8` 为准，正文不引用 0.054 |
| `analysis/summary_tables.json` / `partB_*_metrics.json` 的 `abs_delta_pred.*` | 平铺统计 | 与 `delta_mae`（层级）不同权重 | **保留原值**，在 `E8` 中标注为 pair/view 加权；未修改原文件 |

---

## 4. 与论文其他证据的关系（只标对应关系）

- **性能/效率主张**（LCRC 在 Lipophilicity 等基准上的精度/开销）属于论文的**另一类证据**，
  与本文件的方法/追溯主张并列而非互相替代。不得用 oracle R²（0.996 / 0.991，这是
  计算参照上的拟合）替代原任务的性能优势论证。
- **真实编辑监督**（`E15`）与**断键越界**（`E14`）是边界背景：前者说明「编辑监督在该配置下
  没有联合收益」，后者说明「断键会把输入推出 Train 分布，FULL 下的巨大位移是管线越界，
  不是化学信息量」。两者都不支持任何「LCRC 独有」的表述。
- **input aliasing**（`E11`）是方法的**边界**证据：C1 输入把 5 对非同构分子映射到同一预测，
  因此「模型响应的结构分辨率」本身有限；这也是「不得声称签名完整」的直接理由。

---

## 5. 一句话版本（可直接放进摘要的中性表述）

> LCRC 以显式局部结构统计进行预测，并允许把具体连接改动追溯到统计输入、局部读出及池化
> 读出的变化。在既定 Lipo 结构面板中，实验 logD 模型的偏好依赖家族和可匹配人群；在同结构库
> 的计算参照任务上，模型能在多类改线中跟随目标的方向与幅度，并在组成不变量任务上表现出
> 较小但非零的响应。这支持有边界、可检验的任务相关结构依赖分析，不等同于普遍化学因果机制
> 或唯一原子贡献。

英文版见 `PAPER_SECTION.md` §E1。该段**不**声称 LCRC 独有该能力，**不**声称优于其他解释模型。
