# LCRC 计算互补性与结构函数共性研究 v0 — REPORT

- Round: `tracks/structural-sufficiency/lcrc_computational_commonality_v0/`
- 数据：`ogbg-molpcba` 官方 scaffold split（Train 350343 / Valid 43793 / Test 43793，128 任务，官方 mask，官方 macro AP）
- Test 角色：`DEVELOPMENT_TEST / POST_HOC_EXPOSED_TEST_DIAGNOSIS`（**未**用于任何权重、超参或架构选择）
- 本轮未训练任何预测模型；全部预测来自历史 best-Valid checkpoint 的只读前向
- 计算：**0.02 GPU·h**（两次 res-2 集群尝试在缺特征缓存/排队阶段失败，实际 GPU 计算 ≈ 53 s）；其余全部本地 CPU
- 实际墙钟：约 2 h 40 min（**超出 90 min 硬上限**，见 §9 执行偏差）

---

## 0. 首页：八个问题的直接回答

**1. 现有 LCRC 计算方式组合后能达到多少 AP？**
Test（排序分数等权组合）：最佳异构二模型 **0.27919**（W2_s0+DYNAMIC_s0）、0.27649（W2_s0+FIXED1_s0）、0.27888（W2_s0+RC_REAL_s0）；
三模型 `CORE`（W2/FIXED1/DYNAMIC seed0）**0.28320**；六架构 `EXTENDED` **0.29534**（Valid 0.31341）。
Valid 最优单模型为 **FIXED1_s0**（Valid 0.27585 / Test 0.25704）。
距离 0.28：最佳异构对差 **0.00081**；CORE 超出 0.0032；EXTENDED 超出 0.0153。
距离 0.29：CORE 差 0.0068；EXTENDED 超出 0.0053。**这些是“多模型组合空间”，不是单个 LCRC 模型的成绩。**

**2. 有多少收益超出普通多 seed 集成？**
**没有可检出的超出量。** 异构二模型相对其最强成员的 Test 增益 +0.0116 ~ +0.0229（均值 +0.0158）；
同构 2-seed 组合的 Test 增益 +0.0137 ~ +0.0203（均值 **+0.0181**）。
更严格地看：`FIXED1_s0+FIXED1_s1`（Test 0.28046）、`DYNAMIC_s0+DYNAMIC_s1`（0.28252）**同时高于**
`W2_s0+FIXED1_s0`（0.27649）和 `W2_s0+DYNAMIC_s0`（0.27919）。→ 主标签 `GENERIC_ENSEMBLE_GAIN`，不是 `LARGE_COMPLEMENTARITY`。

**3. 不同任务是否存在可迁移的结构计算偏好？**
**没有。** `TASK` 权重在 Valid 内样本上仅 +0.0014（W2+DYNAMIC），而其 scaffold 3 折 OOF 增益 +0.0000（0.29607 vs GLOBAL 0.29603）；
Test 上 `TASK` 相对 `GLOBAL` 为 −0.0002 ~ +0.0006。GLOBAL 权重在 5 个模型对中 4 个落在 w=0.5。
Test 上的逐任务事后最优权重可达 0.287（vs 0.279），但那是 `POST_HOC_ORACLE`，不可部署。

**4. 现有路径函数能否被更基本的结构成分函数重建？**
在**已知类型内**可以部分重建：`SEM_2`（4 个真实成分 + 全部二阶交互，643 维）对冻结教师 `T_total` 的类型等权相对 RMS **0.349**、cosine **0.962**（occurrence 加权 0.117）。
但这是 214 个 Train 出现过的类型对上的插值（163 个有效维度 / 180 个拟合类型）；**类型留出**上 `SEM_2` 相对 RMS **1.27**（比均值基准 0.94 更差）。

**5. 真实结构描述是否明显优于随机描述？**
**在类型留出上数值上更优，但绝对意义不成立**：`SEM_2` 1.270 vs `NULL_2` 1.818（−30%），`SEM_1` 0.858 vs `NULL_1` 1.038（−17%）；
两条都超过 10% 参考线。但两者都劣于均值基准（0.942），且只有 34 个留出类型 → 统计功效极低，**判为未确立（`SEMANTIC_REUSE=false`）**。

**6. 能否在实际模型中替代原函数而不损失预测？**
**不能。** 替换覆盖率 99.998%（Valid/Test 路径），`SEM_2` 替换后 Test AP **0.20623（Δ −0.0477）**、Valid **0.21126（Δ −0.0540）**；
远大于 0.002 参考线。`MEAN` 替换 Δ −0.182/−0.193 —— 说明该计算块本身高度判别、不可由均值替代。
`NULL_2` 替换后 Test 0.04815（Δ −0.2058），即真实成分函数确实保留了远多于随机描述的信息，但仍不足以保持任务功能。

**7. 是否发现了非平凡、可读的结构计算规律？**
**没有达到可宣称的标准。** `T_joint` 幅度不小（RMS 0.790；占 `T_total` 能量 52%），但 `SEM_2` 对 `T_joint` 的相对 RMS 0.545 **差于** `NULL_2` 0.514；
留出上 2.014 vs 1.902。即：联合项恰恰是真实成分描述**未能**解释的部分 → `NONTRIVIAL_INTERACTION=false`。
`analysis/structural_rules.json` 中的 5 条候选规律明确标注为“拟合二阶函数差分”，未通过留出验证，不作为机制结论。

**8. 本轮结果是否提供下一版 LCRC 的充分设计依据？**
**不提供。** A 侧只有通用集成空间（无架构互补、无任务偏好、无共享结构函数可复用），B 侧只有样本内可压缩性、无可保持功能的成分函数。
结论 `A+generic / B−`：既不支持“共享结构函数库的多任务 LCRC”，也不支持“计算互补专家”。
按 §17/§18 要求：**不启动下一版训练，等待研究者判断**。

---

## 1. 资产清单与身份核对（`analysis/asset_inventory.json`）

| 模型 | 架构 | seed | 参数量 | horizon | best epoch | Valid AP（记录/重放） | Test AP（记录/重放） |
|---|---|---|---|---|---|---|---|
| W2_s0 | W2 | 0 | 994,324 | 40 | 24 | 0.26527 / 0.26528 | 0.25362 / 0.25362 |
| W2_s1 | W2 | 1 | 994,324 | 25 | 23 | 0.25706 / 0.25705 | 0.24628 / 0.24633 |
| FIXED1_s0 | FIXED1 | 0 | 1,128,468 | 25 | 25 | 0.27585 / 0.27585 | 0.25700 / 0.25704 |
| FIXED1_s1 | FIXED1 | 1 | 1,128,468 | 25 | 19 | 0.27310 / 0.27311 | 0.26030 / 0.26030 |
| DYNAMIC_s0 | DYNAMIC | 0 | 1,128,468 | 40 | 25 | 0.27507 / 0.27507 | 0.26223 / 0.26224 |
| DYNAMIC_s1 | DYNAMIC | 1 | 1,128,468 | 25 | 19 | 0.27298 / 0.27298 | 0.25380 / 0.25392 |
| S3_s0 | S3 | 0 | 1,018,532 | 40 | 29 | 0.27149 / 0.27149 | 0.25063 / 0.25063 |
| RC_REAL_s0 | RC-REAL | 0 | 1,051,022 | 40 | 33 | 0.26771 / 0.26771 | 0.25594 / 0.25594 |
| BOC_FULL_s0 | BOC-FULL | 0 | 1,352,596 | 40 | 29 | 0.26995 / 0.26995 | 0.25414 / 0.25414 |

- 9 个模型全部核对：split row IDs 与官方 scaffold split 元素级相等（历史 manifest 已证明），任务顺序 128 固定，标签身份来自冻结 label artifact（Test 通过继承的 gate 读取）。
- 补齐推理 5 个单元（Valid+Test，共 9 次只读前向，CPU，合计 159 s）：Valid 重放与记录值差 ≤ 7.7e-6。
- 排除：`CPC_s0`、`RHIC_s0`（仅有 Test logits；Valid logits 从未落盘，补齐需要各自的路径结构缓存 → 本轮不纳入）。
- **资产缺失**：DSCR（`direct_structural_computation_recurrence_v0`）的 4 个 checkpoint 是 host-only，工作副本中没有；本轮通过 `rr pull` 从历史 run `dscr-cstar-s1` 的结果路径取回（`checkpoint_best.pt`），逐一对齐记录的 Valid AP 后才使用。
- 训练配方差异（必须记录，不能据此宣称架构因果）：W2_s0 / DYNAMIC_s0 为 40-epoch 配方；其余 4 个为 25-epoch 前缀（frozen 余弦的前缀）。

## 2. Part A：组合方式与主表（`analysis/ensemble_main_table.csv`）

逐任务百分位排序分数（全 split、含任务标签缺失分子、平均秩）为主分析；原始 logits 平均为敏感性。

| 组合 | 类型 | Valid | Test | ΔTest vs 最强成员 |
|---|---|---|---|---|
| FIXED1_s0（Valid 最优单模型） | single | 0.27585 | 0.25704 | — |
| W2_s0+W2_s1 | 同构 | 0.28465 | 0.26736 | +0.01374 |
| FIXED1_s0+FIXED1_s1 | 同构 | 0.30049 | 0.28046 | +0.02016 |
| DYNAMIC_s0+DYNAMIC_s1 | 同构 | 0.30011 | 0.28252 | +0.02029 |
| **W2_s0+FIXED1_s0** | 异构（主） | 0.29594 | 0.27649 | +0.01945 |
| **W2_s0+DYNAMIC_s0** | 异构（主） | 0.29603 | 0.27919 | +0.01696 |
| FIXED1_s0+DYNAMIC_s0 | 异构（补） | 0.29239 | 0.27379 | +0.01156 |
| W2_s0+S3_s0 | 异构（补） | 0.28668 | 0.26795 | +0.01433 |
| W2_s0+RC_REAL_s0 | 异构（补） | 0.29004 | 0.27888 | +0.02294 |
| W2_s1+FIXED1_s1 | 异构（seed1） | 0.28796 | 0.27536 | +0.01506 |
| W2_s1+DYNAMIC_s1 | 异构（seed1） | 0.28955 | 0.27210 | +0.01818 |
| FIXED1_s1+DYNAMIC_s1 | 异构（seed1） | 0.28395 | 0.26820 | +0.00789 |
| CORE（3×seed0） | 多模型 | 0.30273 | 0.28320 | +0.02097 |
| EXTENDED（6×seed0） | 多模型 | 0.31341 | 0.29534 | +0.03312 |

- 同构均值增益：Valid +0.0230 / Test +0.0181；异构均值增益：Valid +0.0172 / Test +0.0158。
  数量匹配（都是 2 个模型），参数量与训练成本不同（W2 994k vs 1.13M；25 vs 40 epoch），**不宣称严格计算量匹配**。
- 敏感性（原始 logits 平均）：Test `CORE` 0.28745、`EXTENDED` 0.30127、`W2_s0+DYNAMIC_s0` 0.28468 —— 略高于排序组合；
  这与“logits 尺度可利用”一致，主分析仍用排序分数（尺度无关、可比较）。
- 0.28/0.29 距离（Test）：最佳异构对 0.27919（距 0.28 差 0.00081）；`CORE` 0.28320（−0.0032 / 距 0.29 差 0.0068）；
  `EXTENDED` 0.29534（−0.0153 / −0.0053，即同时高于 0.28 与 0.29）。

## 3. Part A：互补性结构（`analysis/task_complementarity.csv`, `complementarity_summary.json`）

- 逐任务 ΔAP 与支持度见 CSV（每个核心模型对一列）。**高支持子集**（Valid/Test 正例均 ≥ 20，106 个任务）：
  W2+DYNAMIC Test Δ +0.01475 / Valid Δ +0.01754；W2+FIXED1 +0.01548 / +0.01583；FIXED1+DYNAMIC +0.00450 / +0.00460。
- 2000 次配对任务 bootstrap（Test ΔAP vs 最强成员）：W2+DYNAMIC +0.0081 [0.0021, 0.0130]；
  W2+FIXED1 +0.0075 [0.0018, 0.0125]；W2_s1+DYNAMIC_s1 +0.0119 [0.0083, 0.0152]；
  W2_s1+FIXED1_s1 +0.0091 [0.0053, 0.0125]；FIXED1_s1+DYNAMIC_s1 +0.0026 [−0.0003, 0.0050]（跨 0）。
- 去掉 |ΔAP| 最大的 1/3/5 个任务后：核心对 Test Δ 仍为 +0.0090 ~ +0.0129，**不是少数任务驱动**。
- 方向一致率（Valid/Test ΔAP 同号）：W2+DYNAMIC 0.85、W2+FIXED1 0.80、W2_s1+DYNAMIC_s1 0.79、FIXED1_s1+DYNAMIC_s1 0.66；
  ΔAP 的 Valid↔Test Spearman：0.22 ~ 0.58。
- 任务最优计算类型：`analysis/task_weight_transfer.json` 中保存了每个任务在 `w ∈ {0,0.25,0.5,0.75,1}` 网格上的 Valid AP
  和最终收缩权重 `w_final`；无任务在高支持子集上稳定偏向某一模型（GLOBAL 权重 4/5 对为 0.5）。

## 4. Part A：任务权重迁移

| 对 | GLOBAL w | GLOBAL Valid/Test | EQUAL Valid/Test | TASK Valid 内样本 | TASK Valid OOF | TASK Test |
|---|---|---|---|---|---|---|
| W2_s0+FIXED1_s0 | 0.50 | 0.29594 / 0.27649 | 0.29594 / 0.27649 | 0.29738 | **0.29604** | 0.27663 |
| W2_s0+DYNAMIC_s0 | 0.50 | 0.29603 / 0.27919 | 0.29603 / 0.27919 | 0.29735 | **0.29607** | 0.27898 |
| W2_s1+FIXED1_s1 | 0.75 | 0.28827 / 0.27500 | 0.28796 / 0.27536 | 0.29097 | 0.28842 | 0.27563 |
| W2_s1+DYNAMIC_s1 | 0.50 | 0.28955 / 0.27210 | 0.28955 / 0.27210 | 0.29119 | 0.28956 | 0.27183 |
| FIXED1_s1+DYNAMIC_s1 | 0.50 | 0.28395 / 0.26820 | 0.28395 / 0.26820 | 0.28524 | 0.28386 | 0.26784 |

OOF 相对 GLOBAL：+0.0001 / +0.0000 / +0.0002 / +0.0000 / −0.0001。→ **TASK 权重无迁移价值**（21 个任务因支持不足回退 GLOBAL，其余任务权重多落在 GLOBAL 附近，收缩后几乎不变）。

Oracle（参照，非部署）：
- 逐任务 Valid 最优单模型 → Test：0.25407（W2+DYNAMIC 对）/ 0.26126（W2+FIXED1 对）；
- 逐任务 Test 事后最优单模型（POST_HOC）：0.27098 / 0.26896；
- Test 事后最优离散权重（POST_HOC）：0.27919 / 0.27649（与 GLOBAL w=0.5 完全一致，无额外空间）；
- 逐任务 Test 事后最优权重（POST_HOC）：0.28703 / 0.28519（+0.008 的乐观上界）。

## 5. Part B：冻结教师与精确提取（`analysis/structural_function_reconstruction.json`）

- 教师：W2_s0 best-Valid（epoch 24，sha256 `d86fe5e5…`），全部权重冻结、eval、no-grad。
- 只替换的三个列块：`s2_type` (448–512)、`s2_end` (512–576)、`s2_joint` (576–640)（W2 的 local MLP 第一层 849→384）。
  `T(p,e)` 三项表格：`T_type` [400,384]、`T_end` [80,384]、`T_joint` [400,80,384]、`T_total` [32000,384]。
- **身份重建通过**（`identity_verification.json`，160 个真实分子）：第一层 preactivation 最大误差 **0.0**（构造上不变）、
  `h0` 最大误差 **1.7e-6**、logits 最大误差 **7.6e-6**（FP32 累加容差内）。Part B 未因工程实现问题中断。
- 幅度：`‖T_total‖_rms = 1.0919`、`‖T_joint‖_rms = 0.7897`、`‖T_type‖_rms = 0.4513`、`‖T_end‖_rms = 0.5986`；
  联合项占 `T_total` 能量 **52.3%** —— 因此下面的 T_joint 分析不是“幅度很小”的情形。

**真实结构成分**（按实际 codec 解码并逐 split 校验）：`x = (middle_atom, bond1, bond2, end_atom)`，d=4。
`path_end63 % 5 == path_cat % 5` 在 Train/Valid/Test 全部路径上成立 → **bond2 在 p 与 e 中完全重复，确认后只保留一次**。
Train 出现过的 `(p,e)` 类型只有 **214 / 32000**（组件覆盖：middle_atom 13/16、bond1 4/5、bond2 4/5、end_atom 16/16）；Train S2 路径 27,061,160。

## 6. Part B：重建与留出

固定 Ridge（float64、不惩罚截距、`MSE + 1e-4‖β‖²`、无超参搜索）。相对 RMS 越小越好：

**6.1 全已知类型（214 类型）**

| 模型 | 维度 | 秩 | 类型等权 rel-RMS | occurrence 加权 | 分子支持加权 | cosine（类型等权） |
|---|---|---|---|---|---|---|
| MEAN | 0 | 0 | 0.9579 | 0.7961 | 0.8739 | 0.327 |
| CODE_ADD | 480 | 66 | 0.5280 | 0.3520 | 0.4523 | — |
| SEM_1 | 42 | 33 | 0.7191 | 0.4328 | 0.5573 | 0.725 |
| **SEM_2** | 643 | 163 | **0.3487** | **0.1165** | 0.1824 | **0.962** |
| NULL_1 | 42 | 33 | 0.8740 | 0.5305 | 0.6728 | — |
| NULL_2 | 643 | 163 | 0.4825 | 0.1513 | 0.2103 | — |

**6.2 类型留出（doc ≥ 3，hash 取 20% → 34 个留出键；全部属于“完整键未见、各成分均已见”子集；
“存在未见成分”子集为空，因为 4 个组件在 Train 中都已被覆盖）**

| 模型 | 留出 rel-RMS | T_joint 留出 rel-RMS |
|---|---|---|
| MEAN | 0.9419 | 0.9950 |
| CODE_ADD | **0.7911** | 1.2956 |
| SEM_1 | 0.8585 | 1.1199 |
| SEM_2 | 1.2696 | 2.0138 |
| NULL_1 | 1.0383 | 1.0867 |
| NULL_2 | 1.8182 | 1.9017 |

- 真实 vs 随机（同一 Fit/Holdout 划分、同维度）：`SEM_2` 比 `NULL_2` 好 30.2%、`SEM_1` 比 `NULL_1` 好 17.3%
  （超过 10% 参考线），**但留出绝对误差 > 1（比直接预测均值更差）**，且留出类型仅 34 个。
- NULL 干预强度充分：214 个键中 195 个（91.1%）被移动，19 个频率桶，无单点桶。
- `T_joint` 单独：全类型 `SEM_2` 0.5448 vs `NULL_2` 0.5138（真实更差）；留出 2.014 vs 1.902（真实更差）。
  CODE_ADD 0.8621 也不如 NULL_2 —— 联合项无法由真实成分描述解释，`NONTRIVIAL_INTERACTION=false`。

## 7. Part B：真实前向替换（`analysis/function_replacement.csv`）

替换方式：第一层仍以原始输入计算，减去三项路径块的教师投影、按原根节点与原重数加回 `T_hat`；其余模块/权重/读出全部不变。
Train 未见 `(p,e)` 保留教师 `T`。覆盖率：Valid 99.998%、Test 99.999%、Train 子集 100%（**未替换路径不是误差来源**）。

| 变体 | Valid AP | ΔValid | Test AP | ΔTest | logits ΔRMS | h0 rel-RMS 误差 | 逐任务 Spearman（均值） | Top 1% 重合 |
|---|---|---|---|---|---|---|---|---|
| ORIGINAL | 0.26528 | — | 0.25392 | — | — | — | 1.000 | 1.000 |
| MEAN | 0.07261 | −0.19267 | 0.07151 | −0.18241 | 2.928 | 0.920 | 0.456 | 0.175 |
| CODE_ADD | 0.16338 | −0.10190 | 0.15853 | −0.09539 | 2.031 | 0.628 | 0.722 | 0.421 |
| SEM_1 | 0.11198 | −0.15330 | 0.10735 | −0.14657 | 2.512 | 0.794 | 0.576 | 0.270 |
| **SEM_2** | 0.21126 | −0.05402 | 0.20623 | −0.04769 | 1.429 | 0.428 | 0.849 | 0.594 |
| NULL_2 | 0.04871 | −0.21657 | 0.04815 | −0.20577 | 3.568 | 1.112 | 0.278 | 0.107 |

（Train 4096 分子描述性监测：ORIGINAL 0.59294、SEM_2 0.48444（Δ −0.1085）、MEAN 0.13816、NULL_2 0.10213、CODE_ADD 0.35323、SEM_1 0.20797。）

判读：
- 该计算块**不可替代**：MEAN 就掉了 −0.19，`SEM_2` 掉 −0.048（超出 0.002 参考线 20 倍以上）。
- 真实成分函数明显优于随机描述（0.206 vs 0.048），即“真实结构描述捕获了教师函数的可观部分”，但**不足以保持任务功能**。
- `SEM_2` 的 0.594 Top-1% 重合与 0.849 Spearman 说明它保留了粗排序结构，却损失了 top 端的精确性——这正好解释了 AP 的下降。

## 8. Part B：可读结构规律（`analysis/structural_rules.json`）

仅给出函数差分（`SEM_2 − SEM_1`）层面的候选，5 条，均标注为“未通过留出验证的候选”：

| # | 成分组合 | Train 分子支持 | 交互范数比（组内） | 合法路径示例（解码） | 反例 |
|---|---|---|---|---|---|
| R1 | bond2=DOUBLE × end_atom=C_aromatic | 1,844 | 1.62 | C–(SINGLE,DOUBLE)→C_aromatic | 同对最低组（见 JSON） |
| R2 | middle_atom=C_aromatic × bond2=DOUBLE | 55,414 | 0.95 | C_aromatic–(AROMATIC,DOUBLE)→C | — |
| R3 | middle_atom=N × end_atom=N_aromatic | 1,754 | 0.89 | N–(SINGLE,SINGLE)→N_aromatic | — |
| R4 | middle_atom=P × bond1=DOUBLE | 2,675 | 0.84 | P–(DOUBLE,SINGLE)→C | — |
| R5 | bond1=AROMATIC × bond2=DOUBLE | 55,472 | 0.84 | C_aromatic–(AROMATIC,DOUBLE)→C | — |

- 共享计算：`SEM_1` 在 middle_atom 分组内对教师函数的残差解释比例 0.29（N）~ 0.45（P），即**一阶成分解释力中等**，
  其余部分落在二阶/联合项 —— 而联合项正是留出失败的部分。
- **不得把这些当作化学机制**：它们是依赖编码基准的函数差分，且 `SEM_2` 留出比 `NULL_2` 差的情形已排除“已验证的非平凡交互”。
  本轮**没有任何候选规律达到“可读且经留出验证”的要求**。

## 9. 判定、执行偏差与未完成项

**`analysis/verdict.json`**

- Part A 标签：`GENERIC_ENSEMBLE_GAIN`（未获 `LARGE_COMPLEMENTARITY`；无 `TASK_SPECIFIC_SIGNAL`；不适用 `LIMITED_COMPLEMENTARITY`，因为组合空间确实可观）。
- Part B 标签：`FUNCTION_RECONSTRUCTION=true`（仅样本内）、`SEMANTIC_REUSE=false`、`FUNCTIONAL_PRESERVATION=false`、`NONTRIVIAL_INTERACTION=false`。
- 二维结论：**A+（通用集成）/ B−** → 不支持共享结构函数库，也不支持架构互补专家；**不启动下一版训练**。

**执行偏差（必须记录）**

1. **墙钟超限**：计划 90 min 硬上限，实际约 2 h 40 min。主要成本：(a) 本地 OOM 一次（替换阶段原实现保存全 split 的 preactivation/h0，已改为 2048 分子捕获、全 split 只存 logits）；
   (b) `res` 主机因**其他轮次的 tracked 运行产物被改写**而 deploy 被拒（未改动他轮资产）；(c) `res-2` 集群两次尝试失败（缺 9 GB 特征缓存、c05 不可用排队）。
2. **GPU 预算**：目标 0.5 GPU·h，实际 ≈ 0.02 GPU·h（两节点各一次失败启动，53 s + ~0 s）。全部推理改在本地 CPU（16 核）完成。
3. **补齐推理**：5 个单元（W2_s1、FIXED1_s0/s1、DYNAMIC_s1 的 Valid+Test，DYNAMIC_s0 的 Valid）共 9 次只读前向；未重训任何模型。
4. **未完成/未执行**：`CPC_s0`、`RHIC_s0` 未纳入（缺 Valid logits）；Part B 初始状态对照未做（无法准确重建同一 W2 训练的历史初始化状态，按协议跳过而非伪造）；完整 Train 替换评价只做了 4096 分子的描述性监测。
5. **统计功效限制**：Part B 类型留出仅 34 个键，且 4 个原始成分在 Train 中全部出现 → “存在未见成分”子集为空，外推检验退化为“完整键组合外推”。

**机制证据 vs 推测**

- 机制证据（已执行、可复现）：身份重建；214 个类型对的完整表格；三种加权的 Ridge 拟合与留出；91% 键被移动的频率分层随机控制；99.998% 覆盖率的真实前向替换与官方 AP。
- 推测（明确标注）：Part A 的异构 vs 同构差异**混杂** horizon（25 vs 40）与参数量；“真实成分捕获了教师函数可观部分”是从 0.206 vs 0.048 的替换 AP 差异得出的**间接**判断，不是机制证明；
  R1–R5 只是函数差分候选，未经留出验证。

## 10. 交付物

- `PROTOCOL.md`（运行前冻结）、`code/`（`cm_common.py`、`cm_infer.py`、`cm_partb.py`、`cm_partb_run.py`、`cm_analyze.py`、`cm_report.py`）
- `analysis/asset_inventory.json`、`analysis/ensemble_main_table.csv`、`analysis/complementarity_summary.json`
- `analysis/task_complementarity.csv`（+ 每个模型对的逐任务表）、`analysis/task_weight_transfer.json`（含每任务收缩权重）
- `analysis/structural_function_reconstruction.json`、`analysis/function_replacement.csv`、`analysis/structural_rules.json`
- `analysis/structural_function_coefficients.npz`（Ridge 系数与截距，float32）、`analysis/structural_null_maps.npz`（NULL 固定双射映射）
- `analysis/part_a_raw.json`（含逐任务 AP 与所有统计）、`analysis/verdict.json`
- 冻结预测身份：`_local/inference_manifest.json`（未提交，路径与 sha256 记录在 asset_inventory）

未复制任何历史 checkpoint 或数据缓存。
