# EVIDENCE_MAP — BJOC 预测研究重定向的证据矩阵

* 轮次：`bjoc_predictive_research_redirection_v1`
* Track：`tracks/structure-contribution`
* 性质：**证据整合 / 假设选择**。本轮不训练、不读 official test / Locked / 任何新标签。
* 数字来源：`tracks/structure-contribution/results/*/REPORT.md` 与 `tracks/structure-contribution/STATE.yaml`
  （本文件只转录，不改写；如需复核请以对应 REPORT 为准）。
* 证据等级：**【已确认】**同轮配对/匹配对照 + 冻结协议 + CI；**【未闭环】**单 seed、暴露 split 或选择条件化；
  **【竞争解释】**与证据相容但未被证明；**【猜想】**缺证据。

---

## 1. 最强正结果

### 1.1 Masked structural SSL 初始化（唯一跨 split、跨 seed、跨任务复现的机制）

| 事实 | 数字 | 对照 | 来源 |
|---|---|---|---|
| 8k 内部 split A：SSL 初始化 vs 同轮 scratch | locked seed-mean `P−B = −0.011984`，CI95 [−0.017530, −0.005974]，两 seed 同向；`P−C = −0.007713`（C = +50% 监督更新对照） | 同 init / 同 batch order / 同配方，C 控制"多算力"解释 | `results/bjoc_masked_pretraining_v1/REPORT.md` |
| full-10k official train：official-valid | `B 0.099202 → P 0.083549`，seed-mean `−0.015653`，CI95 [−0.021051, −0.009904]，两 seed 同向 | 同轮同 GPU retrain 的 B 基线 | `results/bjoc_full10k_pretraining_transfer_v1/REPORT.md` |
| full-10k：exposed official test | `B 0.094627 → P 0.073957`，`−0.020670`，CI95 [−0.027157, −0.014712] | 冻结 B/P，单次读；**exposed test，非盲测** | `results/bjoc_pretraining_mechanism_terminal_test_v1/REPORT.md` |
| 跨任务：MoleculeNet Lipophilicity | valid RMSE `0.80261 → 0.71351`（Δ−0.08904，CI95 [−0.13291, −0.04270]，5/5 seeds）；test `0.77511 → 0.73444`（Δ−0.04056，CI95 [−0.08053, −0.00057]，5/5 seeds） | 同 split / 同参数量 419,713 / 同预算 / 逐 seed 共享 init；同协议 ECFP4+XGB test 0.85973 | `results/bjoc_cross_task_lipophilicity_v1/REPORT.md` |
| 负对照：保护预训练表示（encoder LR 3e-4, Q 臂） | `Q−P = +0.019484`，CI95 [+0.005177, +0.044905]，tail +0.182，RMSE 0.541 vs 0.230 | 同 init/配方，仅 encoder LR | `results/bjoc_full10k_pretraining_transfer_v1/REPORT.md` |

机制侧已确认事实（label-blind，出处同上）：

* 预训练三个 encoder 全部大幅移动（init→SSL 参数位移 atom 1.844 / bond 2.193 / junction 2.683）；
  junction encoder 在 SSL→fine-tune 的线性 CKA 仅 0.101/0.127（atom 0.81 / bond 0.56），
  junction 有效秩 init→SSL→P 为 5.70→14.41→17.38 —— fine-tune 必须重写 encoder
  （所以"表示不变性"不是收益来源）。
* J-reset（junction encoder 归零、atom/bond 用预训练）: `J−P = +0.007616`（CI95 [−0.000202, +0.019367]，
  `J_RESET_NEUTRAL_VALID`，点估计更差、tail 驱动）→ 完整预训练 encoder 状态必须保留。
* tail 诊断（`bjoc_pretraining_mechanism_terminal_test_v1` Phase B，`TAIL_GENERALISATION_GAP_SUPPORTED`）：
  P−B 的收益是普通分子收益（train ordinary −0.017318 / valid −0.017391），valid tail 反而 +0.009341；
  失败极端分子在无标签 composition/topology 空间离 train 更远（NN 2.37 vs 1.62），
  其 5-NN label 邻域无信息（corr −0.154 tail vs 0.815 ordinary），且是系统性幅值**低估**（signed 占 |err| 相关 0.995）。

### 1.2 监督配方 R3（360 epoch + cosine）

| 事实 | 数字 | 来源 |
|---|---|---|
| R3 vs R0/P（official-valid，seed-mean） | `0.083549 → 0.079844`，Δ−0.003705 CI95 [−0.006390, −0.001011]；ordinary −0.004133；**seed 1 单独仅 −0.001727**；tail +0.002443 且 seed 符号翻转（`TAIL_WORSE`） | `results/bjoc_recipe_tuning_v1/REPORT.md` |
| R3 的官方 test（exposed） | `P3 0.069822` vs `P0 0.073957`，Δ−0.004135 CI95 [−0.007342, −0.001042]；**0.06–0.07 目标带（在 exposed test 上）达标** | `results/bjoc_r3_frozen_terminal_test_v1/REPORT.md` |
| 机制归因 | epochs×schedule 交互为主：+50% constant 更新有害（R2 +0.006673）、240ep cosine 脆弱（R1 −0.003049，CI 跨零）、wd 1e-4 有害（R4 +0.004892）；fit 与 valid 同向但 fit 被低末段 LR 混杂 | 同上 |

**关键限制**：R3 是**在已暴露的 official-valid 上选择出来的**候选（`RECIPE_GAIN_CANDIDATE`），不是独立盲确认；
seed-1 单独不能过 −0.003 门槛；其边际价值也最小（~0.0037，而 SSL ~0.0156）。官方 test 已被历史读取多次，
P3 的 0.069822 只能是"exposed-test 终点测量"，不能用于新的选择或盲测声明。

### 1.3 局部 bond–bond 关系通道（单 seed，未闭环，且不在 BJOC 骨架上）

| 事实 | 数字 | 来源 |
|---|---|---|
| 一跳 real bond 线图关系 + 共享 Pair MLP vs 同机 uniform 对照 | `R1−U = −0.012452`，CI95 [−0.018363, −0.006647]（split-D Monitor，1000 分子，seed 0）；普通/尾部/top-10-removed 全偏 R1 | `results/bond_rw_local_vs_multiscale_v1/REPORT.md`；`results/bond_random_walk_organization_v1/REPORT.md` |
| 多步游走 vs 一跳 | `R123−R1 = +0.002724`，CI95 [−0.001702, +0.007515] → 多尺度不可区分 | 同上 |
| 未解决的混杂 | 对称化 `Ks` 的行和与线图度相关（corr 0.888），Pair MLP 有 bias ⇒ **部分收益可能是度/中心性偏置，而非"真实关系组织"**；单 seed、暴露 Monitor、RMSE 方向相反、两臂都严重过拟合（fit ~0.002 vs Monitor ~0.11） | `results/bond_rw_local_vs_multiscale_v1/MATH_AUDIT.md` |

这是**结构组织线上唯一取得配对 CI 全负的对照**，但它是在 349,985 参数的"无 Junction encoder"骨干上得到的，
且绝对值（R1 Monitor 0.104796）描述性地略落后于 BJOC-Null A2/P3（0.103960，不同预算/轨迹，不可归因）。
它说明"真实局部拓扑有可用配对信息"，但没有证明把该通道加进 BJOC-Null 能带来增量（见 §2 的 DLP/cross-bond 反证）。

---

## 2. 已用匹配对照否定的路线（不得重复的机制家族）

| 路线 | 干预 | 关键数字 | 判定 / 来源 |
|---|---|---|---|
| 真实 Junction→Bond ownership | O = Σ_{j∋b} w_j vs Null `n_b·μ_G`；加 count-matched shuffle S | `O−N = +0.00301` [−0.00269, +0.00990]；`O−S = −0.00199` [−0.01179, +0.00507]；seed 1 未购；O 局部通路被使用（删除 Δ0.532）但 S 绕过局部通路（Δ0.022）仍接近 N | `OWNERSHIP_NOT_MORE_VALUABLE_THAN_SHAM`，`results/bjoc_source_ownership_v1/REPORT.md` |
| 全局条件化 ownership 门控 GLRC（+34 参数，零初始化） | `W'_b = W_null + tanh(θ·g_G)(W_local−W_null)` | Monitor `+0.001247` [−0.001145, +0.003644]（两 seed 均更差）；Locked `−0.001024`（符号翻转）；门活跃但方向 seed 不稳 | `GLRC_NO_GAIN`，`results/bjoc_architecture_breakthrough_v1/REPORT.md` |
| 距离条件 rank-32 pair products DLP（+27,648 参数） | d=3/4/≥5 桶 pair 乘积拼到 reader | Monitor `+0.002204` [−0.001134, +0.005518]；seed 1 未购 | `DLP_NO_GAIN`，同上 |
| 跨键端点环境乘积 | `q_b = √96·μ_u⊙μ_v`（与 W,V,n 的 exact 分解证明新增 31% 方差） | `P−O = +0.001952` [−0.003703, +0.007767]；信息新但无预测增益 | `CROSSBOND_INFORMATION_BUT_NO_PREDICTIVE_GAIN`，`results/bjoc_cross_bond_interaction_v1/REPORT.md` |
| 共享加性结构残差 STRC（28 α + 129 β，零初始化） | `ŷ = f_θ(G) + s_aΣα[type_i] + s_jΣβ[ctx_j]` | `S−P = +0.004543` [+0.000927, +0.009536]（两 seed 均更差）；7/8 门失败；α/β 活跃且跨 seed 稳定（Spearman 0.99/0.89），但 S−P 差异由 base 解改变解释（R² 0.43/0.73）而非残差（R² 0.003/5e-6） | `STRC_NO_PREDICTIVE_GAIN`，`results/bjoc_strc_predictive_feasibility_v1/REPORT.md` |
| Junction 内容/入口/组织 | A0 无 W/V；A1 广播 μ_G/ν_G；A2 Null count；C count-only；D 直接 readout；E type-conditioned pool | `A1−A0 = −0.015275`（CI 跨零）；`A2−A1 = −0.002280`；`C−A2 = +0.023412` [+0.014641, +0.032549]（内容必要）；`D−A1 = −0.002818`（seed 0 −0.006760 / seed 1 +0.001125）；`E−A2 = −0.001679` | `COUNT_ONLY_INSUFFICIENT` + `J_DIRECT_READOUT_NO_GAIN` + `J_TYPE_POOL_NO_GAIN`，`results/bjoc_junction_sufficiency_audit_v1/REPORT.md`、`results/bjoc_junction_mechanism_exploration_v2/REPORT.md` |
| 监督期 task-adaptive masked SSL（TA-SSL） | 每 4 个 batch 加 atom-only CE λ=0.1 | `+0.000083` [−0.005414, +0.004620]；ordinary +0.002797 更差 / tail −0.035439 更好 / RMSE −0.055235；监督 wall 4.2× | `TASSL_NO_GAIN`，`results/bjoc_task_adaptive_ssl_v1/REPORT.md` |
| 固定两键 Junction Signature 解释线 | DEV+SIG vs DEV-only top-4 response | `+0.00053` [−0.00030, +0.00136]；交叉拟合 −0.00024；context +(−0.00043)；k=1 −0.00747 | `INCONCLUSIVE`，**该线永久关闭**，`results/bjoc_signature_conditional_value_v1/REPORT.md` |
| 学习结构划分 + 加性 Reader | L-A vs F-A / F-O | 加性 vs 同容量 opaque reader 损失 ~0.04 MAE；最好 arm 0.1232 > BJOC-Null 参照 0.1043 | `STRUCTURE_UNITS_READOUT_BOTTLENECK`，`results/cscl_task_adaptive_structure_units_v1/REPORT.md` |
| Full-Y 距离池化 / 三体算子 / 预算对照 | Soft5 核；中心三体算子；7.2k→10k | Soft5 `+0.000743` CI 跨零；三体 case C（0.45%，CI 跨零）；预算对照 `−0.015440`（数据为准） | `research_core.md` 变更记录；`results/distance_pooling_bottleneck_v1/`、`results/strong_nongnn_predictor_v1/`、`results/zinc_full_y_budget_control_v1/` |

**跨家族共性**：这些干预都是"在固定 10k 结构上给模型加结构/读出/组织容量或任务"。
它们在**机制上确实被使用**（删除/置换/合成 witness 均通过），但**没有**转成 held-out 泛化增益；
多次出现"某个 seed/某个 split 有效、另一个 seed 符号翻转"（GLRC、D/E、ownership、O8/N8）。

---

## 3. 预测瓶颈对照表

四种候选瓶颈，各自给出：支持证据、最强反对证据、未排除的竞争解释、可观察量、是否值得再检验。

### B1 表示内容不足（Atom/Bond/Junction 编码缺必要的局部化学差异）

* 支持：`C−A2 = +0.023412`（去掉 Junction 内容 W,V、只留 count 后显著变差，两 seed 一致）；
  A0（无 learned junction 内容）0.121514 vs A2 0.103960；`bjoc_junction_sufficiency_audit_v1` §3.3
  显示 A0 有 34.8% 设计容量结构性失活。
* 最强反对：所有"补充局部内容"的尝试（cross-bond q、DLP、type-pool E、STRC、typed signature）
  均无增益或更差；输入可观测性审计（`results/bjoc_predictive_bottleneck_v1/REPORT.md` §4）在
  10k train 上只找到 3 组完全同构重复（y 相同），
  `INPUT_COLLISION_FLOOR_NOT_ESTABLISHED`（0 行输入相同而 y 不一致）→ 输入层不是可测瓶颈。
* 未排除的竞争解释：现有内容"必要但已被 Null 聚合充分近似"；缺的不是内容而是**组合方式**（→B3）。
* 预期可观察量：若为主瓶颈，任意合理的新增局部特征（DLP/cross-bond 级）应改善 ordinary MAE——未观察到。
* 判断：**低优先**。它解释不了"加了内容却无用"的一致模式。

### B2 结构组织信息不足（全局统计丢失关系归属或高阶组合）

* 支持：`bjoc_junction_sufficiency_audit_v1` B1 条件性代数盲区**已确认**（同 w-multiset / counts 下，
  Null 对 true ownership 不可区分；worked example Null [3,3,3,3] vs Organized [2,4,3,3]）；
  B3 在 10k train 找到"不同分子的最近 reader 距离 3.35"（远小于 Fit 独立对 p01 = 14.26），
  即表示对某些真实不同结构强压缩。
* 最强反对：真实 ownership 的四个独立干预（O、S/GLRC、Organized、D 直接读出）全部未获增益；
  `O−N` 为正、`GLRC +0.0012`、`DLP +0.0022`、`crossbond +0.0020`；RW 一跳增益只出现在
  无 Junction 骨架上，且纠缠 degree 偏置。
* 未排除的竞争解释：组织信息**已被 count + 局部 bond/atom 特征近似替代**（B5b：ownership 会部分
  泄漏进 w 编码）；或者其预测价值存在于尾部但被 bulk 抵消。
* 预期可观察量：若为主瓶颈，`O−N`/`GLRC`/`DLP` 至少一个应稳定为负；实际全为正或跨零。
* 判断：**不值得再检验**（stop 已多次确认，重复风险最高）。

### B3 组合与读出不足（局部信息存在，但汇总/Reader 组合不出整体性质，尤其极端分子）

* 支持：
  * reader 只接收 `[Σ_b c_b, Σ_i a_i, Σ_i a_i², g]`（321 维）；B2 层析审计显示
    composer-sum 从其余块的可预测 R² 仅 0.536，但其**有效秩 27.9（d90=4）**；atom_sq_sum 线性冗余 R² 0.927；
  * `results/bjoc_predictive_bottleneck_v1/REPORT.md` §3：四模型共同高误差 36/100 集中在
    大环富分子（平均 23.1 原子、cyclomatic rank 3.14、junction 36.3），残差相关 0.907–0.949，
    误差符号一致 93–98% → **persistent model-family error**，不是数据随机性；
  * A0 vs A2 的 RMSE 反转由单个分子（row 514, y=−15.327）贡献 96%–110% SSE
    （`results/bjoc_junction_mechanism_exploration_v2/REPORT.md` Part I）；
  * TA-SSL 显示存在 bulk↔tail 的可移动 trade-off（ordinary +0.0028 / tail −0.0354 / RMSE −0.0552），
    即读出行为可以被目标改变，但当前目标不是最优组合；
  * TASU：严格加性分解 vs 同容量 opaque reader 损失 ~0.04 MAE（读出形态确实有代价，
    但那是解释性约束的代价，不是当前 BJOC 的瓶颈）。
* 最强反对：
  * **intensive 统计在信息上可由现有输入导出**（Σ 与 n 均在输入中 ⇒ mean、std 可计算），
    所以"scale-separated readout"是重参数化而非新增信息；而重参数化路线（STRC）已经失败；
  * 所有显式组合/校准修正失败（STRC、count-conditioned、signature、distance pooling）；
  * 模型 fit MAE ~0.0018（R3 soup）说明容量不是限制，是数据限制；
  * 尾部失败有 coverage/statistics 签名（B4），不是组合算子的必然结果。
* 未排除的竞争解释：第二矩信息（Σc²）确实不可由现有输入导出；也可能只是优化重参数化差异。
* 预期可观察量：若为主瓶颈，matched-capacity 的 scale-separated readout 应改善 RMSE/tail 而不牺牲 ordinary；
  若改善只出现在 fit、不出现在 held-out，则确认是优化重参数化。
* 判断：**低先验、可低成本检验**，保留为预设计好的 fallback，不作为首选。

### B4 数据覆盖与泛化不足（少见结构组合缺监督，极端分子在训练分布薄弱区）

* 支持：
  * `results/bjoc_predictive_bottleneck_v1/REPORT.md`：**SAMPLE_EFFICIENCY_BOTTLENECK_SUPPORTED**。
    同更新数下 `N8−N4 = −0.037304` [−0.043517, −0.031045]（两 seed 同向、top-10-removed −0.036507、
    ordinary −0.034592、tail −0.074469），是额外优化效应（−0.008665）的 ~4×；
    N4 把训练集拟合得**更好**（fit 0.029 vs 0.040）但泛化差得多（0.147 vs 0.110）→ 典型数据受限区。
  * SSL 是唯一大幅正结果，而 SSL 本质就是"从同一结构里再榨出监督"（有效数据效率机制）；
    它在标签更少的 Lipophilicity 上收益更大（valid −11.10% RMSE，5/5 seeds）。
  * tail 无标签诊断：失败极端分子离 train 更远（NN 2.37 vs 1.62）且 label 邻域无信息
    （`bjoc_pretraining_mechanism_terminal_test_v1` Phase B）。
* 最强反对：
  * 增加**未标注结构**不增加极端**标签**样本；tail 的失败是 label-space 尾部 + structure-space 稀疏的组合，
    外部结构只改善后半部分；
  * `bjoc_full10k_pretraining_transfer_v1` §5 明确说"exposure 路线已饱和（96.8 exposures/molecule，
    bond CE 0.016–0.019，plateau by visit ~80）" → 仅增加同目标曝光不会有增益；
  * 8k→10k 的"diversity 而非 exposure"说法是**跨 split、跨基线**的观察
    （8k 是内部 split A locked，10k 是 official-valid），不是同轮的干净对照——diversity 剂量-反应从未被测过；
  * 现有固定 10k 结构 + 无结构对照（S/U）都表明"organizing 同一批数据"无用，指向数据侧。
* 未排除的竞争解释：瓶颈是 **label 数量**（不可扩，协议禁外部标注数据）而不是**结构覆盖**；
  或持久误差是模型家族偏差（B3），加数据也修不动。
* 预期可观察量：若 B4 是主瓶颈且"结构覆盖"是其可动部分，则在**固定 SSL 更新数/固定下游预算**下，
  扩大预训练结构分布应产生单调（或至少不劣化）的 held-out 增益，且增益与"held-out 局部环境在预训练
  分布中的覆盖提升"相关；反之若更大语料反而更差，则是 exposure/优化问题而非覆盖问题。
* 判断：**四者中潜力最高，且是唯一未被匹配对照检验过的机制**；也是本轮的推荐方向。

---

## 4. 公平比较审计（哪些数字可以并排，哪些不行）

| 对比 | 可比性 |
|---|---|
| P−B（8k 锁定 vs 10k valid） | 不可直接比大小：不同 split、不同 baseline 轮次 |
| P−B、Q−P、J−P（同一轮内） | 可比：同轮同 GPU、同 init/batch order/配方 |
| R3−P（recipe 轮） | 可比但 selection-conditioned（在 exposed valid 上选出） |
| P3 test 0.069822 vs P0 test 0.073957 | 可比（同轮冻结模型、单次读）；但 test 已 exposed，不能当盲测 |
| R1−U / R123−U（RW 轮） | 同轮 matched（同 init/SSL/batch order），但**不同骨架、单 seed、暴露 Monitor、含 degree 混杂**；与 BJOC-Null 的绝对 MAE 不可归因比较 |
| A0/A1/A2/C/D/E（split-D Monitor） | 同轮 matched；split-D Monitor 为历史暴露探索集 |
| 各轮 official-valid 绝对值 | 多次读取、多次选择 → 只能作 dev 参照 |
| official test 全部历史数字 | 已 exposed（多次读取）；只能作 reporting，禁止再用于任何选择 |

另外两处必须披露的缺陷（本轮不修复、只记录）：

1. `results/bjoc_architecture_breakthrough_v1` 之前的 `bjoc_performance_mechanism_long_v1` 存在
   preprocessing 期 Monitor 标签访问缺陷（见该轮 `ERRATUM_label_access.md`）；后续轮已用 `LabelGuard` 结构性修复。
2. 远程 CPU 的 float32 归约是 run-to-run 非确定的（~1e-7），因此若干 replay 门使用噪声地板而非逐位相等
   （如 `bjoc_task_adaptive_ssl_v1` 的 A1 amendment）；这不影响 GPU 训练（deterministic given seed/order）。

---

## 5. 重复探索风险清单（本轮必须避免）

1. **ownership / incidence 组织**：O、S、GLRC、Organized、D 直接读出、E type-pool —— 全部已测且无稳健增益；
   "再加一条 ownership 通路"或"换 pooling 方式重做 ownership"都在同一 family 内。
2. **pair / 高阶乘积通道**：cross-bond q、DLP（d=3/4/≥5）、RW 多步（R123）—— 已测；
   一跳 RW 的正结果不在 BJOC 骨架上且含 degree 混杂，不能据此再开通道。
3. **显式加性残差 / 输出校准**：STRC（157 参数）已永久停止；"count-conditioned scale calibration"是同一 family
   （`results/bjoc_strc_predictive_feasibility_v1/REPORT.md` §5 已明确拒绝 Route B）。
4. **固定结构词表/签名**：两键 Junction Signature 及其 context 扩展永久关闭
   （`results/bjoc_signature_conditional_value_v1/REPORT.md`）。
5. **监督期 auxiliary SSL / task-adaptive 重建**：TA-SSL 已测（+0.000083，4.2× 成本）；
   λ/frequency/head 扫描被冻结规则禁止。
6. **在同一 10k 结构上加距离/池化/统计**：distance pooling、atom-sq、type features 均已有负结果。
7. **更多结构词表统计**（环/度数/更大半径）：Signature 轮已明确禁止。

**尚未关闭、且不属于上述 family 的只有三件事**：
(i) 预训练**结构覆盖分布**（本 Proposal 的 H-COVER，唯一性能导向机制）；
(ii) 读出 scale-separation / 第二矩（预设计 fallback，低先验）；
(iii) 在**新内部 split**上独立确认 SSL 与 R3、并用 frozen-encoder 对照分离"表示 vs 优化条件"（证据审计，不是新模块）。
