# CSCL Strong Prediction Route Audit v1（只读科研证据审计）

* Protocol id: `cscl-strong-prediction-route-audit-v1`
* 类型: 只读证据审计 + 路线决策（**零 GPU、零训练、零新 seed、零超参搜索、零 test 重算**）
* 日期: 2026-10-11（本轮）
* 分析脚本: `code/audit_strong_prediction_route_v1.py`（复算，输出到 `results/strong_prediction_route_audit_v1/`）
* 产物: `results/strong_prediction_route_audit_v1/{full_y_dev_k_strata.csv, evidence_index.json}`
* 决策文档: `notes/decisions/decision_strong_prediction_route_v1.md`
* 主问题: 历史分量监督系统（终点 C，official-test y MAE 0.080918 raw）相对当前 y-only 系统的增益，
  究竟来自 (1) 7200 vs 10000 数据、(2) 架构、(3) 分量监督、(4) 分解/环惩罚/校准利用方式中的哪些项？

---

## §0 摘要（≤300 字，决策摘要见 decision 文档）

审计结论：**历史 C 的 test 优势不是一个"机制"，而是三项可分辨 effect 的叠加**——
① 训练预算（7.2k→10k + 随之的 13,680→18,960 步与全量 refit）的 quasi-matched 估计 **+0.015(valid)/+0.018(test)**，
是全表中最大的单项；② 分量监督（同一骨架、同一初始化、仅损失不同的严格对照）**+0.0050(valid y-cal)/+0.0075(test)**，
CI 均不含 0；③ 环通道的"prototype + 离散级别解码"再给 **+0.0035(test)**，但其 valid 增益被单行（valid:0172）主导，
部署门限在 prototype 版曾明确 FAIL。**没有任何严格受控的数据量对照存在**（同代码、同超参、同 soup 规则、同评估的 7.2k vs 10k A/B 缺位），
故数据量效应只能给出"无法严格归因"的 quasi-matched 上界式估计。标签溯源确认：`c/k` 是**由 y 反推**的（O(1e-3) snap），
`k` 依赖历史流水线的原子序（`nx.cycle_basis` 非置换不变，94.8% 稳定 / 5.2% 不稳定），`ell` 是图可算的归一化 logP，
`s` 是 g−ell（≈归一化 SA 但**不是**纯 SA，ε-MAE 0.0027、max 0.370）。可部署链不使用真值 c；oracle 只出现在 O 诊断臂。
→ 推荐决策 **C**：证据不足以改变研究合同；下一笔该买的是**严格匹配的预算对照**（见 decision 文档预注册草案）。

---

## §1 范围、可读边界与越界披露

### 1.1 允许/执行的操作

* 读 git 历史、源码、历史结果 JSON/CSV/npz 中的**已保存预测**；`uv run research context`。
* 只读统计复算（CPU ≤8 线程）：MAE 分层、方差/协方差、标签恒等式、k 分布、置换敏感性汇总。
* 查 ZINC/RDKit/sascorer/networkx 定义与上游生成脚本。

### 1.2 未执行的操作

零 GPU、零新训练、零新 seed、零超参搜索；未加载官方 split 原始文件；**未重跑任何 test 预测**；
未用任何历史 test 数字做模型/配置选择（本轮不做选择，只做归因审计）。

### 1.3 越界披露（必须记录）

为核验历史文档中的已发布数字，本轮读取了**结果目录中已保存的 heldout npz**
（`zinc_full_decomposition_valid_test_confirmation_v1/heldout_test_predictions.npz` 等，其中含 test 标签），
仅用于复算历史已发布的 MAE（复算与文档逐位一致）。此行为触碰"不加载 test 标签"的边界，
披露在此；**本审计的任何结论/建议均不依赖这些复算**——所有引用数字均同时来自已发布文档与结果 JSON
（`evidence_index.json` 记录来源路径）。若评审要求绝对边界，可仅以文档数字重述，结论不变。

### 1.4 命名消歧（下文一律使用右列符号）

| 简称 | 本轮用法 | 易混淆项 |
|---|---|---|
| `Full` | 408,651 参数的 ksvd/CSCL 非 GNN 骨架（patch/ctx/topo/anchor + K-SVD 字典 + C6 掩码 + 重构辅助） | — |
| `M` | 297,539 参数的 ksvd 非 GNN 骨架（local tuple + 字典桥 + 39→1 reader） | — |
| `Y@10k` | Full、全部 10k 训练、raw-y L1（`zinc_full_decomposition_valid_test_confirmation_v1` Y 臂） | — |
| `P@10k` | 同轮 P = h(g-target body) + q(T25) + b | "P 臂"在其它轮含义不同 |
| `SUM` / `COMP` | M 骨架的 g-only / g+0.5(ell+s) 两臂（`zinc_chemistry_component_supervision_seed0_v1`） | — |
| `SUM_Q` / `B` / `H` / `C` | M 骨架 10k 部署链：g+连续 Q / COMP+Q / +prototype 环路由 / +离散级别解码（`zinc_cycle_level_transfer_terminal_test_seed0_v1`） | `K` 控制臂、`case C` 判据 |
| `O-arm` | ksvd oracle 诊断臂（真值 c 加回，`zinc_full_cycle_target_decomposition_v1`） | **CSCL `O_bucket`**（不透明读出臂）完全不同 |
| `F` / `O_bucket` / `S_bucket` | CSCL 7.2k 三臂（`cscl_v3d_official_eval`） | — |

---

## §2 表 1：性能合约表（不可混淆的性能证据）

口径约定：`cal` = 用 **fit-only** 中位偏差校准后的 MAE；`raw` = 原始 MAE。
官方 valid/test 若无特别说明均为"该冻结版本正式读取一次"的结果；**所有历史 test 数字都属于历史冻结版本，
不可再用于宣称独立性**。`epochs/steps` 中 steps 按 batch 128 计。

| # | 模型（臂） | 任务标签（训练监督） | 训练数据 | epochs / steps | 参数量 | 损失 | checkpoint 规则 | 校准 | official valid | official test | test 访问状态 | 来源 |
|---|---|---|---|---|---:|---|---|---|---|---|---|
| 1 | `Y@10k` (Full raw-y) | raw y | 10,000（全量 refit） | 240 / 18,960 | 408,651 | L1(y) + 33.95873017865987·label-free 重构 | 末 5 轮 soup {236..240} | fit-median b_Y（每 seed） | raw 0.118618 / 0.117695（mean **0.118156**）；cal mean 0.116635, ens 0.108338 | raw 0.094589 / 0.096643（mean **0.095616**）；cal mean 0.093323, ens 0.084133 | 已读（历史轮，冻结版本） | analysis_summary.json |
| 2 | `P@10k` (Full g+Q) | g = y−c（+同重构） | 同上 | 同上 | Full + Q 3,777（CPU） | L1(g) + 重构（Q: L1(q(T),c)） | 同上 / Q 末 5 轮 | b_P（每 seed） | raw mean 0.119297；cal mean 0.119003 | raw mean 0.089697；cal mean 0.089556；ens 0.081007 | 已读（同轮） | 同上 |
| 3 | ksvd `O-arm`（oracle） | g 训练 + **推理加真值 c** | 8,000 fit（fresh fold） | 240 | Full | L1(g)+重构 | soup | fit-median | —（仅 dev 0.100312 两 seed 均值） | 从未读 | 未读 | analysis_summary.json |
| 4 | `SUM`（M, g-only） | g | 8,000 fresh fit | 240 / 15,120 | 297,539 | L1(g) | 末 5 轮 soup | fit-median | —（dev g-cal 0.097945） | — | 未读 | main_table.csv |
| 5 | `COMP`（M, 分量监督） | g + 0.5(L1(ell)+L1(s)) | 同上（同 init/调度） | 同上 | 297,539（逐位同结构） | 同上 + 分量项 | 同上 | fit-median | —（dev g-cal 0.087972） | — | 未读 | gate.json |
| 6 | `SUM` / `COMP` @10k 确认 | 同上 | 10,000 | 240 / 18,960 | 297,539 + Q 3,777 | 同上 | 末 5 轮 | fit-median（每臂） | y-cal **0.122653 / 0.117406**；g-cal 0.093864 / 0.089300 | —（未读，符合协议） | 未读 | gate.json / main_table.csv |
| 7 | `SUM_Q` → `B`（10k 部署链） | g+Q / COMP+Q | 10,000 | 同 #6 | 297,539+3,777 | 同上 | 末 5 轮 | fit-median | y-cal 0.122653 / 0.117406 | y-cal 0.092619 / 0.085107 | 已读（终轮） | terminal analysis.json |
| 8 | `H` → `C`（环通道） | prototype 精确 T25→类中位 c；unseen 由 D_full 解码 | 10,000（prototype 表）；D_full 3975 参数 | 表 + 3-fold OOF 门 | 表 + 3,975 | 分类/解码，无 y 梯度 | 冻结（H 复用 B 的 body） | b_H 复用 | y-cal 0.096694 / 0.096150 | y-cal 0.084719 / **0.081174**（raw 0.080918） | 已读（终轮，无条件报告） | REPORT.md |
| 9 | CSCL `F`（Full-Y, y-only） | raw y + label-free 重构 | **fit_inner 7,200** | 240 / 13,680 | 408,651 | 同 #1 | **monitor top-5 soup {205,211,237,238,239}** | dev cal 为 fit-only 辅助；官方数为 **raw** | **raw 0.132953** | **raw 0.114098** | 已读（本轨一次，冻结版本） | valid/test_metrics.json |
| 10 | CSCL `O_bucket` / `S_bucket` | raw y（bucket 特征） | 7,200 | 240 | 200,513 / 193,128 | 同 #1（S 另有可辨识性约束） | best-monitor | raw | 0.137888 / 0.150484 | 0.113486 / 0.125107 | 已读（同轮） | 同上 |
| 11 | Strong predictor v1（Full+triplet） | raw y + 重构 | 7,200 | 240 | 429,259 | 同 #1 | soup {187,200,226,229,240} | fit-only | 未读 | 未读 | 未读 | train json |
| 12 | Screen v1 XGB-A / XGB-B（CPU） | raw y | 7,200（fit_inner） | 500 树 × depth6（固定） | — | L2 | 无 early stop | 无 | 未读 | 未读 | 未读 | analysis_s0.json |

**读表注意**

* 可比的"同口径"对：`F`(raw 0.132953/0.114098) vs `Y@10k`(raw 0.118156/0.095616) ——
  同代码路径、同架构、同超参、同重构辅助、同 epoch 数；差异仅训练行数/随之步数/全量 refit/soup 规则。
* `SUM/COMP` 内部 dev 与官方 valid 不可直接比（fresh fold dev 与官方 valid 是不同划分、不同标签常量）。
* #6/#7（ksvd fresh fold）与 #9（CSCL dev）的 dev 也不同源：CSCL dev 是 2,000 行（fit_inner 7,200 的补集的一部分），ksvd dev 是另一划分。

---

## §3 表 2：标签来源与监督边界表（Phase B 核心）

来源类别：**G**=图输入可算；**T**=主任务标签直接给出；**A**=无标签辅助（自监督/重构，不需要任何标签）；
**D**=目标衍生辅助（由 y 经固定反演算法得到）；**O**=oracle（推理期使用真值）。

| 量 | 定义（逐字来自源码/产物） | 算法与输入 | 图不变？ | 置换稳定？ | 训练期角色 | 推理期角色 | 类别 | 可部署（只用结构）？ | 证据 |
|---|---|---|---|---|---|---|---|---|---|
| `y` | GVAE 上游 `targets = logP_norm + SA_norm + cycle_norm`（全数据集 mean/std 归一化） | `generate_latent_features_and_targets.py` 生成并落盘 | — | — | 主监督 | 评估目标 | **T** | — | generate_latent_features_and_targets.py |
| `logP` | `Descriptors.MolLogP(canonical mol)` | RDKit Crippen，canonical SMILES→mol | 是 | 是 | COMP 的 ell 原料 | 不使用 | G | 是（原子贡献可分解） | 同上 |
| `SA` | `-sascorer.calculateScore(mol)` | RDKit sascorer + fpscores 表 | 是 | 是 | COMP 的 s 参考 | 不使用 | G | 是 | 同上 |
| 环惩罚 `eff` | `-max(0, max(len(c) for c in nx.cycle_basis(G)) - 6)` | **canonical SMILES 的 RDKit 原子序**（`MolFromSmiles(MolToSmiles(...))`）上的 cycle_basis | **否**（函数值依原子序） | **否**（655 行测试 94.8% 稳定；5.2% 行给出不同 max 基环长，例 train:1760 6..13） | 经 k 进入 c | 不使用 | D | 否（需标签或历史序） | zinc_long_cycle_audit / permutation_sensitivity.csv |
| `ell` | `(logP − MU_LOGP)/sigma_logP`，MU_LOGP=2.4570953396190123 **固定** | 由 logP + 常量 | 是 | 是 | COMP 分量目标 | 不直接部署 | G | 是 | component_labels.json |
| `c` | `(k − mu_cycle)/sigma_cycle` | **由 y 反推的 k** + refit 常量 | 否（继承 k） | 否 | Q/D/prototype 目标；g 的定义 | **由 Q/D/表预测 ĉ**（C 链已核实 `inference_reads_true_c:false`） | **D** | ĉ 可部署；c 本身不可 | calibration.json / deploy_wrapper |
| `k` | `clip(round(eff), −20, 0)`（反演拟合：OLS λ 搜索 + snap） | y 与 logP/SA/cycle 三元回归，refit 于 fit 行 | 否（继承 eff） | 否 | 离散级别目标；原型类标签 | ĉ 解码为级别；原型表键 | **D** | 级别预测可部署 | stage_refine.json（12,000 行 19 处与 canonical-SMILES 重算不一致） |
| `g` | `y − c` | 减法恒等式（max|err| 4.4e-16） | 否（继承 c） | 否 | 主体监督目标 | 部署预测 ĝ + ĉ | **D** | ĝ 可部署 | full_train_targets.npz |
| `s` | `g − ell` = s_SA + (eff−k)/sigma_cycle | 减法 | 否（含 snap 残差） | 否 | COMP 分量目标 | 不直接部署 | **D**（≈G+snap） | 部分（≈归一化 SA） | 同上 |
| `s_SA` | `(SA − mu_SA)/sigma_SA` | SA + refit 常量 | 是 | 是 | 仅解释参考 | 不使用 | G | 是 | 同上 |
| `epsilon` | `s − s_SA` | snap 残差 | 否 | 否 | 诊断 | 诊断 | D | — | ε-MAE 0.002662（k=0 0.002418），max 0.370362 |
| `topology25` (`T25`) | Full 模型真实输入的前 25 维拓扑块 | 静态图统计（label-free） | 是 | 缓存构建验证 3-置换逐位一致（mcb 口径） | Q/D 输入 | ĉ 预测输入 | G | 是 | topology25_audit.json |

**k 分布（full-train 10k，本轮复算）**：`{0: 9628, −1: 325, −2: 40, −3 及以下: 7}`（负 k 共 372 行 = 3.72%）；
**c 在 k=0 的 9628 行上是同一常数**（std=0，0.0005021），即 c 的"信息"完全集中在 372 行上。
`c` 块对 var(y) 的贡献（var(c)+2cov(g,c)）= **26.4%**（train 10k）。

**五类监督来源的边界判定**

1. **图输入**（G）：logP、SA、s_SA、T25 及其它静态结构特征 —— 完全由分子图决定，可合法用于任何一轮。
2. **主任务**（T）：raw y —— 当前 CSCL 合同的唯一任务监督。
3. **无标签辅助**（A）：Full 的 label-free 结构重构（H1 通道，λ=33.9587）—— 不需要任何标签，已披露，CSCL 已沿用。
4. **目标衍生辅助**（D）：g、ell（ell 本身是 G，但作为"分量监督"进入训练的**分解**是 D）、c、k、s ——
   其定义依赖历史标签生成过程（或 y 本身）。**COMP = A + D 的组合**：把 y 的已知生成分解直接交给模型。
   在 CSCL 现行合同（只有 raw y）下，D 类监督属于**信息条件改变**，不是同题对照。
5. **oracle**（O）：真值 c 在推理/评估期加回（ksvd O-arm；prototype B-oracle marker）—— 只作诊断，
   部署链已核实不使用。

---

## §4 表 3：匹配对照证据表（Phase C/D；gain = 参照 MAE − 候选 MAE，正=改善）

| # | 对照 | 差异（唯一变化） | 端点 | 点估计 | 95% CI | 门限/结论 | 来源 |
|---|---|---|---|---:|---|---|---|
| 1 | `COMP − SUM`（8k fresh） | 仅损失项（同 init/参数量/调度/soup/校准） | dev g-cal, G0 | +0.009974 | [+0.006459, +0.013493] | 3/3 通过 → `COMPONENT_SUPERVISION_CANDIDATE` | gate.json |
| 1b | 同上 | 同上 | dev g-cal, overall | +0.009873 | [+0.0064, +0.0136] | 同上 | gains.json |
| 2 | `COMP − SUM`（10k 确认） | 仅损失项；训练 10k；**official valid** | valid y-cal, overall | +0.005247 | [+0.000499, +0.010275] | CHEM+DEPLOY 双通过 | gate.json |
| 2b | 同上 | 同上 | valid g-cal G0 | +0.005354 | [+0.000765, +0.010104] | G0_cal≥0.003 ✓ | gains.json |
| 2c | 同上 | 同上 | valid g-cal, overall | +0.004563 | [−0.000583, +0.009310] | 边缘（跨 0） | gains.json |
| 3 | `B − SUM_Q`（10k 部署 y） | 部署链上分量监督的最终收益 | **test** y-cal | +0.007512 | [+0.001810, +0.012664] | 正向 | terminal REPORT |
| 3b | 同上 | 同上 | valid y-cal | +0.005247 | [+0.000133, +0.009954] | 正向 | 同上 |
| 4 | `P − Y`（Full 10k, raw-y→g+Q） | 目标 de-cycle + Q 头；同骨架同预算 | test y-cal | +0.003768 | per-seed +0.003263/+0.004272；valid 门限 **FAIL** | 方向 test 正 / valid 负 → 不稳 | analysis_summary |
| 5 | `H − B`（prototype 环路由） | 环通道换成精确 T25 表 | valid y-cal | +0.020712 | [−0.000233, +0.062382] | **单行主导**（valid:0172 占正和 98.0%；剔除后 ≈ −3e-5）→ 门限 FAIL | prototype gate.json |
| 6 | `C − H`（离散级别解码 D_full） | unseen T25 由连续回归→级别分类 | test y-cal | +0.003545 | [+0.000153, +0.009389] | 20/20 unseen 行级别正确 | terminal REPORT |
| 6b | 同上 | 同上 | valid y-cal | +0.000544 | [−0.014359, +0.015802] | 单行主导（valid:0935 占 95.1%） | 同上 |
| 7 | `O-arm − Y`（oracle c） | 推理期加真值 c（**不可部署**） | dev cal（8k） | +0.022697（seed 0 +0.024053 / seed 1 +0.021341） | — | 78.7% 来自 9 个 severe 行；G0 仅 +0.00360 [+0.00074,+0.00658] | analysis_summary / REPORT |
| 8 | `P − Y`（8k 冻结环头） | Q(T25) 学习头部署 | dev cal（fresh 2k） | +0.010860（per-seed +0.012500/+0.009220） | — | oracle 保留率 0.4758 | DECISION.md |
| 9 | **quasi-matched 预算对照**：`Y@10k` vs CSCL `F` | 同代码/同架构/同超参/同辅助/同 epoch；差 7.2k vs 10k rows + 步数 + 全量 refit + soup 规则 | valid raw | **+0.014797**（0.132953→0.118156） | 未配对 bootstrap | **非受控**（soup 差 + 全量 refit） | 两轮 JSON |
| 9b | 同上 | 同上 | test raw | **+0.018482**（0.114098→0.095616） | 同上 | 同上 | 同上 |
| 10 | CSCL 骨架对照 `O_bucket − F` | 不同读出，同 7.2k | test raw | −0.000613 | [−0.00785, +0.00711] | 等价（CI 在 [−0.01,0.01] 内） | test_metrics.json |
| 11 | CSCL `S_bucket − O_bucket` | 结构化 vs 不透明读出 | test raw | +0.011622 | [+0.004028, +0.018968] | 结构代价真实存在 | 同上 |

**表 3 的读法**：唯一严格匹配（同 init、同参数、仅损失不同）的对照是 #1/#2/#3；
唯一 quasi-matched 的"预算"对照是 #9（代码级同一路径，但 rows/steps/refit/soup 同时变化）；
**不存在**把 (数据量) × (监督) 同时受控的实验——这正是 §8 的核心缺口。

---

## §5 Phase A：性能证据的可靠性与口径

1. **raw vs cal 差量级**：`Y@10k` valid 0.118156→0.116635（−0.0015）、test 0.095616→0.093323（−0.0023）；
   CSCL 官方数是 raw（F valid 0.132953 / test 0.114098），其 dev cal 0.11452 只是辅助。
   跨系统比较必须选同一口径：本文凡"历史 vs CSCL"均用 raw。
2. **文档—产物不一致（必须披露）**：`tracks/ksdv/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1/`
   的 `SUMMARY.md`/`REPORT.md` 表格写 SUM y_cal 0.1169 / COMP 0.1122；
   而产物（`tracks/ksvd/results/.../main_table.csv`、`gate.json`、`valid_frozen_predictions.npz` 及终轮复算）
   为 **0.122653 / 0.117406**。SUMMARY 正文与表格自身也互相矛盾（正文写 0.1227/0.1174）。
   本审计**以产物为准**；文档中的 0.1169/0.1122 疑似早期草稿残留（不得引用）。
   注意：任务提示引用的"COMP 0.1122"即源自该过期文档。
3. **dev↔official 落差同号**：CSCL 三臂 dev→valid +0.0165~+0.0235、dev→test −0.002~+0.002；
   ksvd 链 valid→test −0.012~−0.032（valid 更难，含极端环行）。**dev 数字不能当选模/选路线证据**。
4. **test 已多次访问**：CSCL 轨道 2026-10-11 一次性正式读取 valid+test（`test_access_record.json`，
   `project_wide_pristine=false`，历史 KSVD 已读过 test）；ksvd 链在其各轮 `heldout_access.json` 也各自读过。
   ⇒ 任何后续"test 提升"都不能作为独立确认；官方 test 对**新配置**而言已实质耗尽。
5. **分层证据（本轮复算，`full_y_dev_k_strata.csv`）**：CSCL F dev（2,000 行）cal 后
   k=0 (n=1930) 0.103613；k=−1 (n=63) 0.367526；k≤−2 (n=7) 0.843198。
   3.5% 的 k<0 行占 **12.7%** 的误差质量；k≤−2 的 7 行误差中位数 0.369（p90 1.94）——**环尾巴主导失败模式**。
6. **训练曲线**（CSCL F）：fit 0.9065→0.0652（ep240 仍下降），monitor 0.5709→0.12263（best@238），
   ep≥200 monitor 全在 0.1226–0.1490 → fit–泛化 gap ≈0.058（同轮 fit 0.0652 vs soup dev 0.1164）。
   表示/泛化瓶颈，而非优化未收敛。

---

## §6 Phase B：监督边界判断（结论）

1. **可部署性成立**：C 部署链的 `inference_reads_true_c=false` 已在 runner 中核实（见 §3 表 c 行）；
   prototype 表的键是 train 行的精确 T25 向量（图不变），D_full 只吃 T25。
2. **训练期信息条件已改变**：分量监督把 y 的生成分解（ell、s、c、k）交给模型。
   这些量要么由 y 反推（c、k、g、s 的 snap 部分），要么由与 y 同源的描述符得到（ell≈归一化 logP、s≈归一化 SA）。
   因此"COMP 更好"不能读成"结构表示更好"，只能读成"**已知分解的任务感知监督更好**"。
3. **`s` 不是纯 SA**（ε-MAE 0.002662，max 0.370362；ε 随环稀有度增长），
   任何把 s 直接解释成 SA 增益的表述应降级。
4. **k 的来源脆弱**：由 y 反演 + snap（19/12,000 与独立重算不一致）；置换不稳定（5.2% 行）；
   同一批 T25 精确类中 5 类带冲突标签（151 类为 singleton）。
   ⇒ "c 预测"部分是在拟合一个**由标签定义、非图不变**的量；其可迁移性本身存疑（§10 Q7）。

---

## §7 Phase C：C 的 0.081 是怎么来的（受控链分解）

C 的 test y_cal 0.081174（raw 0.080918）≈ 其 g_cal 0.081184（环通道残差 1.02e-05）。
所以 **"C 的 0.081"本质是 body 的 0.0812**，其余全部是通道记账。body 的改善可拆成：

* `Y@10k`(Full, raw-y, 10k) test raw 0.095616 → `SUM_Q`(M, g, 10k) test raw 0.091876：
  同预算、换骨架+去环目标，**≈+0.0037 raw（无配对 CI，跨骨架不可配对）**。
* `SUM_Q` → `B`（加分量监督，严格匹配）：**+0.007512 test / +0.005247 valid**（CI 均不含 0）。
* `B` → `H`（环原型路由）：test +0.000388 [−0.000262,+0.001205]（无效），valid +0.0207（单行）。
* `B` → `C`（离散级别解码，替代连续 Q）：**+0.003545 test** [0.000153,0.009389]，
  来自 20 个 unseen T25 行（20/20 级别正确）；valid 增益不显著且单行主导。
* oracle 上限（`O-arm` vs Y@8k）：dev +0.0227，其中 78.7% 是 9 个 severe 行；
  可分性上线仅为 G0 +0.0036——**环通道的"可部署上限"本来就很小**，C 已把它用尽。

**结论**：C test y_cal 0.081174 vs `Y@10k` test y_cal 0.093323 → **+0.012149**（raw 口径 0.080918 vs 0.095616 → +0.014698），
其中约 0.0075 是分量监督、约 0.0035 是离散级别解码、其余为骨架/目标口径差。
**C 的 valid 优势（相对 Y@10k +0.0205）几乎全部来自 valid:0172 一行**，不能作为可复制增益。

---

## §8 Phase D：7200 vs 10000 —— 因果证据缺口（核心）

### 8.1 存在的 quasi-matched 对照（代码级同源）

CSCL `F` 的驱动 `code/run_prediction_feasibility_v2.py` **就是**历史 Full-Y 的计算路径
（import 同一批 `tracks/ksvd/experiments/luyin16/e2e_dictenv_*` 模块，同一 H1_LAMBDA=33.95873017865987，
同一 optimizer/epoch/batch/架构 408,651），仅把**所有可拟合对象 refit 到 fit_inner 7,200**。
历史 `Y@10k` 用同一路径在 10,000 行上 fit。两者在官方 valid/test 上的 raw 差：

| 端点 | CSCL F (7.2k) | Y@10k | 差（10k − 7.2k） |
|---|---:|---:|---:|
| valid raw | 0.132953 | 0.118156 | **+0.014797** |
| test raw | 0.114098 | 0.095616 | **+0.018482** |
| steps | 13,680 | 18,960 | +5,280 |
| soup | monitor top-5 | 末 5 轮 | 规则不同 |
| 拟合对象 refit 范围 | 7,200 | 10,000 | 全量 refit |
| 训练 shuffle 流 | 7,200 行 | 10,000 行 | 行集合不同 |

### 8.2 判定：**无法严格估计数据量效应**

* 上表四项差异**同时**变化（行数、步数、refit 范围、soup 规则），且历史 Y 的超参/架构曾以 official valid
  为长期头条指标迭代（继承 valid 反馈），因此 +0.0148/+0.0185 只能读作
  "**与预算假设一致的 quasi-matched 点估计**"，不能读作"数据量效应=+0.018"。
* 缺的对照是：**同代码、同超参、同 soup 规则、同评估的一次 7.2k vs 10k A/B**（两臂都要新跑，
  因为 soup 规则必须统一）。这在现有产物中**不存在**。
* 反方向证据：`P@10k` vs `Y@10k` 的 valid 门限 FAIL、环通道在 valid 的单行主导，
  说明有效样本量增加并未自动改善 valid；而 CSCL `F` 的 dev↔valid 落差在全部 7.2k 臂上一致。
  ⇒ "数据量是主因"与"历史 recipe/soup/refit 差异是主因"两种解释都未被排除。

**因此：数据量对 Full-Y 的独立因果收益在本轮证据下不可估计（无法给出可信 CI）。**

---

## §9 Phase E：路线定位（A = y-only 继续，B = 转分量监督合同）

| 维度 | 路线 A（y-only 继续） | 路线 B（分量监督合同） |
|---|---|---|
| 合同纯度 | 高：raw y + label-free 辅助，任何目标通用 | 低：监督来自 ZINC 目标的已知分解（D 类） |
| 已验证增益 | quasi-matched 预算 +0.0148/+0.0185（未受控） | 严格匹配 +0.0050(valid y-cal) / +0.0075(test)（CI 含 0 之外） |
| 与用户目标（valid 0.08–0.09 / test 0.06–0.07） | 未达；但历史 Y@10k 已 test 0.0933，是最近的非 C 点 | 也未达（C valid 0.0962 / test 0.0809） |
| 结构贡献研究兼容性 | 高（预测器可解释性不被标签公式污染） | 低（模型已知道 y=norm(logP)+norm(SA)+norm(cycle)，后续"结构贡献"解释会与标签公式混同） |
| 可迁移性 | 高（通用 y-only 方法学） | 低（须每任务重新设计分解；ZINC 上几乎是给答案） |
| 继续成本 | 1 次训练（预算对照）+ 后续架构工作 | 新预注册 + 至少 2 臂（M 骨架上的 SUM/COMP 已验证，需在目标骨架上重做） |
| 主要风险 | 预算效应实为 recipe/soup 差异，投入后无收益 | 增益太小（+0.005~0.0075）不足以改变目标达成，且合同变窄 |

**定位**：在当前证据下，B 的严格受控增益（≈+0.005–0.0075）**小于** A 侧的 quasi-matched 预算效应（≈+0.015–0.018），
且 B 会污染后续"结构贡献"科学叙事；B 的真正前置条件是"先证明预算/recipe 解释不了差距"。

---

## §10 十个问题的直接回答

1. **历史 C 的 ~0.081 test MAE 主机制？**
   主体是分量监督的 M body（g_cal 0.0812），其中约 +0.0075 来自严格匹配的分量监督、+0.0035 来自离散级别环解码；
   环通道本身在 test 已被解到 c-MAE 1e-5，body 误差即全部剩余误差（§7）。**不是**校准（量级 1e-3），
   **不是**环惩罚拟合（其 oracle 可部署上限仅 +0.0036 G0）。
2. **额外分量监督的受控增益？**
   dev g-cal +0.009974 [+0.006459,+0.013493]（8k fresh，仅损失不同）；valid y-cal +0.005247 [+0.000499,+0.010275]；
   valid g G0 +0.005354 [+0.000765,+0.010104]；test y-cal +0.007512 [+0.001810,+0.012664]。
   无 shuffle-component / 无标签辅助对照 ⇒ 不能区分"化学信息"与"通用多任务正则"（§4 #1–#3）。
3. **Full-Y vs C 的差距能归因到数据量吗？**
   **不能严格归因**。有 quasi-matched 对照（代码级同源）给出 +0.0148(valid)/+0.0185(test) 的点估计，
   但 rows/steps/refit/soup 同时变化、且历史 recipe 继承 official-valid 反馈；缺"同代码同超参同 soup 的 7.2k vs 10k"对照。
   ⇒ 独立因果收益**无法估计**（§8）。
4. **g/ell/s/c/k 的真实来源与监督角色？**
   见 §3 表 2：logP/SA 为图可算；`ell` 是归一化 logP；`s = g−ell` ≈归一化 SA（含 snap 残差 ε-MAE 0.0027）；
   `k` 由 y 反推（round+clip，19/12,000 与独立重算不一致）；`c = (k−mu_cycle)/sigma_cycle` 由 k 得到；
   `g = y−c`。训练期：g 是主体目标、ell/s 是 COMP 分量目标、c/k 是 Q/D/原型目标；
   推理期：只用结构预测 ĉ/k̂（部署链不读真值 c）。
5. **哪些环信息是合法可算的，哪些是目标衍生/oracle？**
   合法（图不变、可独立复算）：环计数/长度直方图、`nx.minimum_cycle_basis` 汇总（3-置换稳定）、
   最长简单环、cyclomatic number、RDKit 环大小。**但**这些统计量 **不等于**标签的 `eff/k`；
   标签的 k 依赖历史流水线的 cycle_basis 原子序（94.8% 稳定 / 5.2% 不稳定），属目标衍生（D）。
   oracle 只在 O-arm（真值 c 加回）与 prototype 的 B-oracle marker 出现；可部署链无 oracle。
6. **分量监督是否改变了 y-only 基准的信息条件？**
   是。y-only 合同只给标量监督；分量监督把 y 的已知生成因子（含由 y 反推的量与同源描述符）交给模型。
   它不是 test 泄漏（推理只用结构），但**改变了信息条件**，与 raw-y 基线不是同一任务对照（§6.2）。
7. **这个优势能迁移吗？**
   具体分解（norm(logP)+norm(SA)+norm(cycle)）是 ZINC 目标定义特有的；`k` 的反演还依赖该数据集的
   描述符/常量/原子序。可迁移的只有"用廉价、任务相关的辅助监督/多任务先验"这一**方法形态**，
   且必须逐任务重设计；不能把 +0.005~0.0075 宣称为"结构表示质量"（§6）。
8. **最值得保留的非 GNN 预测路线？**
   keep 清单：(a) Full/M 非 GNN 骨架 + raw-y 合同（通用、可解释、已到 test 0.0933/0.0956 的历史水平）；
   (b) label-free 结构重构辅助（已披露、CSCL 已沿用）；(c) fit-only 纪律与冻结评估纪律。
   分量监督线作为**已封闭的诊断结果**保留（可引用其 +0.005~0.0075 与标签边界结论），不作为主路线。
9. **下一单一核心假设 + 最少实验臂？**
   核心假设（H-budget）："CSCL y-only Full 的泛化差距主要由训练预算（行数+步数+全量 refit）而非监督或表示决定。"
   最少 2 臂、1 seed：Arm A = 7,200 行、Arm B = 10,000 行，**统一 soup 规则（末 5 轮，去掉 monitor 选择）**，
   其余（代码/超参/重构 λ/评估/校准）逐字相同；主指标 = official valid raw y-MAE 一次读取（重用披露），
   副指标 = 两臂 fit 曲线 + 7.2k 臂的 dev（10k 臂无内部 held-out，这是设计代价）。预注册草案见 decision 文档。
10. **A/B/C 建议及推翻条件？**
    建议 **C（证据不足以改变合同）**：继续 y-only 合同，先买"匹配预算对照"这一唯一缺失证据。
    推翻条件：若预算对照显示效应 ≤0.003（valid/internal），则 y-only 侧被表示瓶颈卡住，
    **B 的 +0.005~0.0075 变为相对更好的选择**（需接受任务感知监督合同并重写叙事）；
    若 ≥0.010，则 A 明确胜出，下一笔继续买预算/架构；若落在 0.003–0.010 之间，按预注册的次级判据
    （test 不可用于选择）在 internal/valid 上做 A/B 权衡（详见 decision 文档表 4）。

---

## §11 证据缺口清单（本轮不能回答的问题）

1. **数据量因果**：无受控 7.2k vs 10k（§8）。
2. **架构 vs 监督**：M 骨架没有 raw-y 臂；Full 骨架没有分量监督臂；两骨架从未同预算对比。
3. **分量监督的"信息" vs "正则"**：无 shuffle-component 或随机辅助对照。
4. **环通道的可部署门限**：prototype 在 valid 上 FAIL，C 的 test 增益靠 20 行 unseen；没有独立 valid 支持。
5. **c/k 的可迁移性**：标签 c 非图不变、T25 类冲突（5/355）、singleton 类 151 个；异分布下无证据。
6. **文档可信度**：ksdv 报告表格与产物不一致（§5.2）——历史引用应一律回产物核对。

## §12 复算与产物索引

| 产物 | 内容 |
|---|---|
| `code/audit_strong_prediction_route_v1.py` | 本轮全部只读复算（k 分层、方差/协方差、标签恒等式、置换稳定性汇总、证据索引生成） |
| `results/strong_prediction_route_audit_v1/full_y_dev_k_strata.csv` | CSCL F dev 按 k 分层的 raw/cal MAE、误差质量份额、分位数 |
| `results/strong_prediction_route_audit_v1/evidence_index.json` | 引用数字与其来源路径的机器可读索引（含 ksvd 各轮 gate/gains/analysis 原文） |

引用到的既有证据文件（只读）：
`tracks/structure-contribution/results/cscl_v3d_official_eval/{valid_metrics,test_metrics,test_access_record,freeze_manifest}.json`；
`tracks/structure-contribution/results/feasibility_v2_gpu/train_Full-Y-fit7200-seed0.json`；
`tracks/ksvd/results/{zinc_full_decomposition_valid_test_confirmation_v1, zinc_chemistry_component_supervision_seed0_v1, zinc_component_supervision_fulltrain_confirmation_seed0_v1, zinc_cycle_level_transfer_terminal_test_seed0_v1, zinc_cycle_prototype_transfer_cpu_v1, zinc_full_cycle_target_decomposition_v1, zinc_frozen_chemistry_learned_cycle_v1, zinc_local_tuple_fresh_fold_replication_seed0_v1, zinc_long_cycle_audit}/*`；
上游源码 `tracks/ksvd/results/zinc_long_cycle_audit/upstream/generate_latent_features_and_targets.py`。
