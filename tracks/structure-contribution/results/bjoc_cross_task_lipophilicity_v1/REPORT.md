# REPORT — `cscl-bjoc-cross-task-lipophilicity-v1`

BJOC 一次性结构编码（Atom→Bond→Junction，Null composer，单次前向，无 GNN/attention/
迭代消息传递）在 Masked Structural SSL 初始化下，从 ZINC 迁移到 **MoleculeNet
Lipophilicity**（DeepChem 2.8.0 确定性 scaffold split 3360/420/420，单任务 logD 回归）。

* Track: `tracks/structure-contribution`
* 预注册: `configs/bjoc_cross_task_lipophilicity_v1.yaml` + `notes/bjoc_cross_task_lipophilicity_v1.md`
* 冻结执行 commit: `b56362243cb1`（remote `res`，a100-2）；终局读取 commit `27f9711cdd02`
* 预训练前证据: `protocol_freeze.json`（config sha `426b4891…`，driver `d9f6a92c…`、
  adapter `a0b10816…`、composer `1c83a97e…`、objects `8f4ba44d…`）
* CPU 门禁: **21/21 PASS**（remote `res` 与本地各一次，`cpu_gates.json` / `cpu_gates_remote.json`）
* GPU 支出: **1.7842 / 4.0 GPU-h = 44.6%**（`gpu_budget_ledger.json`）
* 终局结论: **`CROSS_TASK_SSL_CONFIRMED`** + **`COMPETITIVE_WITH_STRONG_BASELINES`（仅对本协议内
  ECFP 基线）**；**不主张 SOTA**（无同协议公开结果可比）

---

## A. 一句话结论

在完全相同的 split 索引、逐分子标签、raw-scale 度量、单模型、无 validation 选模、
无外部分子参与的前提下，**Arm P（train-only masked SSL 编码器初始化）在 5/5 个种子上
一致优于参数量完全相同的 Arm S（scratch）**：

| 区域 | Arm S（5 seeds 均值） | Arm P（5 seeds 均值） | Δ（P−S） | 95% paired bootstrap CI | 相对下降 |
|---|---:|---:|---:|---|---:|
| valid (420) | RMSE **0.80261** ± 0.02184 / MAE 0.60856 | RMSE **0.71351** ± 0.01441 / MAE 0.52853 | **−0.08904** | [−0.13291, −0.04270] | 11.10% |
| **test (420，一次性读取)** | RMSE **0.77511** ± 0.02781 / MAE 0.57682 | RMSE **0.73444** ± 0.01667 / MAE 0.54328 | **−0.04056** | [−0.08053, −0.00057] | 5.23% |

方向在 5/5 个种子上一致（valid 与 test 均如此），MAE 同步改善，改善不是单一离群点
驱动（valid top-1 分子贡献 4.5% 的改善量），gate 的 8 项条件全部满足 → 冻结决策
`TRANSFER_SSL_GAIN_PROMISING`，随后按预注册路径执行 Stage G（确认种子 2/3/4）与
Stage H（终局一次读取）。

---

## 1. 问题与设计

**问题**：ZINC 上得到的最佳配方（BJOC-Null + 掩码结构自监督初始化 + R3）能否跨任务
迁移到 Lipophilicity；以及在本协议下是否具备 SOTA 竞争力。

**两个目标**
* A（机制/迁移）：masked SSL 的**初始化**收益是否跨任务存在；
* B（可行性）：在匹配 split / 度量 / 监督预算 / 单模型下，与强基线（ECFP4+RF/XGB，
  以及文献中的 GNN）差距如何。

**两个主臂**（推理参数量完全相同，逐 seed 共享同一初始权重）
* Arm S = BJOC-Lipo-Scratch；
* Arm P = BJOC-Lipo-SSL（同一初始权重，仅编码器换成该 seed 的 masked SSL 结果）。

**明确不做**：GNN/attention/Transformer/learned iterative message passing；随机 split
搜索；验证集选模；第二套 mask ratio/backbone/损失；测试后任何选择或调参。

---

## 2. 数据协议、复现与标签隔离

| 项目 | 值 / 证据 |
|---|---|
| 数据集 | MoleculeNet `Lipophilicity`（ChEMBL 公开沉积，DeepChem 重分发） |
| 下载 | 282,606 bytes，sha256 `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`；4200 行，列 `CMPD_CHEMBLID,exp,smiles` |
| RDKit 解析 | 4200/4200 成功，0 失败；0 规范化 SMILES 重复；1 个多组分分子（row 1646，保留） |
| Split | DeepChem **2.8.0** `ScaffoldSplitter`（Bemis–Murcko，确定性无 RNG），隔离 venv 直接执行上游代码 |
| 上游核验 | `splitters.py` sha256 `5adf9a5c…`（与 wheel RECORD 一致）；`data_loader.py` `c71329c8…`；`lipo_datasets.py` `f16dde9d…`（`upstream/UPSTREAM_PROVENANCE.md`） |
| 尺寸 | **3360 / 420 / 420**，与 DeepChem 2.8.0 参考 **逐索引一致**（`reference_match: true`） |
| 结构泄漏 | 跨集合 scaffold 组 **0**、规范化 SMILES **0**；2408 个 scaffold 组 |
| 行冻结 | train/valid/test 行号 sha256 `76a4112a…` / `ee39ed55…` / `f975ef0c…`；结构/行清单 `row_manifest.json` |
| 标准化 | **train-only**（mean 2.1629 / std 1.2110，`standardizer_v1.json`，gate `train_only_standardisation` 复核 n_fit=3360） |
| 标签隔离 | train 标签来自冻结 split；valid/test 标签只存在于本地 vault `data/lipophilicity/labels.npy`；训练/预测命令在 vault 存在时**硬拒绝运行**（`assert_score_labels_absent`，4 个调用点，gate `score_labels_blocked_for_training` 逐行复核 raw 数据集无标签列） |
| 远端隔离 | 远端 `res` 从未出现 vault（gate 记录 `vault_present: False`）；所有训练产物标注 `labels_used: "train rows raw y only"`、`official_valid_loaded: False`、`official_test_loaded: False` |
| 读取纪律 | valid 标签：Stage F 一次（2-seed 决策）+ Stage G 一次（预注册的 5-seed 重算，两次均有 lock 记录）；test 标签：**全程仅 1 次**（所有 10 个 test 预测文件先 SHA-256 冻结，`artifact_manifest_pre_test.json` 记录 57 个模型/预测/决策文件的哈希后，才执行 `score-test`） |

---

## 3. 冻结的模型、配方与开销

**模型**（BJOC-Lipo-Null，与 ZINC 回合同架构，仅输入宽度/pretext 词表按新数据集重算）

| 模块 | 参数量 | 说明 |
|---|---:|---|
| atom_encoder | 25,184 | 15 类原子（train-only 词表）+ UNK/MASK |
| bond_encoder | 64,096 | 6 类键槽 + 立体 |
| junction_encoder | 116,416 | ring/bond/junction 交叉特征 |
| group_composer | 64,256 | Null 分组（无学习式迭代） |
| reader | 149,761 | 全图读出 + 回归头 |
| **合计（推理）** | **419,713** | exact（gate `parameter_budget_exact`；ZINC 为 413,505，+6,208 = +1.50%） |

**R3-Lipo（双臂逐 seed 完全相同的监督配方，冻结于 GPU 前）**
Adam，lr 1e-3→1e-5 per-step cosine，batch 128，wd 1e-5，grad clip 5，360 epoch
（27 step/epoch = **9,720 updates**），最后 5 个 epoch 参数 soup，无 validation 选模。

**Masked SSL（Arm P 的初始化，冻结于 GPU 前）**
仅用 train split 图结构；在**原始图（raw codes）**上以 15%/15% 掩码原子类/键类后
完整重建特征（隐藏真值替换不变性 60/60 通过）；曝光匹配规则
`U_SSL = round(96 × 3360 / 128) = 2,520` updates/seed，无外部分子、无标签。

**成本（见 `gpu_budget_ledger.json`）**

| 项 | 实际 |
|---|---:|
| masked SSL（5 seeds） | 3,617 s ≈ **1.005 GPU-h** |
| 监督 R3-Lipo（10 runs） | 2,780 s ≈ **0.772 GPU-h** |
| smoke + 两次 predict | 20 s ≈ 0.006 GPU-h |
| **GPU 合计** | **6,423 s = 1.7842 GPU-h（预算 4.0 的 44.6%）** |
| 远端 CPU（prep/gates） | 42 s；本地 CPU（RDKit ECFP 基线 47 s、门禁/评分 <1 min） | 
| 新增输入特征 | **无**（仍然是 atom/bond/junction 原始图；无外部分子、无额外描述符） |
| 预训练词表变化 | pretext 头词表 15 原子类 / 6 键槽类；推理头在训练前移除（gate `no_ssl_heads_at_inference`） |

> 记账口径：每个 rr process job 由 pool 绑定到单张 A100（`CUDA_VISIBLE_DEVICES` 单卡）；
> 并发在 gpu0/gpu1 上的两个 job 分别计数，绝不合并为 1 份。

---

## 4. CPU 门禁（GPU 前）

`cpu_gates.json`（本地，RDKit 可用）与 `cpu_gates_remote.json`（`res`，无 RDKit）
均为 **21/21 PASS**，其中与结论直接相关的：

* `raw_conversion_deterministic`：10 行黄金 fixture（首行/最小/最大/多组分/带电/手性/
  三键/UNK 映射/键立体/macrocycle）在部署图上逐字段一致，record round-trip 精确；
* `masked_hidden_truth_swap_invariance` 60/60；`masked_sample_round_trip`；
* relabel 不变性 1.5e-8、键端点交换 0.0、batch 组成 2.2e-8、save/load 重放 0.0；
* `vocab_mask_slots_disjoint`、`unseen_class_unk_fallback`、`ring_bond_junction_crosscheck`
  （200 探针 0 失败）；
* `parameter_budget_exact` 419,713；`supervised_schedule_matched` 9,720；
  `init_parity_scratch_vs_ssl`；`ssl_train_structures_only`（3,360）；
* `no_gnn_attention_iteration`（AST+module 扫描为空）、`no_zinc_access`、
  `score_labels_blocked_for_training`。

远端本来没有 RDKit（`lipo` 依赖组只装在本地），因此门禁 1 改为「黄金 fixture +
record round-trip + 有 RDKit 时才现场重解析」；全局 pytest：**24 passed**
（`tests/test_bjoc_cross_task_lipophilicity_v1.py`，RDKit-free）。

---

## 5. Stage F / G：验证集结果与冻结判定

**Stage F（seeds 0/1，预注册门槛）**：S 0.82875 vs P 0.71517，Δ=−0.11358（−13.70%），
8 项条件全部满足 → `TRANSFER_SSL_GAIN_PROMISING`，`seeds_to_confirm=[2,3,4]`，
`test_authorized=true`（`validation_decision.json`）。

**Stage G（5 seeds 逐 seed 表，valid）**

| seed | S RMSE | S MAE | P RMSE | P MAE | ΔRMSE | ΔMAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.83178 | 0.63531 | 0.73731 | 0.53519 | −0.09448 | −0.10012 |
| 1 | 0.82571 | 0.62139 | 0.69303 | 0.50337 | −0.13268 | −0.11802 |
| 2 | 0.78523 | 0.60058 | 0.71152 | 0.54928 | −0.07371 | −0.05130 |
| 3 | 0.79181 | 0.58720 | 0.71778 | 0.53055 | −0.07403 | −0.05665 |
| 4 | 0.77851 | 0.59831 | 0.70793 | 0.52428 | −0.07058 | −0.07402 |
| **均值** | **0.80261** | **0.60856** | **0.71351** | **0.52853** | **−0.08904** | **−0.08015** |

* RMSE 95% CI [−0.13291, −0.04270]；MAE 95% CI [−0.10805, −0.05124]（5,000 次分子级配对 bootstrap）；
* 子群诊断：低 logD 尾部分位（y ≤ train q10，n=47）Δ=−0.0796，普通行 Δ=−0.0932 →
  **不是尾部/离群点驱动**；top-1 分子贡献 4.5%、top-10 贡献 34.2%；
* 一个可复现的伴随现象：Arm P 的 train-fit 明显更好（soup fit MAE 0.0022–0.0031 vs
  S 0.0084–0.0146，约 4–5×），且 P 的种子间方差更小（0.0144 vs 0.0218）——SSL
  初始化同时改善了优化条件与稳定性（§8 讨论其对解释的影响）。

---

## 6. Stage H：终局一次读取（test）

**终局前置冻结**（`artifact_manifest_pre_test.json`，共 57 个文件 + 6 个代码文件）：
10 个 soup/10 个 pretrain 检查点、valid 预测、validation metrics/decision、baselines
全部哈希登记；随后 `predict --zone test`（远端无标签环境）生成 10 个预测文件并冻结
（`prediction_manifest_test.json`，combined `39c6cc6caec9b320`，逐文件 SHA-256），
最后才在本地执行**唯一一次** `score-test`（`test_read.lock`，y-sha 已记录）。

| seed | S RMSE | S MAE | P RMSE | P MAE | ΔRMSE | ΔMAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.77430 | 0.57964 | 0.74166 | 0.54469 | −0.03264 | −0.03495 |
| 1 | 0.78211 | 0.57918 | 0.74916 | 0.55234 | −0.03294 | −0.02683 |
| 2 | 0.81737 | 0.60126 | 0.71740 | 0.53050 | −0.09997 | −0.07075 |
| 3 | 0.77161 | 0.57499 | 0.75222 | 0.55188 | −0.01939 | −0.02312 |
| 4 | 0.73018 | 0.54905 | 0.71177 | 0.53698 | −0.01841 | −0.01207 |
| **均值 ± std** | **0.77511 ± 0.02781** | **0.57682 ± 0.01664** | **0.73444 ± 0.01667** | **0.54328 ± 0.00850** | **−0.04056** | **−0.03354** |

* RMSE 95% CI [−0.08053, −0.00057]（**上界勉强低于 0**，须如实披露；见 §8）；
* 方向一致性：5/5 seeds 为负（−0.0184 … −0.1000）；
* 分位数改善：median |err| 0.4477→0.4147，p90 1.2020→1.1208；
* 子群：test 尾部（y ≤ q10，**仅 20 个分子**）Δ=−0.0993，普通行 Δ=−0.0366 —— 在
  test 上改善更多来自难样本，但尾部样本量很小，只作描述不作结论；
* ECFP4 基线（同索引、同 train 标签、无选模）：RF test RMSE 0.87660 / MAE 0.70321，
  XGB test RMSE 0.85973 / MAE 0.68225。

**判定**：终局结论 `CROSS_TASK_SSL_CONFIRMED`（跨任务 SSL 初始化收益得到确认），
且在本协议内显著优于 ECFP 基线（P 比 ECFP4-XGB 低 14.6% RMSE，比 ECFP4-RF 低 16.2%；
S 也分别低 9.8% / 11.6%）。**不主张 SOTA**（§7）。

---

## 7. SOTA 可比性审计（严格纪律）

**检索结论（2026-10-14）**：没有找到任何公开结果在**完全相同的 DeepChem 2.8.0 确定性
scaffold split（3360/420/420）**、不使用外部分子、单模型、同监督预算下报告 Lipophilicity
RMSE。因此本轮**不做任何排名性 SOTA 主张**。

| 参照类 | 数字（已核实） | 为什么不可直接比较 |
|---|---|---|
| MoleculeNet 原文 | XGB 0.799 / GraphConv 0.655（**random split**） | 分割方式不同（random 通常更易） |
| GROVER 论文 Table 1（复现） | D-MPNN 0.653±0.046、AttentiveFP 0.650±0.030、MPNN 0.672±0.051、GraphConv 0.712±0.049 等 | 3 次随机 scaffold 8:1:1 的均值，非 DeepChem 确定性 splitter |
| GROVER（含 ~11M 外部分子预训练） | GROVERbase **0.563±0.030**、GROVERlarge 0.560±0.035、无预训练消融 0.643 | 额外数据 + 不同 split；与本轮"零外部分子"协议不同 |
| Uni-Mol / SMI-TED / Chemprop 原文 | `NOT VERIFIED` | 访问失败（arXiv 查询空 / HTTP 503）或文本中无该列，不猜测 |

**差距陈述**：与本协议唯一可比的是表 A（同协议）内的 ECFP4 基线与自身两臂；
与 GROVER 0.560 类数字之间**无法给出协议内差距**。若未来出现同协议外部复现，
本轮的 `artifact_manifest_pre_test.json` + `prediction_manifest_test.json` 可直接复核。

---

## 8. 证据充分性与局限（明确披露）

**支持的**
1. 5/5 seed 方向一致（valid + test 双区域）；两区域 paired bootstrap CI 均低于 0；
2. MAE 与 median/p90 |err| 同步改善；非离群点驱动（valid top-1 贡献 4.5%）；
3. 严格对照：S/P 逐 seed 共享同一初始权重、同一 batch 顺序/调度/soup；
4. 21/21 CPU 门禁 + 24 个单元测试 + freeze 严格绑定代码/数据哈希；
5. 标签隔离可审计（训练产物三标记 + vault 硬拒绝 + 远端无 vault）；
6. test 只读一次，10 个预测文件先哈希冻结。

**局限 / 反方证据**
1. **test CI 上界 −0.00057 非常接近 0**：test 增益（5.23%）明显小于 valid（11.10%），
   说明 valid 上的 11% 有乐观偏差（valid 被读两次，虽为预注册路径）；
2. **test 尾部子群只有 20 个分子**，test 的改善结构（尾部 Δ=−0.0993 vs 普通 −0.0366）
   不足以支撑"困难样本更强"的结论；
3. **单数据集、单 split**：只在 Lipophilicity 的 DeepChem scaffold 划分上验证；
4. 没有同协议的 GNN/消息传递基线（预算内未运行 Chemprop 类参照；文献数字不可比），
   因此"相对强图模型"的结论**未验证**；
5. 机制未分离：P 的 train-fit 好 4–5×、种子方差更小，收益可能部分来自**优化条件**
   而非新增结构信息（本轮无法区分）；
6. 训练本身处于插值区间（soup fit MAE ~0.002–0.015），绝对泛化水平对配方/数据规模
   可能敏感；R3-Lipo 从 ZINC 冻结而来，未在 Lipo 上做过任何配方搜索（这是纪律，也是局限）。

---

## 9. 终局结论与下一步

**结论词**
* `CROSS_TASK_SSL_CONFIRMED` — masked SSL 初始化收益跨任务（ZINC→Lipophilicity）确认；
* `COMPETITIVE_WITH_STRONG_BASELINES` — 在**相同协议内**（split/度量/单模型）优于
  ECFP4+RF 与 ECFP4+XGB；
* 明确**不是** `SOTA_SUPPORTED_UNDER_MATCHED_PROTOCOL`（无同协议外部结果），
  也不是 `SOTA_CANDIDATE_REQUIRES_EXTERNAL_VERIFICATION`（当前 0.734 的绝对水平在
  不同 split 文献背景下不足以称候选）。

**下一步建议（三选一中的判断，非证据）**
1. **首选：方法瓶颈诊断（表征 vs 优化条件）**。最小设计（预注册、约 0.8 GPU-h）：
   在同一 split 上比较三种监督路径 —— (a) Arm S；(b) Arm P（编码器可微调）；
   (c) **SSL 编码器冻结**、只训读出（representation-only）。若 (c)≈(b) 且都优于 (a)，
   说明 SSL 提供的是表征；若 (c)≈(a) 而 (b)≫(a)，则收益主要是优化再条件化，
   "结构贡献"的表述必须相应收窄。
2. 备选：ESOL 作为第二个跨任务目标（数据集小、成本低，约 0.5–1 GPU-h），用同一
   冻结协议再做一次方向一致性检验，用于检验"跨任务"而不是"跨数据集配对"的普遍性。
3. 不推荐现在停止，也不推荐直接追 SOTA：绝对水平受协议不可比限制，继续加大模型/
   外部分子会破坏"零外部数据 + 一次性结构编码"的立论。

---

## 10. 直接回答预注册的 7 个问题

1. **Lipophilicity RMSE 是多少？** 终局 test：Arm S **0.77511（±0.02781 std）**、
   Arm P **0.73444（±0.01667）**；valid：S 0.80261、P 0.71351（均为 420 分子、5 seeds
   的 seed-mean，raw-scale，单模型）。
2. **相对匹配 Scratch 的增益（2 或 5 seeds）？** 2 seeds（valid，Stage F 决策依据）：
   Δ=−0.11358（−13.70%）；5 seeds（valid）Δ=−0.08904（−11.10%），
   95% CI [−0.13291, −0.04270]；**5 seeds test** Δ=−0.04056（−5.23%），
   CI [−0.08053, −0.00057]，5/5 seed 同向。
3. **相对 ECFP / 图基线？** 同协议 ECFP4+RF test 0.87660（MAE 0.70321）、
   ECFP4+XGB test 0.85973（0.68225）；Arm P 分别低 16.2% / 14.6%，Arm S 低 11.6% / 9.8%。
   图基线（D-MPNN/AttentiveFP/GROVER 等）**没有同协议数字，不可比、未验证**。
4. **与最佳已验证同协议公开结果的差距？** 不存在同协议公开结果；最近的可核实参照是
   GROVER 0.560±0.035（3 次随机 scaffold + ~11M 外部分子预训练）与 MoleculeNet
   random-split 0.655——**差距不可计算**，本轮不据此做任何领先/落后结论。
5. **新增输入/参数/预训练/计算成本？** 新增输入特征 **0**（无外部分子、无新描述符）；
   推理参数 413,505→**419,713**（+1.50%）；预训练 2,520 updates/seed（曝光匹配，
   仅 train 结构）；GPU 合计 **1.7842 h / 4.0 h**（SSL 1.005 h + 监督 0.772 h +
   推理/冒烟 0.006 h）；远端 CPU 42 s。
6. **证据是否充分？** 对"跨任务 masked SSL 初始化收益"**充分**（5/5 seed、双区域 CI
   低于 0、MAE 同步、非离群驱动、门禁与标签隔离可审计）；对"具体机制""相对强图模型"
   "绝对 SOTA 水平"**不充分**（test CI 上界 −0.00057、test 尾部仅 20 分子、
   单数据集、无同协议外基线与机制分离实验）。
7. **下一步：ESOL vs 方法瓶颈 vs 停止？** 建议**先做方法瓶颈诊断**（SSL 编码器冻结
   的 representation-only 对照，约 0.8 GPU-h）；随后如仍有余量再做 **ESOL** 第二目标
   复现；不建议在机制未澄清前追求 SOTA，也**不建议现在停止**（已有稳定、可复现的
   跨任务收益，值得再花 <1 GPU-h 澄清机制）。

---

## 11. 操作事故与披露（不影响科学性）

1. `lipo-gates` 首次在 `res` 失败（无 RDKit）→ 改为黄金 fixture 门禁，重跑 21/21；
2. 远端 prep 重写了被 git 跟踪的 `protocol_freeze.json`/`init_parity.json`，导致
   fast-forward 部署被拒 → 用一次记录在案的 `rr run --allow-stale`（`lipo-wt-repair`）
   恢复远端工作树，随后把这类"运行期生成物"从 git 跟踪中移除（提交 `b5636224`）；
3. 一个本地 shell 串行编排被中断（无科学影响）：所有后续作业改为逐个提交 + 显式
   轮询，最终 24 个 rr 运行全部 `completed, exit 0`（1 次预期内的无 RDKit 失败除外）。

---

## 附录 A — 关键哈希（完整列表见 `artifact_manifest_pre_test.json`）

| 文件 | sha256（前 16） |
|---|---|
| `dataset/lipo_raw_v1.json.gz` | `8416837ea3e56e3e` |
| `split_lipo_v1.json` | `7f0e0b991b0a0465` |
| `protocol_freeze.json` | `（运行期生成，未跟踪）` |
| `artifact_manifest_pre_test.json` | `ea470faaa0c80ac4` |
| `prediction_manifest_valid.json` | `32a40ad7173b7a61` |
| `prediction_manifest_test.json` | `712b9cf50a038dc3` |
| `validation_metrics.json` | `5fc97f4b3e0f76a5` |
| `validation_decision.json` | `b6e8a3b70f1ec852` |
| `test_metrics.json` | `b32192551d9e0d5e` |
| `test_read.lock` | `d4159b8312b83443` |
| `gpu_budget_ledger.json` | `03fa82bc77d3d986` |
| `cpu_gates_remote.json` | `2263b04e36c5a988`（本地版 `cpu_gates.json`，21/21） |
| `gpu_smoke.json` | `160361486f542cd5` |

检查点哈希（每 seed）：`pretrain_s0..4.pt` = `03b730df/ e621b3d3/ 5c23a0ea/ cfc5a11f/ f4047360`；
`soup_S_s0..4.pt` = `ae3026ca/ 1b6ec1f1/ 17487f2b/ 0358eab6/ bf96616b`；
`soup_P_s0..4.pt` = `1a8292ac/ 505f840e/ cdcec6bf/ c3ed6dd4/ 1b94df56`（全哈希见清单）。
test 逐文件预测哈希：S0–S4 = `c615c9f2/ 9addff3c/ a40ae8c5/ 8fcaf774/ 94a94dc8`，
P0–P4 = `ab27a7c4/ 1ac1437d/ a32ebf07/ 453f522a/ 4117632f`。

## 附录 B — 运行清单（全部经 `rr`，远端 commit 见括号）

`lipo-prep`(32a81e1)/`lipo-gates`(失败，预期：无 RDKit)/`lipo-wt-repair`/`lipo-prep2`(b563622)/
`lipo-gates2`(21/21)/`lipo-smoke`/`lipo-ssl-s0..s4`/`lipo-S-s0..s4`/`lipo-P-s0..s4`/
`lipo-predict-valid`/`lipo-predict-valid-5`/`lipo-predict-test-5`(27f9711)。
所有科学运行 `exit_code=0`；每个运行的 pool、节点 `a100-2`、`cuda_visible_devices`、
GPU 名称、torch/CUDA 版本均记录在 `gpu_budget_ledger.json` 与 rr job meta 中。
