# REPORT — LCRC Predictive Strength & Traceability Benchmark v1

Round / Protocol: `lcrc-predictive-trace-benchmark-v1`
Track: `tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/`
参照 HEAD：`9628ab4ca901aeecd2957b9e2c172a9fab485f7d`
本文件只报告**本轮实际执行了什么、得到了什么**。设计定义见 `PROTOCOL.md`
（三份设计文件在任何正式 GPU 训练之前冻结，SHA-256 记在 `analysis/protocol_freeze.json`，
之后未再修改）。

> **后续勘误（2026-11，`lcrc_posthoc_diagnosis_v1`）**：本报告的三处表述已被更正：
> (1) FULL/C1 胜负计数为 **6/10 更好、4/10 更差**，不是 4/10、6/10；
> (2) 分子任务的旧 “Δ 95% CI” 实际是 **各 seed bootstrap 区间的外包络**
> (`SEED_WISE_CI_ENVELOPE`)，不是跨 seed 平均效应的 CI；修正后的跨 seed 条件
> bootstrap 在 BACE 与 MolHIV 上排除 0；
> (3) typed-S2 的 10/12 与 11/13 是 12 个本轮 checkpoint 与 13 个含历史
> checkpoint 的范围差异（按 5% 边际判据）。
> 详细数字、判据和复现代码见
> `tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1/ERRATA.md`。
> 原始分数、checkpoint、冻结协议与 `analysis/statistics.json` 均未修改。

---

## 0. 一页结论

| 假设 | 结果 | 判定 |
|---|---|---|
| **H1** 预测强度：C1（228 674）能否替代 A0（247 106） | 旧 seed-wise envelope 10/10 跨 0；修正后分子任务 4/6 跨 0（BACE、MolHIV 排除 0）；C1 在 6/10 更好、4/10 更差，方向不一致 | `NO_DETECTABLE_DIFFERENCE`（不是等价；勘误 P0-1/P0-3） |
| **H2** 结构机制：typed-S2 通道被训练权重依赖 | 10/12 个本轮 checkpoint（5% 边际判据；含历史为 11/13）中 typed-S2 是最强输入通道；C1 与 A0 同等依赖 | `TYPED_S2_DOMINANT`（固定权重敏感性，非训练后必要性） |
| **H3** 局部编辑误差能否被迹特征预测 | `Gain = +0.0015`，CI [−0.0108, +0.0114] | `NO_INCREMENT_THIS_PROBE` |
| **H4** TRACE vs NATIVE/FAMILY/GLOBAL | 与零预测器打平 | `NO_INCREMENT_THIS_PROBE` |

* **H1 的"没有差异"不等于"等价"**：CI 只说明在这个样本量下无法分辨，
  不能当作 C1 与 A0 等效的证明。
* TU 方案由**唯一允许的依据**（实测硬件成本）选出：`A_10x2`、`B_5x3` 均超预算，
  冻结为 **`C_5x2`（RESOURCE_LIMITED）**，fold_seed 20261102，
  每折外层训练集内 10% 分层 Valid（seed = 20261102 + fold_id）。
* 所有新 Test 结果标记 **`EXPOSED_TEST_EXPLORATORY`**：这些 Test 标签历史轮次已读过，
  本轮不是盲测。
* `Lipophilicity`、`ZINC` **不重训**，只读冻结历史并单独标注来源等级。

---

## 1. 执行事实

| 项 | 值 |
|---|---|
| 开始（UTC，第一条命令） | 2026-11-02T16:05Z |
| 正式训练窗口（UTC） | 2026-11-02T17:00Z → 2026-11-02T21:40Z |
| 墙钟（到最终提交） | **≈ 6.7 h**（硬上限 8 h） |
| GPU | 2 × A100-SXM4-40GB（`res-gpu0`、`res-gpu1`），每 pool 至多一个正式训练进程 |
| 正式训练单元 | 分子 **36** + TU **80** + local-edit **6** = **122** |
| 账本条目 | 128（含 6 条重复的 TU 条目，见 §8） |
| **GPU 时间（全部）** | **9.091 GPU-h**（软上限 14，硬上限 16） |
| 失败单元 | **0** |
| Test 访问事件 | 36（每个分子单元一次；未重复读取） |
| CPU 工作 | 13 个门禁、P1 回溯 3054 条迹、探针 CV/Audit、聚合与统计 |

调度：`code/ptb_queue.py` 在每个单元前读 `analysis/compute_ledger.json`，
拒绝会突破硬上限或墙钟保留时间的单元，已完成的单元跳过；
两个 pool 之后用 `--shard K/2` 做**确定性互斥划分**（按单元名哈希）。
上游 `res` 工作树的脏改动没有被丢弃：先复制到 `~/cy/rr-backups/20261102T163030Z/`
再 `git stash --include-untracked`（`stash@{0}` 仍可取回）。

---

## 2. 正确性门禁（`analysis/correctness_gates.json`，13/13 PASS）

| 门禁 | 关键数字 |
|---|---|
| 参数量与列映射 | A0 = **247 106**，C1 = **228 674**，差 **18 432**；局部输入列 529 → 433 |
| Tox21 头加宽 | 248 536 / 230 104，差仍 18 432 |
| 共享初始化 | C1 由 A0 初始 state 按冻结列映射删除，共享张量逐位相等 |
| 冻结重放 | 本仓库 A0 重实现 vs 上游冻结 Lipophilicity s0 Valid 预测，max\|Δ\| = **1.43e-6**（tol 1e-5） |
| dropout 语义 | 只在共享 `h` 上、每 forward 一次 |
| 重编号不变性 | 预测对原子重编号不变 |
| 批量/隔离 | 同批不同批一致；样本间不串扰 |
| s1/s2 路径构造 | 按冻结定义重建 |
| 梯度可达性 | 每个可训练张量都收到非零梯度（含 Tox21） |
| 归一化 | 只用训练部分拟合，Test 未被读取 |
| checkpoint 重载 | SHA 校验后重载，预测逐位一致 |

表格：`analysis/report_tables.md` §E。

---

## 3. H1 — 分子任务（36 个正式单元，全部完成）

主指标（每格 3 seeds；`Δ` = C1 − A0；回归 RMSE 越小越好，分类 ROC-AUC 越大越好）：

| dataset | metric | A0 mean ± sd | C1 mean ± sd | Δ C1−A0 | Δ seed-wise CI envelope（非平均效应 CI） |
|---|---|---|---|---|---|
| ESOL | RMSE | 0.5645 ± 0.0299 | **0.5534 ± 0.0299** | −0.0111 | [−0.0507, +0.0248] |
| FreeSolv | RMSE | 1.0922 ± 0.0606 | **1.0711 ± 0.0732** | −0.0211 | [−0.0921, +0.1017] |
| BACE | ROC-AUC | 0.8168 ± 0.0066 | **0.8263 ± 0.0050** | +0.0095 | [−0.0259, +0.0453] |
| BBBP | ROC-AUC | 0.7041 ± 0.0083 | **0.7060 ± 0.0142** | +0.0019 | [−0.0169, +0.0283] |
| Tox21 | macro ROC-AUC | 0.8353 ± 0.0104 | **0.8374 ± 0.0087** | +0.0022 | [−0.0023, +0.0153] |
| ogbg-molhiv | ROC-AUC（官方 evaluator） | **0.7716 ± 0.0055** | 0.7611 ± 0.0135 | −0.0105 | [−0.0511, +0.0093] |

* **旧口径 6/6 envelope 跨 0**：上表区间是三个 seed 各自行 bootstrap 区间的
  最小值/最大值外包络，**不能**当作跨 seed 平均 C1−A0 效应的 95% CI。
  修正后的 seed-conditional mean-difference bootstrap 见
  `lcrc_posthoc_diagnosis_v1/analysis/mean_seed_bootstrap_ci.csv`：
  4/6 分子任务跨 0，BACE 与 MolHIV 排除 0；仅条件于当前 6 个 checkpoint。
* 方向：C1 在 ESOL、FreeSolv、BACE、BBBP、Tox21 略好，在 MolHIV 明显更差
  （−0.0105，主要来自 seed 2 的 0.7491）；方向不一致 → 不支持"C1 更强"或"C1 更弱"。
* **胜负计数勘误**：逐数据集均值为 **C1 6/10 更好、4/10 更差**（不是 4/10、6/10）。
* MolHIV 上限 100 epoch，**未扩展**；6 个单元的 best epoch 为 29–53，
  **没有**触发 `BUDGET_LIMITED_POSSIBLE`（阈值 ≥90）。即便如此，
  仍不与历史 360-epoch A0 做配对比较。
* 结果标签：全部 `EXPOSED_TEST_EXPLORATORY`。

**冻结历史（只读，不重训）Lipophilicity**

| repeat | Valid RMSE A0 / C1 | Test RMSE A0 / C1 |
|---|---|---|
| 122 | 0.7066 / 0.7053（−0.0013） | 0.5770 / 0.5784（+0.0014） |
| 123 | 0.7610 / 0.7591（−0.0019） | 0.5827 / 0.5667（−0.0160） |
| 124 | 0.6373 / 0.6367（−0.0006） | 0.5391 / 0.5420（+0.0030） |

三个 repeat 上 C1 的 Valid RMSE 一致略低（−0.0006…−0.0019），
Test 差值在 −0.0160…+0.0030 之间摆动（repeat 123 是离群），
与"删掉 `b_mean` 在 Lipophilicity 上几乎无影响"的既有结论一致。
来源：`lcrc_lipo_model_dissection_v1/analysis/tables/variant_means.csv`（SHA 记录于 csv）。

---

## 4. H1 — TU 实验包（80 个单元，`C_5x2`）

方案选择只看**实测成本**（`analysis/cost_measurement.json`，20 行，A0/C1 各测一次；
逐数据集取两变体较慢值以免 CUDA 预热偏差）：

| 方案 | 单元 | 预测 GPU-h（×1.2 安全系数） | 预测墙钟 | 结论 |
|---|---|---|---|---|
| `A_10x2` | 160 | 13.32 | 13.32 h | 超预算（含 1.0 保留 → 18.87）→ 不可行 |
| `B_5x3` | 120 | 9.95 | 9.95 h | 超预算（含保留 → 15.50 > 14）→ 不可行 |
| **`C_5x2`** | **80** | **6.66** | **6.66 h** | **选中（RESOURCE_LIMITED）** |

结果（10 个单元 = 5 fold × 2 seed 每格；Δ 的 CI 以 **outer fold** 为重抽单元）：

| dataset | A0 acc | C1 acc | Δ C1−A0 | Δ 95% CI |
|---|---|---|---|---|
| NCI1 | 0.8296 ± 0.0067 | 0.8241 ± 0.0081 | −0.0055 | [−0.0113, +0.0018] |
| PROTEINS | 0.7592 ± 0.0298 | 0.7533 ± 0.0281 | −0.0059 | [−0.0180, +0.0031] |
| IMDB-BINARY | 0.7370 ± 0.0276 | 0.7260 ± 0.0322 | −0.0110 | [−0.0310, +0.0075] |
| IMDB-MULTI | 0.4810 ± 0.0262 | 0.4843 ± 0.0288 | +0.0033 | [−0.0073, +0.0130] |

* 4/4 的 CI 跨 0；C1 在 3/4 上低 0.005–0.011，IMDB-MULTI 上高 0.0033。
* 方向与分子任务一样不一致，量级与 fold 间波动（sd 0.03）同阶。
* 标签：`NEW_INTERNAL_CV`（内部交叉验证，非官方 Test）。

---

## 5. H2 — 结构机制审计（`analysis/mechanism_audit.json`）

### 5.1 只读冻结证据

从五个冻结轮次重读 104 条对比值（follow-up 勘误表优先，legacy 表只作对照），
每条带源文件与 SHA-256、split 标记。没有重训、重打分或重读 Test。

### 5.2 `FROZEN_MODEL_INPUT_INTERVENTION`（固定权重输入通道干预）

把 typed-S2 输入列（`LOCAL_LAYOUT["s2_type"]…["s2_content"]`，529 维局部块的
**320:520 共 200 列**）整体置零，固定权重前向；对照为同 checkpoint 上
`g`（全局 42 列）与 `mean_h2`（rho 64 列）置零。

| dataset | model | typed-S2 \|Δ\| | g off | mean_h2 off | Spearman(基线, off) |
|---|---|---|---|---|---|
| Lipophilicity* | A0 | **0.949** | 0.385 | 0.229 | 0.34 |
| ESOL | A0 / C1 | **0.532 / 0.544** | 0.362 / 0.335 | 0.466 / 0.467 | 0.79 / 0.69 |
| FreeSolv | A0 / C1 | **0.610 / 0.608** | 0.215 / 0.183 | 0.301 / 0.272 | 0.79 / 0.76 |
| BACE | A0 / C1 | **6.33 / 7.60** | 1.21 / 1.45 | 1.51 / 1.54 | 0.20 / 0.21 |
| BBBP | A0 / C1 | **6.84 / 6.29** | 0.59 / 0.78 | 1.39 / 1.50 | 0.42 / 0.41 |
| Tox21 | A0 / C1 | **3.23 / 2.70** | 1.33 / 1.16 | 2.39 / 2.31 | 0.29 / 0.32 |
| ogbg-molhiv | A0 / C1 | 4.18 / 4.36 | 1.23 / 1.26 | **4.31 / 4.23** | 0.15 / 0.16 |

单位：回归为 raw logD，分类为 logit。\* Lipophilicity 用上游冻结 checkpoint（A0）。

* 按 **5% 边际判据**，13 个 checkpoint 中有 11 个（12 个本轮中有 10 个；
  MolHIV 两行视为 `mean_h2` 与 typed-S2 “comparable”）typed-S2 是**影响最大的
  可干预输入通道**，且排序稳定性被严重破坏（分类任务 Spearman 降到 0.15–0.42）。
  严格数值最大为 12/13（本轮 11/12）；MolHIV:C1 的 typed-S2 仅比 `mean_h2` 高 3.0%。
  这是**固定权重下输入块置零的敏感度比较**，不是训练后 S2 必要性的证明。
* **C1 与 A0 的依赖程度基本相同**（同一数据集内差异 ≤ 0.5 个 logit / 0.02 logD）：
  删掉 `b_mean` 没有把依赖转移到别处，也没有降低对 typed-S2 的依赖。

**边界声明**：固定权重下的输入通道干预，**不是**同容量重训消融，
**不是**合法化学编辑，不能据此声称该通道"必要"或"充分"。

### 5.3 参数差分解（实测）

A0 − C1 = **18 432**，全部落在 `local.mlp.lins.0.weight`：
`192 × 529 = 101 568` → `192 × 433 = 83 136`（96 列 × 192 投影）。其余张量逐位相同。

---

## 6. 效率（同环境、同 batch、同数据）

正式训练实测每 epoch 时间（3 seeds 均值，`analysis/epoch_time_measured.json`）：

| dataset | A0 s/epoch | C1 s/epoch | C1/A0 |
|---|---|---|---|
| ESOL | 0.2142 | 0.2174 | 1.015 |
| FreeSolv | 0.1086 | 0.1084 | 0.998 |
| BACE | 0.3888 | 0.3758 | 0.967 |
| BBBP | 0.4372 | 0.4373 | 1.000 |
| Tox21 | 1.6019 | 1.5828 | 0.988 |
| ogbg-molhiv | 8.8691 | 8.7159 | 0.983 |
| NCI1 | 2.3055 | 2.3188 | 1.006 |
| PROTEINS | 0.6849 | 0.6846 | 1.000 |
| IMDB-BINARY | 0.6745 | 0.6812 | 1.010 |
| IMDB-MULTI | 0.8362 | 0.8182 | 0.978 |

其它配对测量（`analysis/efficiency_table.csv`，Validation 结构缓存，batch 128）：

| 指标 | A0 | C1 |
|---|---|---|
| 可训练参数 | 247 106 | 228 674（−7.5%） |
| 参数显存（fp32） | 0.943 MB | 0.872 MB |
| 推理吞吐（ESOL / molhiv） | 16 397 / 6 752 graphs·s⁻¹ | 19 230 / 10 751 |
| CUDA peak allocated（molhiv） | 46.0 MB | 43.7 MB |
| 结构缓存（molhiv） | 328.2 MB（共享） | 328.2 MB |
| 缓存构建（molhiv） | 2.96 s（共享） | 2.96 s |
| 长尾成本比（最长/最短 decile） | 1.2–2.6 | 1.2–2.9 |

结论：删掉 96 维输入块让参数少 7.5%，但**每 epoch 时间、推理吞吐、显存、
缓存占用都在噪声内**——这个规模的模型不是输入列受限的。

---

## 7. H3 / H4 — Local-edit 回溯（独立冻结轮次）

在 `../lcrc_local_edit_error_diagnosis_v1/` 内完成，本轮未修改其科学定义，
只补上缺失的 `code/edit_probe.py`。

| 项 | 值 |
|---|---|
| 训练单元 | 6（A0/C1 × seed 0,1,2，360 epoch，Fit2016 / Select336） |
| GPU 时间 | 0.3992 GPU-h |
| 迹 | 3054 条；恒等式 max 残差 1.1e-6；反序 5.7e-7；批量 4.8e-7；重编号 8.3e-7 |
| Audit | 74 对 / 26 个配对组件 / 19 个共享类，**只读一次** |
| C1 `Gain` | **+0.0015**（+0.0030 / +0.0036 / −0.0022），CI [−0.0108, +0.0114] |
| A0 `Gain` | +0.0033（3/3 正号） |
| 预注册判定 | **`NO_INCREMENT_THIS_PROBE`**（阈值 0.03） |

关键：Audit 上每个单元的最佳方法基本都是 `NATIVE`（预测 e_delta = 0）
或与其相差 <0.003 的 `FAMILY`；`Gain > 0` 只因 Diagnose 选出的 `BEST_BASE=GLOBAL`
在 Audit 上比零预测器差 0.004–0.006。详见该目录 `REPORT.md` / `DECISION.md`。

---

## 8. 执行偏差与不可声明项

**偏差（全部记录、不修改任何科学定义）**

1. **TU 重复运行 6 个单元**：在引入互斥分片之前，两个 pool 各自扫描缺失单元，
   `tu_IMDB-MULTI_f1_s0_C1`、`f1_s1_A0`、`f1_s1_C1`、`f2_s0_A0`、`f2_s0_C1`、
   `f2_s1_A0` 各被跑了两遍，浪费 **≈0.29 GPU-h**（账本保留全部条目）。
   这 6 个目录的 `checkpoint.pt` SHA 与 `run.json`/`selection.json` 记录**逐一核对一致**，
   两次运行使用完全相同的冻结配方/折/种子，结果按其中一次记入。
   之后加入 `--shard K/2`（按单元名哈希的确定性划分）以避免再次发生。
2. **ESOL 成本探针的首个变体受 CUDA 预热影响**（0.368 s vs 0.188 s）。
   方案选择与预算投影因此对每个数据集取两变体较慢值（保守方向）。
   正式训练实测值（§6）显示两变体实际相等。
3. **MolHIV 先冻结先评分**：随后 `ptb_freeze.py --molecule` 无法重新冻结已评分数据集，
   已加入"已冻结且已评分 → 原样保留"的分支；manifest 的 SHA 与 checkpoint 一致。
4. **上游脏工作树**：`res` 上 `lcrc_tu_quick_screen_v1` 的脏改动先备份到
   `~/cy/rr-backups/20261102T163030Z/` 再 stash，未使用 `reset --hard` / `clean -fd`。

**不能声明**

1. 不能说 C1 与 A0 "统计等价"：CI 跨 0 只表示证据不足。
2. 不能把 MolHIV 100-epoch 结果与历史 360-epoch A0 做配对比较
   （即便本轮 6 个单元都没有触发预算受限标记）。
3. 不能把 TU 的 fold×seed 分数当独立样本；`C_5x2` 的证据强度低于 10 折。
4. 不能把 `FROZEN_MODEL_INPUT_INTERVENTION` 当作消融或化学编辑结论。
5. 不能把 Lipophilicity / ZINC 的冻结历史结果当本轮新证据。
6. Test 结果均为 `EXPOSED_TEST_EXPLORATORY`，不是盲测。
7. 本轮**没有**复现任何强基线（ESA+/GRIT/PNA/GIN/GINE/CIN），
   `literature_reference.csv` 只做定位，不做"击败/被击败"的断言。

---

## 9. 交付与复现

| 文件 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结设计（本文件不改它） |
| `configs/protocol.json`、`configs/run_matrix.json` | 机器可读协议与运行矩阵 |
| `analysis/protocol_freeze.json` | 设计文件 SHA-256 + 代码哈希 + 选中方案 |
| `analysis/source_manifest.json` | 数据/上游来源与哈希（6 分子 + 4 TU + 12 个上游文件） |
| `analysis/split_manifest.json` | TU 折划分、20 个折指纹、80 个单元 checkpoint SHA |
| `analysis/correctness_gates.json` | 13 个门禁 |
| `analysis/run_registry.json`、`analysis/compute_ledger.json` | 调度记录与 GPU 记账 |
| `analysis/checkpoint_manifest.json` | 冻结 checkpoint（含 SHA；冻结后禁止重训） |
| `analysis/test_access_ledger.json` | 36 次一次性 Test 访问审计 |
| `analysis/molecular_results.csv` | 36 个新单元 + 7 条冻结历史（Lipophilicity ×3 repeats、ZINC 标注） |
| `analysis/tu_results.csv` | 80 个 TU 单元 |
| `analysis/paired_differences.csv` | 配对差（分子按行、TU 按 outer fold、冻结历史按 repeat） |
| `analysis/statistics.json` | bootstrap 协议、seed 离散、失败计数 |
| `analysis/sample_diagnostics.json` | 逐样本改善/退化、极端误差集中度、top-k 移除敏感性、size bin |
| `analysis/efficiency_table.csv`、`analysis/epoch_time_measured.json` | 效率 |
| `analysis/mechanism_audit.json` | H2 证据 + 干预 + 参数分解 |
| `analysis/literature_reference.csv` | 固定外部参照（无实时检索） |
| `analysis/report_tables.md` | 由 `code/ptb_report.py` 从上述 CSV 生成 |

复现（在 `res` 上，按顺序）：

```bash
uv run python code/ptb_gates.py                     # 13 个门禁（CPU）
uv run python code/ptb_source_manifest.py           # 来源与哈希
uv run python code/ptb_protocol.py                  # 由实测成本冻结 TU 方案
uv run python code/ptb_queue.py --plan molecule --pool-label res-gpu0
uv run python code/ptb_queue.py --plan local_edit_then_tu --pool-label res-gpu1 --shard 1 --shards 2
uv run python code/ptb_queue.py --plan tu --pool-label res-gpu0 --shard 0 --shards 2
uv run python code/ptb_freeze.py --molecule --tu
uv run python code/ptb_score.py --all-frozen         # 一次性 Test
uv run python code/ptb_aggregate.py
uv run python code/ptb_efficiency.py --device cuda
uv run python code/ptb_mechanism.py --device cuda --datasets Lipophilicity ESOL FreeSolv BACE BBBP Tox21 ogbg-molhiv
uv run python code/ptb_report.py
```
