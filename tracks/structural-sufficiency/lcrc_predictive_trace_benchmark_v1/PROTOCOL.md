# PROTOCOL — LCRC Predictive Strength & Traceability Benchmark v1（冻结执行协议）

Track: `tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/`
Round / Protocol ID: `lcrc-predictive-trace-benchmark-v1`
参照 HEAD：`9628ab4ca901aeecd2957b9e2c172a9fab485f7d`（本轮开始时实际 HEAD 与之一致，工作树干净）

本文件只声明**已由研究者给定**的研究设计，并把它映射到机器可读配置与代码入口。
本文件 + `configs/protocol.json` + `configs/run_matrix.json` 在任何正式 GPU 训练
之前写出并冻结（`analysis/protocol_freeze.json` 记录三者的 SHA-256 与代码哈希）。
之后只允许追加"执行记录"和"事后偏差"，不得修改已冻结的科学定义。

---

## 0. 执行映射

| 协议要素 | 机器可读 / 代码入口 | 冻结记录 |
|---|---|---|
| 模型 A0 / C1（参数量、列删除映射） | `code/ptb_model.py`、`code/ptb_common.py` | `analysis/correctness_gates.json` |
| 分子任务矩阵（36 单元） | `configs/run_matrix.json::molecule` | `analysis/protocol_freeze.json` |
| TU 折划分与方案选择 | `code/ptb_data.py::fold_partition`、`code/ptb_protocol.py` | `analysis/protocol_freeze.json`、`analysis/split_manifest.json` |
| 训练配方 R3 / TU 配方 | `code/ptb_common.py::RECIPE`、`code/ptb_train_mol.py`、`code/ptb_train_tu.py` | 同上 |
| 硬件成本测量（唯一允许的 TU 方案依据） | `code/ptb_cost.py` | `analysis/cost_measurement.json` |
| 预算感知调度 | `code/ptb_queue.py` | `analysis/compute_ledger.json`、`analysis/run_registry.json` |
| checkpoint 冻结 → 一次性 Test | `code/ptb_freeze.py`、`code/ptb_score.py` | `analysis/checkpoint_manifest.json`、`analysis/test_access_ledger.json` |
| 正确性门禁 | `code/ptb_gates.py`、`tests/` | `analysis/correctness_gates.json` |
| 统计与表格 | `code/ptb_aggregate.py` | `analysis/*.csv`、`analysis/statistics.json` |
| 效率表 | `code/ptb_efficiency.py` | `analysis/efficiency_table.csv` |
| 结构机制审计 | `code/ptb_mechanism.py` | `analysis/mechanism_audit.json` |
| Local-edit 回溯（独立冻结轮次） | `../lcrc_local_edit_error_diagnosis_v1/code/edit_*.py` | 该目录自己的 `analysis/` |

---

## 1. 模型

* **A0 / FULL**：冻结 LCRC-A 计算图（`lcrc_lipo_model_dissection_v1` 的 `A0`），
  单任务参数 **247 106**。
* **C1 / NO_BOND_MEAN**：只删除 `b_mean = b_sum/deg` 的 96 维输入块与
  LocalMLP 第一层对应 96 列权重，单任务参数 **228 674**，
  差值 **18 432**。C1 是本轮**预先指定**的主模型，A0 为配对参照。
* Tox21 需要 12 输出：`f` 与 `rho` 的最后一层从 1 扩到 12，两个变体各自
  +1 430 参数，差值仍为 **18 432**（`analysis/correctness_gates.json`）。
* 同一训练种子内 C1 由 A0 的初始 state 按冻结列映射删除得到；共享张量
  逐位相等（每个 run 记录 `shared_state_equal_to_full`）。
* 不添加消息传递、注意力、原型、距离桶、特征堆叠、SSL 或任何结构组合模块。

TU 变体沿用 `lcrc_tu_quick_screen_v1` 的 `FULL`/`C1` 主体与特征适配器
（`tu_model.TULCRC`），FULL/C1 同样严格相差 18 432 参数。

---

## 2. 实验包 A：分子任务（36 个正式训练单元）

| 数据集 | 模型 | Seeds | Epochs | 主指标 |
|---|---|---|---|---|
| ESOL | A0 / C1 | 0,1,2 | 360 | RMSE（原始单位） |
| FreeSolv | A0 / C1 | 0,1,2 | 360 | RMSE |
| BACE | A0 / C1 | 0,1,2 | 360 | ROC-AUC |
| BBBP | A0 / C1 | 0,1,2 | 360 | ROC-AUC |
| Tox21 | A0 / C1 | 0,1,2 | 360 | 12 任务 observed-label macro ROC-AUC |
| ogbg-molhiv | A0 / C1 | 0,1,2 | **100** | 官方 OGB ROC-AUC |

* 数据：ChemBench 已发布划分（冻结字节）用于前五个；`ogbg-molhiv` 用官方
  OGB scaffold split 与官方 evaluator。**不重新生成官方划分**，不删除困难分子。
* 配方 R3：AdamW，peak LR `3e-4` per-step cosine 到 `1e-5`，wd `1e-3`，
  batch 128，dropout 0.10（在共享 `h` 上，每 forward 一次），grad clip 5.0，
  无 SSL/EMA/SWA/soup，无早停，无超参搜索。
* checkpoint 选择**只用**主 Validation 指标（回归严格更小；分类并列取更低
  Valid CE，再并列取更早 epoch）。
* MolHIV 上限 100 epoch，**不自动扩展**；best epoch ≥90 时记
  `BUDGET_LIMITED_POSSIBLE`。不与历史 360-epoch A0 做公平配对比较。
* Lipophilicity 与 ZINC 不重训：直接读取既有冻结结果并标注来源等级。

### Test 访问

数据集的 36/6 个单元全部完成 → `ptb_freeze.py` 写出该数据集的
`checkpoint_manifest.json` → 之后 `ptb_score.py` 才一次性读取 Test 标签。
所有新 Test 结果统一标记 `EXPOSED_TEST_EXPLORATORY`（历史轮次已读过 Test）。

---

## 3. 实验包 B：TU 交叉验证

数据集：NCI1、PROTEINS、IMDB-BINARY、IMDB-MULTI；模型 A0/FULL 与 C1。

**方案优先级（只能依据硬件测量选择，不能依据任何 Valid/Test 分数）**

1. 方案 A：`StratifiedKFold(10, shuffle=True, random_state=20261102)` × seeds 0,1
2. 方案 B：`StratifiedKFold(5, shuffle=True, random_state=20261102)` × seeds 0,1,2
3. 方案 C：`StratifiedKFold(5, shuffle=True, random_state=20261102)` × seeds 0,1，
   标记 `RESOURCE_LIMITED`

选择规则（写入 `configs/protocol.json::cost_projection`，含全部算术）：

```
unit_gpu_hours = measured_mean_epoch_seconds * formal_epochs / 3600
family_total   = sum(unit_gpu_hours) * 1.20 + fixed overheads
eligible       = (molecule + local_edit + TU + 1.0 reserve <= 14 GPU-h)
                 and (projected two-GPU makespan <= 7 h)
```

取第一个 eligible 的方案。降级方案必须按指定规则**重新构建**真正的 5 折划分，
不得把 10 折的前 5 折当作独立的 5-fold CV。

**折内协议**

* outer fold = 评价集；outer-training 中固定分层取 10% 作为 checkpoint 选择
  Validation，`random_state = 20261102 + fold_id`。
* 两个模型、所有训练种子共享完全相同的 fold；按显式
  `(dataset, fold, seed, model)` 键对齐。
* 每 fold 的节点类型词表与输入标准化只从该 fold 的实际训练部分拟合
  （`code/ptb_data.py::prepare_fold`，含 Train-only 均值审计）。
* 训练配方：batch 64，AdamW LR `1e-3`、wd `1e-4`、per-step cosine 到 `1e-5`，
  dropout 0.10，CE loss，grad clip 5.0，200 epoch，无增强。
* 选择：Valid accuracy 最大 → 并列取更低 Valid CE → 再并列取更早 epoch。
* outer held-out fold 只在该单元 checkpoint 哈希落盘后评估一次。
* 不把 fold×seed 全部分数当作独立样本做显著性检验。

---

## 4. 实验包 C：结构机制与效率审计（只读复用）

只读取并复核（**以 follow-up 勘误后的 contrast 表为准**）：
`lcrc_lipo_random42_controls_v2` 的容量匹配 S2-off、
`lcrc_lipo_mechanism_generalization_ssl_v2` 的 scaffold CV / 容量控制 / SSL、
`lcrc_lipo_model_dissection_v1` 与 `_followup_v1` 的 C1 简化与 J/C 诊断、
`zinc_dccl_bottleneck_v1` 与 `zinc_error_mechanism_audit_v1` 的有限样本瓶颈。

在新冻结 checkpoint 上只做**无需重训**的输入通道干预：把 typed-S2 输入列
（`LOCAL_LAYOUT["s2_type"]…["s2_content"]`）整体置零，报告固定权重下的
`Δprediction`，标记 `FROZEN_MODEL_INPUT_INTERVENTION`；不得称为同容量重训消融，
也不得称为合法化学编辑。

效率（相同环境、相同 batch、相同数据，A0/C1 配对测量）：可训练参数量、参数
显存、结构缓存构建时间、预处理数据大小与缓存占用、每 epoch 训练耗时、best
checkpoint 选取前的累计 GPU 时间、固定 batch 推理吞吐、CUDA peak allocated、
长尾图成本。

---

## 5. 实验包 D：Local-edit 回溯（在独立冻结目录内完成）

目录：`tracks/structural-sufficiency/lcrc_local_edit_error_diagnosis_v1/`
协议：该目录 `PROTOCOL.md`（509 对 / 211 类变换 / 2016-336-504-504 划分 /
R3 360 epochs / Fit 训练 / Select 选择 / 探针定义 / L2=5 / 一次性 Audit）。
本轮**不修改**其中任何科学定义，只完成：6 个训练单元、P1 完整回溯、
缺失的 `code/edit_probe.py`（`--stage fit|audit|report`）、一次性 Audit 评分、
bootstrap / 影响度 / 闭环 / gauge 诊断、该目录的 `REPORT.md` 与 `DECISION.md`。

主结果：`Gain = Audit_MAE(BEST_BASE) − Audit_MAE(TRACE)`，verdict 只按该协议
§7 的预注册规则给出。额外 `gauge_stability` 只作描述性审计，不改变主指标。

---

## 6. 资源门禁与预算

* 硬上限：8 h 墙钟、16 A100 GPU-hours（含失败、smoke、推理）。
* 软上限：14 A100 GPU-hours；预计 7 h 内结束正式训练并开始收尾。
* 开始前 `rr doctor res`；两张 GPU 均须可用。
* 每 pool 至多一个正式训练进程；`res-gpu0` 优先分子任务，`res-gpu1` 优先 TU，
  分子任务结束后空闲 GPU 执行 Local-edit 或协助 TU。
* 每个单元独立身份、独立 checkpoint、独立日志、独立预算记录；完成后立即
  持久化；`analysis/compute_ledger.json` 每次启动前检查。
* 未完成单元标 `NOT_RUN_BUDGET`，不补算、不伪造、不用较短训练冒充正式完成。

### 远端脏工作树的处理（不得静默丢弃用户数据）

开始时 `res` 远端 checkout 位于 `52a94d55`，且
`tracks/structural-sufficiency/lcrc_tu_quick_screen_v1/analysis/gates.json`
（另一条 track 的 tracked 文件）与一个未跟踪的 `smoke_cuda.json` 存在本地改动。
处理方式：先把两者**复制备份**到 `~/cy/rr-backups/<utc>/`，再用
`git stash push --include-untracked` 保存为可恢复的 stash（记录 stash sha），
**不使用** `git reset --hard` 或 `git clean -fd`。备份路径与 stash id 写入
`analysis/source_manifest.json::remote_worktree_rescue`。

---

## 7. 统计

* 每条结果都带来源标签：`NEW_INTERNAL_CV`、`NEW_EXPOSED_TEST_EXPLORATORY`、
  `FROZEN_HISTORICAL_SAME_SPLIT`、`FROZEN_HISTORICAL_OTHER_PROTOCOL`、
  `LITERATURE_SAME_OFFICIAL_SPLIT`、`LITERATURE_UNMATCHED_PROTOCOL`、
  `FROZEN_MODEL_MECHANISM_ONLY`。
* 主表保留每个 seed/fold 的原始分数、mean/std、A0/C1 配对差、各 gap、参数量、
  GPU 小时、样本量与划分信息、best epoch、失败/缺失/未执行单元数。
* 配对 bootstrap：分子任务按**分子行**、Local-edit 按**不可拆分 pair
  component**、TU 按 **outer fold** 重采样；相同 resample 索引同时用于被比较
  模型；seed 变异与 bootstrap 分开报告（`seed=20261102`，2000 次）。
* CI 跨零不声称优势，也不声称等价。

---

## 8. 外部基线：固定引用，不调研、不复现

只进入 `analysis/literature_reference.csv`，并标注来源等级与不可比原因。

* ZINC-12k：DCC-L（本仓库冻结四种子 Test MAE 0.067319 ± 0.000709）；
  GRIT ≈ 0.059 ± 0.002；ESA+ ≈ 0.052 ± 0.001（论文报告）。ESA+ 的 ZINC-FULL
  不得混入 ZINC-12k。DCC-L 是另一套 ZINC 专用模型家族，不是 C1 的结果。
* ogbg-molhiv 官方参照：GIN+virtual node 0.7707±0.0149、PNA 0.7905±0.0132、
  GINE 0.7921±0.0128、CIN 0.8094±0.0057、HIG 0.8403±0.0021
  （`https://ogb.stanford.edu/docs/leader_graphprop/`）。
* TU：GIN 常见论文基准 NCI1 ≈82.7%、PROTEINS ≈76.2%、IMDB-BINARY ≈75.1%、
  IMDB-MULTI ≈52.3%（描述性定位，协议不完全一致）。
* Lipophilicity 与其它 MoleculeNet：复用
  `lcrc_molnet_benchmark_pilot_v2/analysis/literature_matrix.csv` 与同划分
  XGBoost / PNA-native 历史结果，保留分割/指标/特征/来源等级。

---

## 9. 禁止事项（本轮硬约束）

不复现 ESA+/GRIT/PNA/GIN/GINE/CIN；不联网调研或新增未经确认的论文数字；
不新增架构/注意力/SSL/特征/损失/搜索空间；不按 Test 选择模型；不用单种子最佳值
代替均值；不使用被勘误覆盖的旧 contrast 表；不把不同划分的分数当作严格可比；
不把输入通道干预称为真实分子编辑；不把局部贡献系数解释为化学因果；
不把计算恒等式当作解释忠实度证明；不修改已有冻结协议、历史权重或用户数据；
不超出 8 h / 16 A100 GPU-h；不因单项失败更换研究目标。

---

## 10. 执行记录（冻结后追加，不改动上文）

见 `REPORT.md` §11 与 `analysis/`：
`protocol_freeze.json`、`source_manifest.json`、`split_manifest.json`、
`correctness_gates.json`、`cost_measurement.json`、`run_registry.json`、
`checkpoint_manifest.json`、`compute_ledger.json`、`test_access_ledger.json`。
