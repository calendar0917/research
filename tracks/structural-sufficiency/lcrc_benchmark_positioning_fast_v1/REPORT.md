# LCRC Benchmark Positioning Fast v1 — 事实报告

- **Round ID**：`lcrc-benchmark-positioning-fast-v1`
- **结果标签**：`EXPLORATORY_FAST_BENCHMARK`
- **模型**：仓库已有 **LCRC-C1 / NO_BOND_MEAN**（仅删除 local input 的 96 维 `b_mean` 块；其余 S1/S2、local contribution、global `rho` 与冻结宽度保持不变）
- **正式训练代码 commit**：`4c24f5f50db0`（`4c24f5f50db08ddb5661147d6e166019bc8bd464`）
- **汇总代码 commit**：`29e51c9ba2f5`（`bp_aggregate.py` 的 blocked-dataset 标记；不参与训练或模型选择）
- **机器**：`res` / `a100-2`，2× NVIDIA A100-SXM4-40GB，driver 550.163.01，torch 2.5.1+cu124
- **队列执行窗口（UTC）**：2026-11-03T21:17:59Z 至 2026-11-03T22:40:14Z；seed0 重跑 2026-11-03T22:42:01Z 至 2026-11-03T23:01:27Z
- **报告生成**：基于 `results/`、`analysis/` 与远端 `runs/` 中的 `run.json`/`test_predictions.npz`；未重新训练、未根据 Test 结果改配置。

> 本报告只记录工程执行结果。截图 reference 数字统一标记 `SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED`，不得当作严格同协议对手。

## 0. 摘要

- 计划任务 **102** 个：TU 9 数据集 × 10 folds = 90；OGB 3 数据集 × 3 seeds = 9；Peptides-func × 3 seeds = 3。
- 完成 **82** 个：7 个 TU 数据集共 **70/70 folds**；3 个 OGB 数据集共 **9/9 seeds**；Peptides-func **3/3 seeds**（seed0 第一次因 25 分钟上限阻塞，使用同一冻结配方重跑一次完成）。
- 资源阻塞 **20** 个：COLLAB、REDDIT-BINARY 各 10 folds。原因均为精确 S2 路径总量超过冻结的 eager cache 预算；未采样、未截断、未改图。
- 其他失败：**0** 个。全部 82 个完成任务在 checkpoint SHA-256 记录之后只评估一次 Test。
- GPU 账本总计 **2.973 A100 GPU-hours**：其中最终 82 个完成任务自身合计 **2.553 h**，Peptides seed0 第一次阻塞尝试 **0.420 h**（1512.8 s）。双 worker 的最终任务（不含手工重跑）合计 **2.654 h**。
- 正式队列墙钟 **约 82 分钟**（21:17:59–22:40:14Z）；加 seed0 重跑，最后任务结束于 23:01:27Z，总墙钟 **约 1 小时 43 分**。远低于 6 h soft / 10 h hard / 18 GPU-hours 上限。

## 1. 协议执行摘要

- 模型：**LCRC-C1 / NO_BOND_MEAN**；冻结宽度 64 / 96 / 96 / 32 / 192 / 64；从 scratch 训练。
- 优化：AdamW fp32，gradient clipping 5；无 ensemble、EMA、SWA、soup、数据增强、类别重采样、SSL。
- 归一化：所有 mean/scale 只在对应任务的 Train 行上拟合；TU 连续节点属性同样 train-only。
- 选择：每个任务只保存 Validation 最佳 checkpoint；TU 以 Validation Accuracy 为主指标（tie: 较低 Validation CE，再早 epoch）；OGB 以官方 Validation ROC-AUC；Peptides 以 Validation mean AP。
- Test：checkpoint 保存并计算 SHA-256 后才读取 Test；每个完成任务只评一次。`results/metric_recheck.csv` 从保存的 `test_predictions.npz` 独立重算，82 个任务的指标 delta 全部为 **0.0**。
- 本报告没有用 Test 结果回选任何配置；没有额外超参搜索、没有新模型/新特征/新消融。

## 2. 数据、缓存与划分

| 范围 | 来源/划分 | 说明 |
|---|---|---|
| TU 7 个完成数据集 | 官方 TUDataset，`cleaned=False` | 保留原始离散 node labels；MUTAG edge labels 按既有类型编码映射；PROTEINS/ENZYMES 连续 node attrs 保留并在 fold Train 上归一化；无连续 edge attrs |
| TU 划分 | `StratifiedKFold(n_splits=10, shuffle=True, random_state=20261104)` | outer Test = 当前 fold；outer Train = 其余 9 folds；Validation 为 outer Train 的 stratified 1/9，`random_state=1000+fold_id`；模型 seed `100+fold_id`；无 inner search |
| OGB 3 个 | 官方 `get_idx_split()` | 使用官方 train/valid/test indices；OGB 官方 Evaluator 计算 Test ROC-AUC；Tox21 12-task 缺失标签 mask 保留，未把 NaN 填成负例 |
| Peptides-func | 官方 PyG LRGBDataset 固定 split | 10 个 sigmoid 输出，BCEWithLogitsLoss；10-label macro AP；9 列原子/3 列键离散特征按合法类别 one-hot；S1/S2 类型码取 column 0；反向边合并为同一物理键 |

### 2.1 缓存与 exact S2 统计

| 范围 | 数据集/划分 | 图/分子数 | max nodes | max edges | exact S2 总数 | max S2/图 | cache MB | build s | cache SHA-256 前 12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| TU | IMDB-B | 1000 | 136 | 1249 | 2,782,108 | 77,830 | 56.8 | 9.7 | 6a7199f84528 |
| TU | IMDB-M | 1500 | 89 | 1467 | 3,056,474 | 107,956 | 61.6 | 10.7 | 9f54799bbe2d |
| TU | MUTAG | 188 | 28 | 33 | 10,856 | 106 | 1.1 | 0.5 | 75f0b5d91941 |
| TU | NCI1 | 4110 | 111 | 119 | 383,116 | 344 | 55.0 | 19.7 | 8e965712d395 |
| TU | PROTEINS | 1113 | 620 | 1049 | 499,732 | 7,420 | 21.5 | 9.2 | b498be81e9ff |
| TU | DD | 1178 | 5748 | 14267 | 7,753,122 | 126,510 | 347.2 | 125.7 | 1a4aab43c58f |
| TU | ENZYMES | 600 | 126 | 149 | 235,194 | 920 | 11.4 | 3.3 | 809d990645ae |
| OGB | MolHIV | 41127 | 222 | 251 | 3,225,188 | 804 | 460.1 | 97.2 | 4c3bf72223b7 |
| OGB | MolBACE | 1513 | 97 | 101 | 160,050 | 278 | 22.5 | 5.8 | b8601e088cb9 |
| OGB | MolTox21 | 7831 | 132 | 145 | 415,290 | 426 | 63.2 | 17.6 | 0bf57b8b5ae6 |
| LRGB | Peptides train | 10873 | 444 | 464 | 4,492,788 | 1,270 | 1669.9 | 820.0 | 7b57289e8a7c |
| LRGB | Peptides valid | 2331 | 435 | 449 | 949,954 | 1,240 | 353.1 | 172.3 | 6c1d66677eff |
| LRGB | Peptides test | 2331 | 439 | 453 | 950,540 | 1,202 | 353.4 | 173.2 | ea897c1b5143 |

说明：TU 的 `max nodes`/`max edges` 来自清洗后的物理图；`exact S2 总数` 是按冻结的 non-backtracking 2-path 定义精确计数/枚举，不含采样。OGB 的图数来自官方 split 行数之和；Peptides 三个划分分别建 cache。

### 2.2 大图预检查与资源阻塞

| 数据集 | 原始图数 | max nodes | max edges | exact S2 总数 | max S2/图 | scan 秒 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| COLLAB | 5,000 | 492 | 40,119 | 3,934,836,808 | 18,089,154 | 53.3 | RESOURCE_BLOCKED (EXACT_S2_CACHE_BUDGET) |
| REDDIT-BINARY | 2,000 | 3782 | 4,071 | 369,653,358 | 9,377,944 | 4.6 | RESOURCE_BLOCKED (EXACT_S2_CACHE_BUDGET) |

- **DD**：5-epoch 无 Test 预检查完成（11.93 s；peak allocated 1.87 GB / reserved 5.95 GB）；正式 10 folds 每 fold 70–116 s，均低于 25 分钟。
- **Peptides-func**：5-epoch 无 Test 预检查完成（50.74 s；peak allocated 0.36 GB / reserved 0.77 GB）；正式 120 epoch 每个 seed 1131–1264 s，低于 25 分钟。
- **COLLAB / REDDIT-BINARY**：精确 S2 预检查在扫描阶段即超过冻结的 `EAGER_S2_LIMIT = 20,000,000`；未建立 eager cache，未启动 5-epoch 或正式训练。两数据集均无任何静默采样/截断/改图。
- TU 其余数据集预处理均低于 30 分钟；OGB cache 建立 5.8–97.2 s；Peptides train cache 820 s、valid/test 各约 172 s。

## 3. 正确性门禁

- **单元测试**：`tests/test_bp_core.py` 在正式 GPU 队列启动前 **11 passed**（本地 CPU）。
- **artifact audit**：`analysis/correctness_gates.json` **26/26 passed**，包括：
  - TU split 两两互斥、覆盖全部样本（7 个完成数据集）；
  - OGB 官方 split 行号与 cache 索引一致、互斥且覆盖；
  - TU/OGB cache 不含 target/label；
  - TU train-only 归一化重算一致（容差 1e-4，float32）；
  - S2 与上游 `molecular_graph.analyse_graph` 在小图上逐项相等；
  - C1 结构 = FULL − `b_mean`，共享参数逐项一致；
  - 单图 vs batch forward 一致；
  - TU accuracy / OGB ROC-AUC / peptide macro AP / Tox21 masked loss 的维度、缺失值与重算；
  - Peptides 三个 split 的 AP 维度与非缺失标签检查。
- 该 artifact audit 的时间戳为 2026-11-03T21:28:09Z，晚于队列启动（21:17:59Z）；它是只读检查，不参与训练或 checkpoint 选择。正式训练前的 11 项 pytest 已通过。
- **metric recheck**：`results/metric_recheck.csv` 82 行，stored vs recomputed 最大绝对 delta = **0.0**。

## 4. 结果

### 4.1 TU：截图定位表

| Dataset | 截图 reference | LCRC-C1 mean Acc | 跨 fold std | Δ (C1 - ref) | folds | 协议标记 |
|---|---:|---:|---:|---:|---:|---|
| IMDB-BINARY | 73.87 | 73.60 | 4.25 | -0.27 | 10/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| IMDB-MULTI | 50.97 | 48.07 | 3.12 | -2.90 | 10/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| COLLAB | 80.72 | NOT_RUN | NOT_RUN | NOT_RUN | 0/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| REDDIT-BINARY | 91.14 | NOT_RUN | NOT_RUN | NOT_RUN | 0/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| MUTAG | 89.20 | 84.53 | 6.55 | -4.67 | 10/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| NCI1 | 83.09 | 82.99 | 1.99 | -0.10 | 10/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| PROTEINS | 77.85 | 75.92 | 2.32 | -1.93 | 10/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| DD | 79.48 | 80.72 | 4.81 | +1.24 | 10/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| ENZYMES | 69.73 | 67.33 | 6.25 | -2.40 | 10/10 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |

说明：reference 来自用户截图；`SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED`。COLLAB/REDDIT-BINARY 的 `NOT_RUN` 是资源阻塞，不是 0 分。

### 4.2 TU：每 fold Test Accuracy（%）

| Dataset | F0 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | 76.00 | 79.00 | 74.00 | 73.00 | 68.00 | 68.00 | 74.00 | 80.00 | 69.00 | 75.00 | 73.60 ± 4.25 |
| IMDB-MULTI | 50.00 | 50.67 | 46.00 | 44.67 | 52.67 | 52.00 | 44.00 | 45.33 | 47.33 | 48.00 | 48.07 ± 3.12 |
| MUTAG | 89.47 | 89.47 | 84.21 | 84.21 | 89.47 | 73.68 | 89.47 | 84.21 | 72.22 | 88.89 | 84.53 ± 6.55 |
| NCI1 | 84.18 | 81.27 | 86.86 | 82.73 | 82.73 | 84.67 | 83.21 | 81.27 | 83.21 | 79.81 | 82.99 ± 1.99 |
| PROTEINS | 77.68 | 76.79 | 74.11 | 72.07 | 74.77 | 77.48 | 77.48 | 72.97 | 76.58 | 79.28 | 75.92 ± 2.32 |
| DD | 83.05 | 82.20 | 79.66 | 82.20 | 74.58 | 87.29 | 83.90 | 85.59 | 73.50 | 75.21 | 80.72 ± 4.81 |
| ENZYMES | 76.67 | 73.33 | 65.00 | 60.00 | 60.00 | 75.00 | 66.67 | 68.33 | 60.00 | 68.33 | 67.33 ± 6.25 |

macro-F1 的每 fold 值与均值/样本标准差保存在 `results/tu_folds.csv` 和 `results/tu_summary.csv`；macro-F1 未参与 checkpoint 选择。

### 4.3 OGB 与 Peptides：每 seed Test 指标

| Dataset | seed | Test metric | score (%) | best/actual epoch | n_test | checkpoint SHA-256 前 12 |
|---|---:|---|---:|---:|---:|---|
| MolHIV | 0 | ROC-AUC | 74.24 | 47/87 | 4113 | 58bb66e92a17 |
| MolHIV | 1 | ROC-AUC | 78.04 | 32/72 | 4113 | c50d70c4bbcd |
| MolHIV | 2 | ROC-AUC | 72.38 | 15/60 | 4113 | 35325747a455 |
| MolBACE | 0 | ROC-AUC | 81.45 | 109/144 | 152 | 31b51f357776 |
| MolBACE | 1 | ROC-AUC | 80.77 | 21/56 | 152 | 8d1bb2b3818d |
| MolBACE | 2 | ROC-AUC | 84.28 | 114/149 | 152 | 79978b02db2d |
| MolTox21 | 0 | ROC-AUC | 74.37 | 47/77 | 784 | 1dad503b3e46 |
| MolTox21 | 1 | ROC-AUC | 74.57 | 55/85 | 784 | 5df9a0a1db3e |
| MolTox21 | 2 | ROC-AUC | 75.49 | 58/88 | 784 | 51a8b63f868a |
| Peptides-func | 0 | mean AP | 51.70 | 116/120 | 2331 | 1da869b87358 |
| Peptides-func | 1 | mean AP | 50.68 | 119/120 | 2331 | 5140a1668446 |
| Peptides-func | 2 | mean AP | 51.80 | 113/120 | 2331 | 7c7dcf89e0eb |

汇总（mean ± sample std，百分比）：

- MolHIV ROC-AUC：**74.88 ± 2.88**
- MolBACE ROC-AUC：**82.17 ± 1.86**
- MolTox21 ROC-AUC：**74.81 ± 0.60**
- Peptides-func mean AP：**51.40 ± 0.62**

### 4.4 分子/Peptides：截图定位表

| Dataset | Metric | 截图 reference | LCRC-C1 mean | 跨 seed std | Δ (C1 - ref) | seeds | 协议标记 |
|---|---|---:|---:|---:|---:|---:|---|
| MolHIV | ROC-AUC | 81.47 | 74.88 | 2.88 | -6.59 | 3/3 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| MolBACE | ROC-AUC | 81.67 | 82.17 | 1.86 | +0.50 | 3/3 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| MolTox21 | ROC-AUC | 76.67 | 74.81 | 0.60 | -1.86 | 3/3 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |
| Peptides-func | mean AP | 71.57 | 51.40 | 0.62 | -20.17 | 3/3 | SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED |

## 5. 资源账与运行记录

| 项目 | 数值 |
|---|---:|
| 计划任务 | 102 |
| 完成（complete） | 82 |
| 资源阻塞（RESOURCE_BLOCKED） | 20（COLLAB 10 + REDDIT-BINARY 10） |
| 其他失败/INPUT_ADAPTER_BLOCKED | 0 |
| 最终完成任务自身 GPU 时间 | 2.553 h |
| 失败/阻塞的 peptide seed0 第一次尝试 | 0.420 h（1512.8 s） |
| 账本总 GPU 时间（含失败尝试与重跑） | 2.973 h |
| 双 worker 记录（含 seed0 第一次阻塞尝试，不含手工重跑） | 2.654 h |
| worker 墙钟合计（两卡并行） | 2.736 h |
| 正式队列墙钟 | 约 82 min |
| 加 seed0 重跑后的总墙钟 | 约 1 h 43 min |
| 单任务最长实际耗时（完成） | 1264.3 s（peptide seed1；上限 1500 s） |
| 单任务 peak allocated VRAM 最大值 | 1.92 GB（DD fold05） |
| 训练参数范围 | 192,100–464,739（随输入/输出维度变化，主干宽度固定） |

- GPU 分配：gpu0 完成 34 个任务记录，gpu1 完成 48 个任务记录；每个 GPU 同时最多一个训练进程。
- `res-gpu0`/`res-gpu1` 均通过 `rr` 启动；`rr status` 两个 worker 均 `completed`，exit code 0。`run.json` 记录 `git_commit = 4c24f5f50db0`、`git_dirty = false`、CUDA 可用。rr run_id / GPU / exit status 明细见 `analysis/rr_runs.json`：worker-gpu0 `lcrc-bp-worker-gpu0-20261104-051753-df91e88c`、worker-gpu1 `lcrc-bp-worker-gpu1-20261104-051804-cd449c9d`、seed0 重跑 `lcrc-bp-peptide-seed0-retry-20261104-064158-e2cd3078`、汇总 `lcrc-bp-aggregate-20261104-070435-2ec50480`。
- Peptide seed0 第一次尝试在 epoch 120 开始前达到 1500 s，`bp_train.py` 主动写 `RESOURCE_BLOCKED / PER_TASK_TIME_LIMIT`，没有 checkpoint、没有 Test。归档于 `analysis/blocked_runs/peptide_peptides-func_seed0_attempt1_PER_TASK_TIME_LIMIT/`；同一 seed/配方重跑 1150.4 s 完成并记录 Test。`results/peptide_seed0_attempts.csv` 保留两次记录。
- 完整逐任务明细（best/actual epoch、秒数、VRAM、参数、checkpoint hash）见附录 A 与 `results/resource_ledger.csv`。

### 5.1 未完成/阻塞清单

| Dataset | 未运行 fold/任务 | 状态 | 原因 | 证据 |
|---|---:|---|---|---|
| COLLAB | 10（fold00–fold09） | RESOURCE_BLOCKED | EXACT_S2_CACHE_BUDGET | `analysis/COLLAB_precheck_counts.json`、`analysis/COLLAB_blocked.json` |
| REDDIT-BINARY | 10（fold00–fold09） | RESOURCE_BLOCKED | EXACT_S2_CACHE_BUDGET | `analysis/REDDIT-BINARY_precheck_counts.json`、`analysis/REDDIT-BINARY_blocked.json` |
| Peptides-func seed0 attempt1 | 1（epoch 120 中止） | RESOURCE_BLOCKED | PER_TASK_TIME_LIMIT（1512.8 s） | `analysis/blocked_runs/peptide_peptides-func_seed0_attempt1_PER_TASK_TIME_LIMIT/run.json`；attempt2 已完成 |

完整 20 行阻塞记录见 `results/failure_list.csv`；82 个完成任务见 `results/run_manifest.csv`。

## 6. 复现

**正式训练 commit**：`4c24f5f50db0`（训练与 Test 逻辑）。**汇总 commit**：`29e51c9ba2f5`（blocked 标记；不改变训练）。

```bash
# 0) 本地/远端 checkout 对齐
git checkout 4c24f5f50db0
rr deploy res

# 1) 生成 102-task 正式计划
python tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/code/bp_plan.py

# 2) 双 GPU worker（示例；deadline 用 epoch 秒）
rr run res lcrc-bp-worker-gpu0 --pool res-gpu0 --time 10:00:00 \
  -- python tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/code/bp_worker.py \
  --worker gpu0 --hard-deadline-epoch <HARD> --soft-deadline-epoch <SOFT> --gpu-budget-hours 18
rr run res lcrc-bp-worker-gpu1 --pool res-gpu1 --time 10:00:00 \
  -- python tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/code/bp_worker.py \
  --worker gpu1 --hard-deadline-epoch <HARD> --soft-deadline-epoch <SOFT> --gpu-budget-hours 18

# 3) 汇总（在训练完成后）
rr run res lcrc-bp-aggregate --pool res-local --time 00:15:00 \
  -- python tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/code/bp_aggregate.py
```

实际使用的 deadline：`SOFT = 1793762268`、`HARD = 1793776668`（UTC epoch 秒）。完整命令、run_id、GPU id、exit status 记录在 `analysis/worker_gpu0.json`、`analysis/worker_gpu1.json`、`results/budget.json`。

## 7. 事实观察（不推导新研究结论）

- 7 个完成的 TU 数据集中，DD 的均值高于截图 reference（+1.24 pp）；NCI1 基本持平（−0.10 pp）；其余 5 个低于 reference：IMDB-BINARY −0.27、PROTEINS −1.93、ENZYMES −2.40、IMDB-MULTI −2.90、MUTAG −4.67 pp。
- 分子任务中，MolBACE 高于 reference（+0.50 pp）；MolHIV −6.59 pp、MolTox21 −1.86 pp、Peptides-func −20.17 pp。
- 跨 fold 样本标准差最大的是 MUTAG 6.55 pp、ENZYMES 6.25 pp、DD 4.81 pp、IMDB-BINARY 4.25 pp；NCI1 最小 1.99 pp。
- 跨 seed 样本标准差：MolHIV 2.88 pp、MolBACE 1.86 pp、MolTox21 0.60 pp、Peptides-func 0.62 pp。
- 以上差异只是本协议下的描述性数字；截图 reference 的原始协议、划分、评估实现与训练预算未在本任务中独立核验，因此不构成同协议优劣结论。

## 附录 A：逐任务资源与 checkpoint

| Task | Status | best/actual epoch | seconds | VRAM alloc (GB) | params | checkpoint SHA-256 前 12 |
|---|---|---:|---:|---:|---:|---|
| `mol_molbace_seed0` | complete | 109/144 | 70.5 | 0.09 | 228674 | `31b51f357776` |
| `mol_molbace_seed1` | complete | 21/56 | 29.1 | 0.09 | 228674 | `8d1bb2b3818d` |
| `mol_molbace_seed2` | complete | 114/149 | 71.9 | 0.09 | 228674 | `79978b02db2d` |
| `mol_molhiv_seed0` | complete | 47/87 | 716.9 | 0.08 | 228674 | `58bb66e92a17` |
| `mol_molhiv_seed1` | complete | 32/72 | 736.9 | 0.08 | 228674 | `c50d70c4bbcd` |
| `mol_molhiv_seed2` | complete | 15/60 | 520.8 | 0.08 | 228674 | `35325747a455` |
| `mol_moltox21_seed0` | complete | 47/77 | 137.3 | 0.07 | 230104 | `1dad503b3e46` |
| `mol_moltox21_seed1` | complete | 55/85 | 151.7 | 0.07 | 230104 | `5df9a0a1db3e` |
| `mol_moltox21_seed2` | complete | 58/88 | 166.8 | 0.07 | 230104 | `51a8b63f868a` |
| `peptide_peptides-func_seed0` | complete | 116/120 | 1150.4 | 0.33 | 464739 | `1da869b87358` |
| `peptide_peptides-func_seed1` | complete | 119/120 | 1264.3 | 0.33 | 464739 | `5140a1668446` |
| `peptide_peptides-func_seed2` | complete | 113/120 | 1130.7 | 0.33 | 464739 | `7c7dcf89e0eb` |
| `tu_dd_fold00` | complete | 26/42 | 87.3 | 1.74 | 237792 | `6ab4a0f2b20f` |
| `tu_dd_fold01` | complete | 8/40 | 77.1 | 1.80 | 237792 | `f29ef310b78d` |
| `tu_dd_fold02` | complete | 9/40 | 70.0 | 1.77 | 237792 | `35e673037147` |
| `tu_dd_fold03` | complete | 33/58 | 93.0 | 1.75 | 237792 | `178a4f450934` |
| `tu_dd_fold04` | complete | 12/40 | 72.5 | 1.67 | 237792 | `bbda7b1f2fac` |
| `tu_dd_fold05` | complete | 26/51 | 82.3 | 1.89 | 237792 | `ee5d71d03b46` |
| `tu_dd_fold06` | complete | 28/40 | 66.5 | 1.76 | 237792 | `cf09cf3efa63` |
| `tu_dd_fold07` | complete | 8/40 | 62.3 | 1.73 | 237792 | `d49e9c69d54d` |
| `tu_dd_fold08` | complete | 46/71 | 116.0 | 1.89 | 237792 | `49db595132a1` |
| `tu_dd_fold09` | complete | 27/52 | 101.4 | 1.92 | 237792 | `72dbb804d127` |
| `tu_enzymes_fold00` | complete | 50/75 | 16.4 | 0.11 | 194796 | `570c5a1c344b` |
| `tu_enzymes_fold01` | complete | 40/65 | 15.6 | 0.11 | 194796 | `8cd33613caa2` |
| `tu_enzymes_fold02` | complete | 36/61 | 19.8 | 0.11 | 194796 | `a16151e7c23a` |
| `tu_enzymes_fold03` | complete | 35/60 | 15.4 | 0.11 | 194796 | `cac666584c83` |
| `tu_enzymes_fold04` | complete | 51/76 | 19.0 | 0.11 | 194796 | `26aefff9ce6c` |
| `tu_enzymes_fold05` | complete | 42/67 | 20.6 | 0.11 | 194796 | `b6f279b8c6d0` |
| `tu_enzymes_fold06` | complete | 47/72 | 18.0 | 0.11 | 194796 | `497c184d5bee` |
| `tu_enzymes_fold07` | complete | 53/78 | 23.3 | 0.11 | 194796 | `e7315fd9b6dd` |
| `tu_enzymes_fold08` | complete | 45/70 | 16.6 | 0.11 | 194796 | `afb0ccf50a50` |
| `tu_enzymes_fold09` | complete | 40/59 | 18.4 | 0.11 | 194796 | `4720de911ff5` |
| `tu_imdb_binary_fold00` | complete | 36/61 | 25.2 | 0.83 | 200092 | `3f82591e8cee` |
| `tu_imdb_binary_fold01` | complete | 39/64 | 26.7 | 0.78 | 200092 | `88668696533a` |
| `tu_imdb_binary_fold02` | complete | 61/84 | 33.1 | 1.00 | 200092 | `592de9c5962a` |
| `tu_imdb_binary_fold03` | complete | 24/44 | 19.1 | 0.86 | 200092 | `c861acb93213` |
| `tu_imdb_binary_fold04` | complete | 28/53 | 20.2 | 0.83 | 200092 | `4e205e7ae4fd` |
| `tu_imdb_binary_fold05` | complete | 7/40 | 16.7 | 0.87 | 200092 | `77e29b5e6b6a` |
| `tu_imdb_binary_fold06` | complete | 50/75 | 40.9 | 0.87 | 200092 | `7869012eed8c` |
| `tu_imdb_binary_fold07` | complete | 52/59 | 26.7 | 0.84 | 200092 | `9043188c8099` |
| `tu_imdb_binary_fold08` | complete | 24/43 | 17.8 | 0.82 | 200092 | `19e9a5c31ec8` |
| `tu_imdb_binary_fold09` | complete | 30/55 | 25.5 | 0.85 | 200092 | `09dd27aab092` |
| `tu_imdb_multi_fold00` | complete | 24/49 | 29.0 | 0.82 | 200222 | `74f9c5d3a16a` |
| `tu_imdb_multi_fold01` | complete | 27/52 | 32.8 | 0.87 | 200222 | `d5e5a279781d` |
| `tu_imdb_multi_fold02` | complete | 20/45 | 25.2 | 0.74 | 200222 | `c28f682e0db9` |
| `tu_imdb_multi_fold03` | complete | 26/51 | 32.9 | 0.75 | 200222 | `100d17622ecc` |
| `tu_imdb_multi_fold04` | complete | 25/50 | 28.2 | 0.95 | 200222 | `1360e6cf425b` |
| `tu_imdb_multi_fold05` | complete | 25/50 | 30.9 | 0.98 | 200222 | `ed4d2e38ce7c` |
| `tu_imdb_multi_fold06` | complete | 5/40 | 23.4 | 0.90 | 200222 | `05780bad3122` |
| `tu_imdb_multi_fold07` | complete | 26/51 | 30.7 | 0.84 | 200222 | `131bc3158e63` |
| `tu_imdb_multi_fold08` | complete | 43/68 | 37.3 | 0.98 | 200222 | `94ba8b335189` |
| `tu_imdb_multi_fold09` | complete | 5/40 | 24.5 | 0.99 | 200222 | `f7e1d86ca086` |
| `tu_mutag_fold00` | complete | 30/47 | 5.0 | 0.04 | 207516 | `b293cf331883` |
| `tu_mutag_fold01` | complete | 12/40 | 5.0 | 0.04 | 207516 | `7bf05a5f9cf0` |
| `tu_mutag_fold02` | complete | 26/40 | 5.1 | 0.04 | 207516 | `679e5dbf02b3` |
| `tu_mutag_fold03` | complete | 26/40 | 5.2 | 0.04 | 207516 | `337a0d22d928` |
| `tu_mutag_fold04` | complete | 35/43 | 5.1 | 0.04 | 207516 | `2433043c8ddd` |
| `tu_mutag_fold05` | complete | 7/40 | 5.5 | 0.04 | 207516 | `ce0408f3feb3` |
| `tu_mutag_fold06` | complete | 25/44 | 5.8 | 0.04 | 207516 | `6e61279b4c03` |
| `tu_mutag_fold07` | complete | 10/40 | 5.4 | 0.04 | 207516 | `ef13efe489eb` |
| `tu_mutag_fold08` | complete | 30/55 | 6.8 | 0.04 | 207516 | `94bd4adf7cae` |
| `tu_mutag_fold09` | complete | 34/40 | 5.5 | 0.04 | 207516 | `f46a75673a1a` |
| `tu_nci1_fold00` | complete | 44/69 | 95.4 | 0.07 | 211692 | `48e388977091` |
| `tu_nci1_fold01` | complete | 49/74 | 122.7 | 0.07 | 211692 | `768de4762725` |
| `tu_nci1_fold02` | complete | 51/76 | 109.0 | 0.07 | 211692 | `47a6bbef4b18` |
| `tu_nci1_fold03` | complete | 70/95 | 158.3 | 0.07 | 211692 | `0c8332388fcc` |
| `tu_nci1_fold04` | complete | 37/62 | 88.0 | 0.07 | 211692 | `9b69207821c8` |
| `tu_nci1_fold05` | complete | 83/108 | 150.7 | 0.07 | 211692 | `683fb0a7de02` |
| `tu_nci1_fold06` | complete | 55/77 | 118.1 | 0.07 | 211692 | `f20d21495c8f` |
| `tu_nci1_fold07` | complete | 64/89 | 147.2 | 0.07 | 211692 | `5a04be1e8f56` |
| `tu_nci1_fold08` | complete | 42/67 | 93.5 | 0.07 | 211692 | `6fc62e14da51` |
| `tu_nci1_fold09` | complete | 47/72 | 96.1 | 0.07 | 211692 | `d3e7487b1517` |
| `tu_proteins_fold00` | complete | 8/40 | 23.4 | 0.16 | 192100 | `126f5e4c57df` |
| `tu_proteins_fold01` | complete | 52/77 | 42.7 | 0.16 | 192100 | `74124ba8ba0f` |
| `tu_proteins_fold02` | complete | 17/42 | 23.4 | 0.17 | 192100 | `b77499f2f4ab` |
| `tu_proteins_fold03` | complete | 19/44 | 19.7 | 0.19 | 192100 | `fd2fd8de4d14` |
| `tu_proteins_fold04` | complete | 8/40 | 17.8 | 0.16 | 192100 | `5ae395cf0a58` |
| `tu_proteins_fold05` | complete | 16/41 | 23.9 | 0.15 | 192100 | `1dac093c3833` |
| `tu_proteins_fold06` | complete | 34/59 | 26.1 | 0.18 | 192100 | `4db18a73ee82` |
| `tu_proteins_fold07` | complete | 52/77 | 41.0 | 0.19 | 192100 | `f1ea0db3e5f2` |
| `tu_proteins_fold08` | complete | 4/40 | 15.8 | 0.16 | 192100 | `17df150f8407` |
| `tu_proteins_fold09` | complete | 21/40 | 17.6 | 0.17 | 192100 | `3e302c0a3a2a` |

## 附录 B：文件索引

- `PROTOCOL.md`、`configs/protocol.json`：冻结协议与机器可读配置。
- `analysis/formal_plan.json`：102-task 顺序；`analysis/rr_runs.json`：worker/重跑/汇总的 rr run_id 与退出状态；`analysis/skipped_tasks.jsonl`：被阻塞跳过的任务。
- `analysis/*_data_manifest.json`：TU 原始文件与 SHA-256；`analysis/*_split.json`：OGB 官方 split 行号与 hash；`analysis/*_folds.json`：TU folds。
- `analysis/*_cache_stats.json`、`*_precheck_counts.json`、`*_graph_distribution.npz`：缓存/S2/预检查统计。
- `analysis/correctness_gates.json`、`tests/test_bp_core.py`：门禁结果。
- `results/tu_folds.csv`、`results/tu_summary.csv`、`results/mol_seeds.csv`、`results/peptide_seeds.csv`、`results/positioning_*.csv`、`results/resource_ledger.csv`、`results/failure_list.csv`、`results/metric_recheck.csv`、`results/run_manifest.csv`、`results/summary.json`、`results/budget.json`。
- 大型 cache、checkpoint 和 `runs/` 未提交；`analysis/prechecks/*/checkpoint.pt` 也未提交。
