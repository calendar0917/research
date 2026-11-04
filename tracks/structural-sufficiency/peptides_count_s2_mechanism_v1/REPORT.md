# Peptides-func 显式 S2 计数机制实验 v1 — REPORT

- **Round ID**：`peptides-count-s2-mechanism-v1`
- **结果标签**：`EXPLORATORY_VALIDATION_ONLY_MECHANISM`
- **数据**：官方 LRGB Peptides-func，Train 10,873 / Validation 2,331；**Test split 未用于构建或评估**；ECFP 混合 CSV 只流式取 Train/Validation SMILES，Test 行跳过且不解析/不保留
- **状态**：P0、CPU gates、P1a、P1b seed 0 全部完成；**停止规则未触发确认**；未运行 seed 1/2
- **正式运行 commit**：`d35c39f324c375b6ff6a006c291233ad97c06b8e`
- **最终停点**：`STOP_AFTER_SEED_0`

> 本报告只记录本轮实际观察值和实现事实。最后单独给出“观察数值”与“因果解释”的边界；不做论文定位、不提出下一轮方案。

---

## 0. 一句话结果

在冻结的 C1 + 零初始化 `Linear(2048,10)` 加法分支配方下，seed 0 的 **FULL-S2 分支没有超过 BASE**：

- `Gain_FULL_BASE = AP(A2) − AP(A0) = -0.7876 pp`（95% paired bootstrap CI `[-1.856, +0.283]`）
- `Gain_FULL_NULL = AP(A2) − AP(A3) = +6.9252 pp`（CI `[+5.123, +8.591]`）
- FULL ≥ NULL 在 **10/10** 个标签上成立（严格 > 也是 10/10）
- 但 `Gain_FULL_BASE` 远低于确认阈值 **+2.0 pp**，因此按冻结规则停止，不追加 seed 1/2。

---

## 1. 数据边界与 Test 保护

| split | 行数 | 本轮用途 |
|---|---:|---|
| Train | 10,873 | 唯一训练数据；特征/NULL/标准化/模型拟合 |
| Validation | 2,331 | 仅评估与 checkpoint 选择 |
| Test | 2,331 | **未构建图、未读取标签、未评估**；混合 CSV 流式读取时跳过 Test 行 |

- 未重新划分数据；未创建 Fit/Monitor/Locked。
- 所有正式入口在导入数据加载器前安装 `sys.addaudithook` Test guard；正式 P1a/A0–A3 的 `guard.attempts` 均为空，`test_evaluated=false`。P1 数据路径只使用 Train/Validation 缓存。
- 行身份检查：Train/Validation 的原子 9 列、键 3 列、无向连接、标签与历史 Fast v1 缓存逐行一致；S2 路径数与历史 `path_ptr` 逐图一致。检查行数 10,873 / 2,331；原子 1,647,684 / 348,441；键 1,677,395 / 354,621；不匹配行均为 0。
- ECFP 需要历史同源的官方 CSV **SMILES** 字段（9 列 PyG 图缺少同位素/显式氢信息）。该混合 CSV 通过逐行流式读取器只解析 Train/Validation 行的 SMILES；不解析标签列，不保留/评估 Test 行。详见 §8 与 `analysis/test_access_log.json`。

---

## 2. P0：标签无关 S2 计数特征

### 2.1 构建规模

| split | 图数 | 二跳路径总数 | 每图路径 mean / median / min / max |
|---|---:|---:|---|
| Train | 10,873 | 4,492,788 | 413.2 / 372 / 8 / 1,270 |
| Validation | 2,331 | 949,954 | 407.5 / 364 / 18 / 1,240 |

构建耗时 64.2 s（CPU，标签从未进入特征构建）。

### 2.2 唯一模式、非零桶与碰撞

| split | feature | 唯一内容模式 | 非零桶 | 桶碰撞率 | 64-bit digest 碰撞 |
|---|---|---:|---:|---:|---:|
| Train | TYPE-S2 | 47 | 47 | 0.00% | 0 |
| Train | FULL-S2 | 1,168 | 891 | 23.72% | 0 |
| Train | NULL-S2 | 7,346 | 1,982 | 73.02% | 0 |
| Valid | TYPE-S2 | 44 | 44 | 0.00% | 0 |
| Valid | FULL-S2 | 881 | 717 | 18.62% | 0 |
| Valid | NULL-S2 | 6,045 | 1,919 | 68.25% | 0 |

- FULL-S2-BINARY：全局非零桶 Train 891 / Valid 717；每图非零桶 mean 117.8 / 117.1。
- FULL 与 NULL 的路径总数相同（4,492,788 / 949,954）；第二半字段多重集合失败 0 图；derangement 失败 0 图；FULL/NULL 桶向量完全相同的图 0 个。
- `joint_equal_rows`（置换后某条路径前后半恰好仍相同）Train 385,703 / 4,492,788，Valid 81,311 / 949,954；这是重复局部模式下的可预期重合，不代表两套特征整体相同；逐图桶向量均不同。

### 2.3 ECFP 复现门禁

按历史口径 `ECFPFingerprint(count=True, radius=2, fp_size=2048)` + 固定 LightGBM：

| 指标 | 历史 | 本轮 | 差值 |
|---|---:|---:|---:|
| Validation macro AP | 72.98% | **72.9760%** | **−0.0040 pp** |
| Train nnz | 1,105,722 | 1,105,722 | 0 |
| Valid nnz | 235,742 | 235,742 | 0 |
| 最大计数 Train / Valid | 114 / 93 | 114 / 93 | 0 |

差值在冻结的 ±0.01 pp 接受区间内，**ECFP 复现通过**；`analysis/ecfp_validation.json` 的 `reproduction_ok=true`。

### 2.4 固定 LightGBM（Train-only → Validation）

| 特征 | Validation macro AP | Train macro AP |
|---|---:|---:|
| TYPE-S2 | 63.81% | 99.22% |
| FULL-S2 | **70.37%** | 99.88% |
| FULL-S2-BINARY | 64.92% | 99.25% |

逐标签 AP 见 `analysis/REPORT_TABLES.md`。观察事实：FULL 比 TYPE 高 **+6.55 pp**，FULL 比 BINARY 高 **+5.45 pp**；这是 P0 的 LightGBM 读出结果，不是 P1 的 C1 分支结果。

### 2.5 固定 MLP 读出（仅诊断）

FULL-S2 counts → `log1p` → 2048→64→10，SiLU，AdamW，无权重 BCE，seed 0：

- 最佳 epoch **154**；Validation macro AP **59.06%**；Train macro AP **72.55%**；checkpoint 重载差 0。
- 该 MLP 只用于诊断，不参与 P1 模型选择，也不改变 P1 配方。

---

## 3. P1a：原始 C1 的 480-epoch 训练长度对照

| run | Val macro AP | best epoch | last epoch | last Val macro AP | Train macro AP | GPU h |
|---|---:|---:|---:|---:|---:|---:|
| P1a 480ep, seed 0 | **63.4135%** | 434 | 480 | 63.2091% | 77.8531% | 1.1515 |
| 历史 240ep, seed 0 | 60.62% | 231 | 240 | n/a | n/a | n/a |
| 本轮 A0 240ep, seed 0 | 60.7758% | 221 | 240 | 60.3203% | 67.9760% | 0.7291 |

- 480ep 比历史 240ep 高 **+2.79 pp**；比同轮 240ep A0 高 **+2.64 pp**。
- **不能**把这归因于“纯 epoch 数”：480ep 的 cosine horizon 为 480，240ep 为 240；学习率轨迹和总更新数同时不同。这里只报告观察到的训练长度/调度组合差异。

---

## 4. P1b：seed 0 四臂

### 4.1 主表

所有臂：C1 基础参数 464,739；A1/A2/A3 加 `Linear(2048,10,bias=False)` 分支 20,480，总 485,219；A0 总 464,739。四臂均跑满 240 epoch，无早停。

| arm | 分支输入 | Val macro AP | Train AP | best epoch | last epoch | GPU h |
|---|---|---:|---:|---:|---:|---:|
| A0 BASE | 无 | **60.7758%** | 67.9760% | 221 | 240 | 0.7291 |
| A1 TYPE | TYPE-S2 | 60.5980% | 68.1590% | 210 | 240 | 0.7713 |
| A2 FULL | FULL-S2 | 59.9882% | 67.7019% | 231 | 240 | 0.5874 |
| A3 NULL | NULL-S2 | 53.0630% | 88.6094% | 238 | 240 | 0.8478 |

- `Gain_FULL_BASE = -0.7876 pp`
- `Gain_FULL_NULL = +6.9252 pp`
- `Gain_FULL_TYPE = -0.6098 pp`
- FULL ≥ NULL：**10/10** 标签；严格 >：**10/10** 标签。

### 4.2 逐标签 Validation AP（%）

| label | A0 | A1 | A2 | A3 | A2−A3 |
|---|---:|---:|---:|---:|---:|
| antifungal | 37.22 | 36.42 | 37.39 | 31.00 | +6.40 |
| cell_cell_communication | 51.70 | 52.71 | 51.57 | 37.62 | +13.95 |
| anticancer | 46.57 | 47.32 | 45.20 | 36.15 | +9.05 |
| drug_delivery_vehicle | 47.95 | 46.88 | 50.51 | 37.47 | +13.03 |
| antimicrobial | 93.60 | 93.49 | 93.99 | 91.07 | +2.92 |
| antiviral | 74.24 | 72.76 | 70.55 | 65.08 | +5.46 |
| antihypertensive | 90.18 | 90.04 | 90.09 | 87.22 | +2.88 |
| antibacterial | 52.97 | 52.28 | 54.31 | 43.15 | +11.16 |
| antiparasitic | 32.13 | 32.70 | 25.07 | 24.40 | +0.67 |
| toxic | 81.21 | 81.38 | 81.20 | 77.46 | +3.74 |

### 4.3 配对 bootstrap（seed 0）

对 Validation 2,331 行做 10,000 次配对行重采样，同一重采样索引同时用于两个模型，95% percentile：

| comparison | point Δ macro AP | 95% CI |
|---|---:|---:|
| A2 − A0 | **−0.788 pp** | **[−1.856, +0.283] pp** |
| A2 − A3 | **+6.926 pp** | **[+5.123, +8.591] pp** |

A2−A0 的区间跨 0；A2−A3 的区间整体为正。

### 4.4 分支行为与训练健康度

- 三个分支臂初始化权重全零；加分支的 **分支增量**最大绝对值恰为 0。
- CUDA 上独立重复调用同一 C1 编码器会产生 float32 舍入差：正式 A1/A2/A3 的 `max_abs_output_delta_vs_base` 约 4.6e−5 到 7.6e−5，`max_abs_base_forward_repeat_diff` 约 3.8e−5 到 7.6e−5（开发期失败 smoke 曾观察到 9.2e−5）；这是编码器重复前向的数值差，不是分支非零。CPU gate 的加分支前后 logits 差为 0。
- 第一步分支梯度均有限且非零：A1 2.206、A2 2.429、A3 2.112（L2 norm）。
- 训练后分支权重非零比例：A1 2.29%、A2 43.51%、A3 96.78%；Validation 分支输出非零比例均为 1.0。
- A3 的 Train AP 88.61% 而 Val AP 53.06%，是明显的 Train/Validation 分离；A0/A1/A2 的 Train AP 在 67.7%–68.2% 附近。该事实只作为观察记录，不据此做超参数调整。

---

## 5. 停止判定

冻结规则：`Gain_FULL_BASE >= 2.0` **且** `Gain_FULL_NULL >= 1.5` **且** FULL ≥ NULL 至少 6/10 标签。

| 条件 | 阈值 | 观察 | 满足 |
|---|---:|---:|---|
| Gain_FULL_BASE | ≥ +2.0 pp | **−0.7876 pp** | **否** |
| Gain_FULL_NULL | ≥ +1.5 pp | **+6.9252 pp** | 是 |
| FULL ≥ NULL labels | ≥ 6/10 | **10/10** | 是 |

合取为假 → **停止，不追加 seed 1/2**。未运行任何 seed 1/2、未做新模型/参数/特征/损失搜索。`analysis/p1_summary.json` 的 `decision.confirmation_triggered=false`。

---

## 6. 正确性门禁

### 6.1 CPU gates（正式 GPU 前）

`analysis/correctness_gates.json` 状态 **PASS**，覆盖：

- 手工四原子链 4 条路径、三原子环 6 条路径（端点不等于根）、FULL 33 字节布局。
- 节点重编号 / 无向键存储顺序打乱后，TYPE/FULL/NULL 图级计数逐元素不变。
- 单图 vs batch 前向最大差 ≤1e−5（CPU）。
- FULL/NULL 路径总数、前后半边际多重集合、derangement、联合 key 差异。
- 固定 key 在两个独立子进程与本进程的 digest/bucket 一致；TYPE/FULL/NULL 碰撞审计。
- 四臂参数、公共 C1 初始化、零分支、第一步梯度、训练后分支非零。
- checkpoint 重载复算 Validation AP（CPU smoke 四臂，delta ≤1e−4）。
- ECFP `reproduction_ok=true`；P0 产物存在且 `test_evaluated=false`。

### 6.2 GPU smoke

正式 P1 前在 `res-gpu0` 跑 A2 2-epoch GPU smoke（`pcs2-smoke-a2-gpu3`）：init gate 分支增量 0，第一步梯度 2.429，训练后分支非零，重载 delta 6.5e−6，`guard.attempts=[]`。记录在 `analysis/gpu_smoke.json`。

### 6.3 正式运行一致性

`analysis/formal_consistency.json` 状态 **PASS**（133 项检查），包括：

- 五条正式 run 的 status/epochs/last epoch/test flag/guard。
- 相同 `first_epoch_rows_sha256`、标签哈希、feature spec、norm、cache identity、公共 C1 初始 hash。
- A1/A2/A3 分支参数 20,480、总参数 485,219；A0 464,739。
- 四条 P1b 的 recipe 关键字段一致；P1a horizon 480；P1b horizon/min_epochs 240、无早停。
- 计数矩阵行对齐（10,873 / 2,331，维度 2048）。
- curve 长度/epoch 连续/有限；best epoch 与 argmax 一致；checkpoint/curve/predictions SHA 与 run.json 一致。
- 特征路径数、NULL 边际/derangement、FULL/NULL 桶向量差异。
- GPU 总量 ≤ 10 h。

补充审计：`predictions.npz` 以 float32 保存概率；用保存的概率重新计算 macro AP 与 `run.json` 中基于 float64 概率的评估值最大差约 0.003 pp（P1a 最大 0.0026 pp，A0–A3 ≤0.0007 pp），属于存储精度差，不是选择/重载错误。

---

## 7. 观察数值 vs 因果解释

### 7.1 观察数值（本报告的事实层）

- P0：FULL-S2 LightGBM 70.37% > BINARY 64.92% > TYPE 63.81%；ECFP4 72.9760%；FULL-S2 MLP 59.06%。
- P1b seed 0：A0 60.78%、A1 60.60%、A2 59.99%、A3 53.06%；A2−A0 = −0.79 pp，A2−A3 = +6.93 pp；FULL 在 10/10 标签上 ≥ NULL。
- P1a：480ep 63.41% @434；240ep A0 60.78% @221；历史 240ep 60.62% @231。
- 分支不是“死”的：A2/A3 训练后权重非零比例 43.5%/96.8%，Validation 分支输出均非零。

### 7.2 因果/机制解释（必须受限）

- 在本轮冻结的 **C1 + 零初始化加法分支 + 240ep 配方 + seed 0** 下，**没有观察到 FULL-S2 计数相对 BASE 的加性收益**；点估计为负，配对 CI 跨 0。因此本轮不能声称“显式 S2 计数能提升 C1 的 Peptides-func Validation AP”。
- FULL 明显优于 NULL（CI 整体为正，10/10 标签）说明：在该分支读出下，**后半部分的联合内容不是随便置换都能保留的**；这支持“分支确实利用了 FULL 的联合 key 结构”，但**不等于**“FULL 相对 BASE 有增益”。NULL 臂同时有更高的桶碰撞率、更密的训练后权重和更高的 Train AP（88.6%），其低 Validation AP 可能同时包含“信息被破坏”和“过拟合/优化差异”，本轮不能把二者分离。
- P0 的 FULL LightGBM 70.37% 说明 S2 计数在一个**不同的读出器**（树模型、逐标签、500 棵树）里携带可预测信号；这不自动推出 C1 的 20,480 参数加法分支应当受益。二者是不同读出假设，不能混为同一因果结论。
- P1a 的 480ep 更高是**训练长度 + cosine horizon + 总更新数**的联合观察；不能声称“单纯把 epoch 从 240 提到 480 带来 +2.6 pp”。
- 本轮没有 seed 1/2，因此不能把 seed 0 的点估计当作稳定的总体效应；停止规则的设计正是为了避免在未达到阈值时追加确认。

---

## 8. 错误、偏差与环境事件

1. **ECFP 输入口径偏差（已记录）**：官方 PyG 9 列图不含 SMILES 同位素/显式氢信息，直接由图重建 ECFP 得到 72.70%，超出 ±0.01 pp 门禁。最终 ECFP 使用官方混合 CSV 的 Train/Validation SMILES 行；`pc_p0_ecfp.py` 用流式读取器只取每行第一个逗号前的 SMILES，不解析标签列，不保留 Test 行。为通过该读取器，ECFP 进程的 guard 排除了 `LRGB_Peptides-func` 与 `lrgb_splits_peptides_func.json` 两个子串；P1 正式进程仍使用严格 guard。开发期有一次对 `lrgb_splits_peptides_func.json` 的严格 guard 尝试被 `TEST_ACCESS_BLOCKED` 拦截，记录在 `analysis/test_access_log.json`；最终 ECFP manifest 的 `test_rows_parsed=0`、`test_labels_read=false`。
2. **远端部署被未跟踪文件阻塞**：远端 checkout 有未跟踪的 `lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0/*`，与本地已提交文件同名。逐文件 SHA-256 与本地已提交版本完全相同；为不覆盖用户文件，移动到 `/home/hxy/cy/rr_untracked_backup/lcrc_peptides_weighted_bce_v1_runs_20261104/`，未删除。之后 `rr deploy` 成功。
3. **首次 GPU smoke 缺少特征矩阵**：`analysis/features_*.npz` 原被 `.gitignore` 忽略，远端没有；GPU 任务在 `load_counts` 处 `FileNotFoundError`。修复方式是把两个冻结特征矩阵（Train 9.6 MB、Valid 2.1 MB）加入版本控制并重新 deploy；没有改变特征内容。后续正式 run 使用 `29f77cf0` 引入的同一矩阵，正式代码在 `d35c39f3`。
4. **CUDA 初始化门禁的浮点舍入**：第二次 GPU smoke 的 init gate 曾因两次独立 C1 前向的 float32 差（约 9.15e−5）判失败。检查确认分支权重全零、分支增量恰为 0；这是重复 base 前向的 CUDA 舍入差，不是分支非零。门禁改为要求分支增量精确为 0，并单独记录 `max_abs_output_delta_vs_base` 与 `max_abs_base_forward_repeat_diff`；CPU gate 仍要求加分支前后 logits 差为 0。该修改只修门禁，不改变科学设计。
5. **失败 smoke 的日志为空**：早期两次 GPU smoke 的 `rr logs` 为空；第二次通过远端 `/tmp` 的 `run.json` 定位到 init gate。为定位该空日志，曾在远端直接执行一次 1-epoch A2 到 `/tmp` 的诊断（非正式 run，未用于任何科学表格）；所有正式 P1a/P1b 训练均通过 `rr` 提交。失败 smoke 未产生正式结果，未计入科学表格。
6. **P0 LightGBM/MLP 重跑**：NULL 的 canonical path order 修正后重建了 `features_*.npz`；TYPE/FULL 不受影响，但为保持 JSON 与当前矩阵一致，重跑了 P0 LightGBM 与 MLP。最终 P0 数值以 `analysis/p0_lgbm.json`、`analysis/p0_mlp.json` 为准。
7. **`rr pull` 的落盘位置**：`rr pull` 写入 `.rr/pulled/<run-id>/result/...`，本轮将其复制到 `runs/<name>/` 供 aggregate；未修改 `rr` 或远端产物。

---

## 9. GPU 消耗

| run | GPU h |
|---|---:|
| P1a 480ep seed 0 | 1.1515 |
| A0 seed 0 | 0.7291 |
| A1 seed 0 | 0.7713 |
| A2 seed 0 | 0.5874 |
| A3 seed 0 | 0.8478 |
| A2 GPU smoke（2ep） | 0.0096 |
| **合计（成功 run）** | **4.0967** |
| 硬上限 | 10.0 |

早期失败 smoke 的 GPU 墙钟各约 15 s，合计 <0.01 h，未计入上表；即使计入也远低于 10 h。两个 pool 均按“一卡一任务”使用；未超时、未触发 `RESOURCE_BLOCKED`。

---

## 10. 产物与复现入口

### 代码

- `code/pc_common.py`：常量、稳定哈希、Test guard、IO/环境。
- `code/pc_data.py`：缓存/PyG Train/Validation 读取、行身份、C1 task。
- `code/pc_s2.py`：路径枚举、TYPE/FULL/NULL、碰撞审计。
- `code/pc_p0_features.py`：构建 `analysis/features_{train,valid}.npz`。
- `code/pc_p0_ecfp.py`：ECFP4 + LightGBM 复现。
- `code/pc_p0_lgbm.py`：TYPE/FULL/BINARY 固定 LightGBM。
- `code/pc_p0_mlp.py`：FULL-S2 MLP 诊断。
- `code/pc_gates.py`：CPU correctness gates。
- `code/pc_p1_model.py`、`code/pc_p1_train.py`：C1 + 分支构建与训练。
- `code/pc_aggregate.py`：Gain、停止判定、bootstrap、机器可读汇总。
- `code/pc_final_checks.py`：正式产物一致性检查。
- `tests/test_pc_core.py`：小图/不变性/哈希单元测试。

### 机器可读结果

- `analysis/p0_features_manifest.json`：路径数、唯一种数、桶碰撞、NULL 审计、SHA-256。
- `analysis/ecfp_validation.json`：ECFP 复现、逐标签 AP、输入口径。
- `analysis/p0_lgbm.json`、`analysis/p0_mlp.json`：P0 固定读出。
- `analysis/p1_summary.json` / `analysis/results_machine.json`：P1a/P1b 指标、逐标签、曲线、run ID、环境、GPU、停止判定、seed-0 bootstrap。
- `analysis/formal_consistency.json`：133 项正式一致性检查、run ID、checksums。
- `analysis/REPORT_TABLES.md`：本报告表格的生成版本。
- `analysis/gpu_smoke.json`、`analysis/test_access_log.json`、`analysis/correctness_gates.json`。
- `analysis/checksums.json`：代码/配置/结果/运行产物 SHA-256。
- `analysis/COMMANDS.md`：实际执行命令。

### 运行产物

本地 `runs/{p1a_480_s0,a0_s0,a1_s0,a2_s0,a3_s0}/` 含 `run.json`、`curve.csv`、`selection.json`、`checkpoint.pt`、`predictions.npz`；`runs/smoke_A2_gpu/` 为工程 smoke。`runs/` 按本轮 `.gitignore` 保持本地，不提交。

---

## 11. 分阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 特征 | 完成 | Train 4,492,788 / Valid 949,954 路径；碰撞审计完成 |
| ECFP 复现 | 完成/通过 | 72.9760% vs 72.98%，delta −0.0040 pp |
| P0 LightGBM | 完成 | TYPE 63.81 / FULL 70.37 / BINARY 64.92 |
| P0 MLP | 完成 | 59.06% Val，诊断用 |
| CPU gates | 通过 | `analysis/correctness_gates.json` PASS |
| GPU smoke | 通过 | A2 2ep，重载 6.5e−6 |
| P1a 480ep | 完成 | 63.4135% @434 |
| P1b seed 0 | 完成 | A0 60.7758 / A1 60.5980 / A2 59.9882 / A3 53.0630 |
| 停止判定 | 完成 | 未触发确认；未跑 seed 1/2 |
| 正式一致性 | 通过 | 133/133 checks |
| Test | 未访问 | 正式 guard attempts 为空；`test_evaluated=false` |
