# LCRC Peptides-func Diagnostic v1 — 事实报告

- **Round ID**：`lcrc-peptides-diagnostic-v1`
- **结果标签**：`EXPLORATORY_VALIDATION_ONLY_DIAGNOSTIC`
- **数据集**：仅 Peptides-func（官方 LRGB 固定划分）
- **主模型**：仓库已有 **LCRC-C1 / NO_BOND_MEAN**（`bp_model.build_c1`，与 Fast v1 逐字节同源）
- **正式 GPU 训练代码 commit**：`f2f2d1b0292f`（`f2f2d1b0292f77cbbd9b3e3d6fa9b0b7d1af08b1`）
- **P2 机器 / 运行窗口（UTC）**：`res` / `a100-2`，1× NVIDIA A100-SXM4-40GB（`CUDA_VISIBLE_DEVICES=0`），driver 550.163.01，torch 2.5.1+cu124；2026-11-04T01:04:12Z 至 01:43:34Z
- **P0 / P1 机器**：本地 CPU 工作站（无 GPU），torch 2.5.1+cu124（CPU）、numpy 2.1.3、scikit-learn 1.5.2
- **历史实验**：`tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/` 只读；本轮只复制不修改
- **报告口径**：只记录工程执行事实，不给研究路线结论

> **Test 政策**：本轮 P0.3 与 P2 的执行路径都安装了 `sys.addaudithook`，任何打开
> `peptides_func_test.pt` / `peptides_func_y_test.npy` / `test_predictions.npz` 的行为会立即抛
> `TEST_ACCESS_BLOCKED`。P2 `run.json` 记录的 `guard.attempts = []`（零次尝试）；P1 只
> 用官方 Train 训练、Validation 评价，`test_rows_read = 0`。历史三个 seed 的 Test AP
> （51.70 / 50.68 / 51.80）只用于核对历史文件，未参与本轮任何选择。

---

## 0. 本轮执行摘要

| 项 | 内容 | 状态 |
|---|---|---|
| P0.1 学习曲线审计 | 3 seeds × 120 epoch 曲线；best epoch / 最后 20 epoch 变化 | 完成 |
| P0.2 标签分布 | 10 标签 ×（Train, Validation）正负样本数、阳性率、常数 AP 基线 | 完成 |
| P0.3 冻结 checkpoint 逐标签 AP | 3 个 Fast v1 checkpoint 全部可用；Train + Validation 重算 | 完成（`CHECKPOINT_UNAVAILABLE` 未发生） |
| P1 Count ECFP4 + LightGBM | 行身份门禁 PASS；Validation macro AP **72.98 %** | 完成 |
| P2 LCRC-C1 240-epoch seed0 | 全程最佳 Validation mean AP **60.62 %** @ epoch 231 | 完成（无早停、无 OOM、无失败） |
| GPU 预算 1.25 A100 GPU-hours | 实际 **0.653 GPU-hours**（2350.8 s） | 未超预算 |
| 失败 / 阻塞 | 无 | — |

资源与门禁一句话：行身份核验覆盖 15,535 行 / 2,344,859 个原子，原子特征、键特征、连接
关系、标签、canonical SMILES 全部 0 处不一致；所有自动门禁（`analysis/gates.json`）为真。

---

## 1. P0.1 历史三个 seed 的学习曲线与最佳 epoch

数据来源：从 res 复制到 `history/peptide_seed{0,1,2}/{curve.csv,run.json,selection.json,checkpoint.pt}`，
SHA-256 与历史 `run.json` 记录逐项核对通过（`analysis/learning_curve_audit.json`）。

### 1.1 每 20 epoch 快照（`analysis/learning_curves.csv`）

| epoch | seed0 loss | seed1 loss | seed2 loss | seed0 Valid AP % | seed1 Valid AP % | seed2 Valid AP % |
|---|---|---|---|---|---|---|
| 1 | 0.67495 | 0.56186 | 0.62722 | 31.35 | 30.57 | 30.68 |
| 20 | 0.27361 | 0.27564 | 0.27532 | 44.70 | 45.22 | 46.13 |
| 40 | 0.25912 | 0.25777 | 0.25672 | 47.40 | 48.26 | 49.54 |
| 60 | 0.24715 | 0.24844 | 0.24380 | 50.55 | 51.36 | 51.58 |
| 80 | 0.23957 | 0.23963 | 0.23466 | 52.42 | 51.86 | 53.31 |
| 100 | 0.23278 | 0.23471 | 0.22952 | 53.34 | 53.52 | 54.11 |
| 120 | 0.23034 | 0.23230 | 0.22755 | 53.57 | 53.79 | 54.76 |

三个 seed 都跑满 120 epoch（没有提前结束；`max_epochs=120`、`min_epochs=40`、`patience=25`），
因此表中没有缺失 epoch。

### 1.2 汇总（`analysis/learning_curve_summary.csv`）

| seed | best epoch | epochs run | best Valid AP % | Valid AP @100 % | Valid AP @last % | Δ Valid AP last 20 (pp) | last-20 内出现新历史最佳 | train loss @100 | train loss @last | Δ train loss last 20 | lr @last |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 116 | 120 | **53.73** | 53.34 | 53.57 | +0.10 | 是（epoch 116） | 0.23278 | 0.23034 | −0.00229 | 1.00e-05 |
| 1 | 119 | 120 | **53.94** | 53.52 | 53.79 | +0.14 | 是（epoch 119） | 0.23471 | 0.23230 | −0.00183 | 1.00e-05 |
| 2 | 113 | 120 | **54.79** | 54.11 | 54.76 | +0.47 | 是（epoch 113） | 0.22952 | 0.22755 | −0.00219 | 1.00e-05 |

补充事实：最后 20 epoch 的 Validation mean AP 仍在上升（+0.10 / +0.14 / +0.47 pp），
训练 loss 仍在下降（−0.0018 至 −0.0023），且三个 seed 的最佳 epoch 都落在最后 8 个 epoch 内。
曲线图见 `analysis/learning_curves.png`；逐 epoch 全量数据见 `analysis/learning_curve_full.csv`。

### 1.3 历史 run 元数据（`analysis/history_runs.csv`）

| seed | 参数量 | 实际耗时 s | best epoch / 实际 epoch | peak lr | wd | dropout | batch | 曲线 SHA-256 前 12 | checkpoint SHA-256 前 12 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 464,739 | 1146.9 | 116 / 120 | 3e-4 | 1e-3 | 0.1 | 64 | `13f825ca12e1` | `1da869b87358` |
| 1 | 464,739 | 1261.0 | 119 / 120 | 3e-4 | 1e-3 | 0.1 | 64 | `c036446716a9` | `5140a1668446` |
| 2 | 464,739 | 1127.4 | 113 / 120 | 3e-4 | 1e-3 | 0.1 | 64 | `fa00c1430f3c` | `7c7dcf89e0eb` |

历史 Test mean AP 与历史文件核对一致（仅核对，不参与任何选择）：
`tracks/.../lcrc_benchmark_positioning_fast_v1/results/peptide_seeds.csv` → 51.70 % / 50.68 % / 51.80 %。

---

## 2. P0.2 十个标签的 Train/Validation 分布与常数 AP 基线

来源：官方 LRGB Peptides-func CSV（`data/lrgb_skfp/LRGB_Peptides-func/peptides_func.csv`，
SHA-256 `9d2e253d2c3f9770…`）+ 官方 split 索引文件（SHA-256 `39ca1b1400761460…`）。
Train 标签与 Fast v1 `peptides_func_y_train.npy`、Validation 标签与 `peptides_func_y_valid.npy`
逐行完全相同（`analysis/label_statistics.json`）。**未读取 Test 标签。**

| label | Train pos | Train n | Train pos % = Train 常数 AP 基线 | Valid pos | Valid n | Valid pos % = Valid 常数 AP 基线 |
|---|---|---|---|---|---|---|
| antifungal | 966 | 10873 | 8.88 | 207 | 2331 | 8.88 |
| cell_cell_communication | 385 | 10873 | 3.54 | 79 | 2331 | 3.39 |
| anticancer | 698 | 10873 | 6.42 | 150 | 2331 | 6.44 |
| drug_delivery_vehicle | 677 | 10873 | 6.23 | 146 | 2331 | 6.26 |
| antimicrobial | 6820 | 10873 | 62.72 | 1464 | 2331 | 62.81 |
| antiviral | 2148 | 10873 | 19.76 | 461 | 2331 | 19.78 |
| antihypertensive | 1162 | 10873 | 10.69 | 250 | 2331 | 10.73 |
| antibacterial | 2002 | 10873 | 18.41 | 433 | 2331 | 18.58 |
| antiparasitic | 217 | 10873 | 2.00 | 43 | 2331 | 1.84 |
| toxic | 2825 | 10873 | 25.98 | 609 | 2331 | 26.13 |

- 10 标签阳性率（非加权）均值：Train **16.463 %**，Validation **16.482 %**。
- 每样本阳性标签数均值：Train 1.6463，Validation 1.6482；两个划分都没有「零标签」样本。
- 最稀疏标签 `antiparasitic` Validation 只有 43 个正例；最密标签 `antimicrobial` 62.81 %。
- 文件：`analysis/label_statistics.csv`，校验：`analysis/label_statistics.json`。

---

## 3. P0.3 冻结 checkpoint 的逐标签 Train/Validation AP

三个 Fast v1 checkpoint 全部可用（未发生 `CHECKPOINT_UNAVAILABLE`），本轮只做前向重评，
**没有重训历史 seed**，也没有读取 Test 数据或 Test predictions。

- 重建模型与 checkpoint 严格一致：参数量 464,739、`local_layout` 一致、`rho_input_dim` 一致、
  **初始化 hash 与历史记录完全相同**（seed0/1/2 均为 `init_state_hash` 匹配）、feature spec hash、
  norm fingerprint、cache SHA-256 全部匹配。
- 独立重算的 Validation macro AP 与历史记录一致（差异来自 CPU 重放的 CUDA 归约噪声）：

| seed | 历史 best epoch | 历史记录 macro AP % | 本次重算 macro AP % | 差 (pp) | Train macro AP % | 前向耗时 s |
|---|---|---|---|---|---|---|
| 0 | 116 | 53.7303 | 53.7350 | +0.0047 | 56.4603 | 8.93 |
| 1 | 119 | 53.9360 | 53.9422 | +0.0062 | 56.0369 | 9.28 |
| 2 | 113 | 54.7917 | 54.7972 | +0.0056 | 57.2187 | 9.04 |

### 3.1 逐标签 AP（`results/p0_frozen_checkpoints_per_label.csv`）

| label | Valid pos %（基线） | seed0 Train AP % | seed1 Train AP % | seed2 Train AP % | seed0 Valid AP % | seed1 Valid AP % | seed2 Valid AP % | Valid AP 均值 % | Valid AP − 阳性率（均值, pp） |
|---|---|---|---|---|---|---|---|---|---|
| antifungal | 8.88 | 33.32 | 33.44 | 33.55 | 34.30 | 34.00 | 33.45 | 33.92 | +25.03 |
| cell_cell_communication | 3.39 | 39.53 | 39.45 | 39.80 | 40.73 | 42.48 | 40.64 | 41.29 | +37.90 |
| anticancer | 6.44 | 44.62 | 43.97 | 46.01 | 37.91 | 39.18 | 39.75 | 38.95 | +32.51 |
| drug_delivery_vehicle | 6.26 | 39.55 | 37.91 | 39.94 | 38.72 | 36.67 | 41.18 | 38.86 | +32.59 |
| antimicrobial | 62.81 | 92.84 | 92.66 | 92.98 | 91.36 | 91.39 | 91.81 | 91.52 | +28.71 |
| antiviral | 19.78 | 68.63 | 65.99 | 70.14 | 65.11 | 63.11 | 66.86 | 65.03 | +45.25 |
| antihypertensive | 10.73 | 90.84 | 90.69 | 91.07 | 88.86 | 89.20 | 89.36 | 89.14 | +78.41 |
| antibacterial | 18.58 | 53.11 | 53.40 | 53.41 | 48.28 | 50.57 | 49.51 | 49.45 | +30.88 |
| antiparasitic | 1.84 | 26.22 | 27.60 | 28.98 | 15.17 | 16.30 | 18.09 | 16.52 | +14.68 |
| toxic | 26.13 | 75.94 | 75.29 | 76.30 | 76.91 | 76.51 | 77.33 | 76.92 | +50.79 |

事实备注：以 seed0 为例，10 个标签中 7 个的 Train AP 高于 Validation AP
（anticancer +6.71、drug_delivery_vehicle +0.83、antimicrobial +1.48、antiviral +3.52、
antihypertensive +1.98、antibacterial +4.83、antiparasitic +11.05 pp），
3 个略低（antifungal −0.98、cell_cell_communication −1.20、toxic −0.97 pp）。
`antiparasitic` 的 Validation AP 均值 16.52 % 是 10 个标签中最低的（其 Validation 正例仅 43 个）。

---

## 4. P1：Count ECFP4 + LightGBM（Train → Validation）

### 4.1 行身份与数据门禁（`analysis/row_identity.json`）

| 检查 | 结果 |
|---|---|
| CSV 行数 / split 尺寸 | 15,535 行；Train 10,873、Valid 2,331、Test 2,331 |
| split 索引 | 互斥且完整覆盖 15,535 行 |
| 标签 | CSV 行标签 == PyG `y` 行 == Fast v1 `y_{train,valid}.npy` 行（0 处不一致） |
| 分子图身份 | 逐行、逐原子（9 个离散列）、逐键（3 个离散列 + 双向连接）与 PyG 图比较：**0 处不一致** |
| 覆盖规模 | 15,535 行 / 2,344,859 个原子全部比较 |
| 二次证据 | 由 PyG 特征重建的 RDKit 分子与 CSV 分子的 canonical（非异构）SMILES 0 处不一致 |
| 状态 | **PASS**（非 `ROW_ALIGNMENT_BLOCKED`） |

因此 `skfp` 的 SMILES 行顺序、官方 split 索引与 Fast v1 使用的 `LRGBDataset` 划分是同一套行身份。

### 4.2 固定配置与特征

- `skfp.fingerprints.ECFPFingerprint(count=True, radius=2, fp_size=2048)`，只覆盖这三个参数，
  原子不变量取 skfp/RDKit 默认（`use_pharmacophoric_invariants=False`、`include_chirality=False`、
  `use_bond_types=True`、`include_ring_membership=True`、`count_simulation=False`）；无额外描述符。
- 特征矩阵：Train `10873 × 2048`（非零 1,105,722，最大计数 114），
  Validation `2331 × 2048`（非零 235,742，最大计数 93）；无空行；重复计算逐元素相同。
- 标签从未进入特征：fingerprint 只接收由 SMILES 构造的 RDKit 分子（`labels_used_as_features=false`）。
- `LGBMClassifier(n_estimators=500, is_unbalance=True, importance_type="gain", random_state=0,
  n_jobs=8, verbose=-1)`，其余 LightGBM 默认值；每标签独立模型；无调参、无早停、无树数搜索。

### 4.3 结果（`results/ecfp_validation.csv`、`results/ecfp_per_label.csv`）

**Validation macro AP（10 标签非加权平均）= 72.98 %**（16.482 % 常数基线之上 +56.49 pp；
独立 sklearn 重算差 0.0）。

| label | Train pos % | Valid pos %（基线） | Valid AP % | AP − 基线 (pp) |
|---|---|---|---|---|
| antifungal | 8.88 | 8.88 | 50.30 | +41.42 |
| cell_cell_communication | 3.54 | 3.39 | 73.51 | +70.12 |
| anticancer | 6.42 | 6.44 | 66.17 | +59.74 |
| drug_delivery_vehicle | 6.23 | 6.26 | 69.23 | +62.97 |
| antimicrobial | 62.72 | 62.81 | 97.23 | +34.43 |
| antiviral | 19.76 | 19.78 | 86.71 | +66.93 |
| antihypertensive | 10.69 | 10.73 | 92.04 | +81.32 |
| antibacterial | 18.41 | 18.58 | 68.59 | +50.02 |
| antiparasitic | 2.00 | 1.84 | 35.32 | +33.47 |
| toxic | 25.98 | 26.13 | 90.66 | +64.53 |

耗时：加载 + 指纹 2.15 s，10 个 LightGBM 合计 6.16 s（单标签 0.56–0.68 s），步骤总 12.45 s。
Validation 正类概率矩阵：`results/ecfp_valid_predictions.npz`（2331 × 10，尺寸/范围检查通过）。

**口径提醒**：文献中 ECFP4+LightGBM 的 74.60 % Test AP 使用 Train+Validation 训练、Test 评价；
本行 72.98 % 是 Train-only 训练、Validation 评价，两者不可直接比较。

**重复性事实**：为记录 CPU 步骤耗时，P1 在本轮被完整独立重跑一次（同解释器、同配置）。
两次运行的 10 个 Validation AP 差别 < 1e-12（macro AP 相同），Validation 概率最大绝对差
4.1e-15；两次结果只在计时列与浮点位级别不同。

---

## 5. P2：一次 LCRC-C1 延长训练（Peptides-func / seed 0 / 240 epoch）

### 5.1 结构门禁与训练配置

- 结构与 Fast v1 一致：参数量 **464,739**（与历史相同）、`local_layout` 一致、
  **`init_state_hash` 与 Fast v1 seed0 checkpoint 完全相同**、feature spec hash 一致、
  Train-only norm fingerprint 一致、Train/Valid cache SHA-256 一致。
- 配方（`configs/protocol.json` → `p2_long_run`）：AdamW，peak LR 3e-4，weight decay 1e-3，
  dropout 0.1，batch 64，gradient clip 5.0，BCEWithLogitsLoss，per-step cosine horizon **240**，
  max 240 epoch，min 80 epoch，patience 40，按 Validation mean AP 选 checkpoint；从头训练。
- 与历史唯一的配方差异是 epoch 上限/最小 epoch/patience/cosine horizon；其余完全沿用。
- 运行：240/240 epoch 跑满，无早停；无 OOM、无非有限梯度、无实现错误。
- Test 访问：audit hook 记录 0 次尝试（`guard.attempts = []`），`test_evaluated = false`。

### 5.2 结果（`runs/peptide_c1_seed0_240ep/`）

| 项 | 数值 |
|---|---|
| 前 120 epoch 最佳 Validation mean AP | **57.72 %**（epoch 120） |
| 第 121–240 epoch 最佳 Validation mean AP | **60.62 %**（epoch 231） |
| 全程最佳 Validation mean AP | **60.62 %** |
| 全程最佳 epoch | **231**（240 epoch 中） |
| 相对 Fast v1 seed0 历史最佳（53.73 %）差值 | **+6.89 pp** |
| 前 120 epoch 相对 Fast v1 seed0 差值 | +3.99 pp |
| 第 121–240 相对前 120 最佳差值 | +2.91 pp |
| 最后 epoch（240）Validation mean AP | 60.27 % |
| Train macro AP / Validation macro AP（选中 checkpoint） | 67.94 % / 60.63 % |
| 重载 checkpoint 复现的 Validation macro AP | 60.628 %（与选择值差 5.4e-5） |

关键 epoch（`runs/peptide_c1_seed0_240ep/curve.csv` 含全部 240 epoch 的 train loss、
Validation mean AP、learning rate、每 epoch 训练/验证秒数与累计耗时）：

| epoch | lr | train loss | Validation mean AP % |
|---|---|---|---|
| 1 | 3.00e-04 | 0.67557 | 32.00 |
| 20 | 2.95e-04 | 0.27377 | 44.71 |
| 40 | 2.81e-04 | 0.25891 | 47.62 |
| 60 | 2.58e-04 | 0.24456 | 50.97 |
| 80 | 2.28e-04 | 0.23284 | 53.95 |
| 100 | 1.93e-04 | 0.22106 | 55.61 |
| 120 | 1.55e-04 | 0.21298 | 57.72 |
| 140 | 1.17e-04 | 0.20682 | 58.91 |
| 160 | 8.25e-05 | 0.20143 | 59.08 |
| 180 | 5.25e-05 | 0.19835 | 60.05 |
| 200 | 2.94e-05 | 0.19579 | 60.58 |
| 220 | 1.49e-05 | 0.19435 | 60.38 |
| 231 | 1.10e-05 | 0.19362 | **60.62** |
| 240 | 1.00e-05 | 0.19380 | 60.27 |

### 5.3 逐标签 Validation AP（选中 checkpoint）

| label | Valid pos %（基线） | Train AP % | Valid AP % | 相对 Fast v1 seed0 Valid AP (pp) |
|---|---|---|---|---|
| antifungal | 8.88 | 45.72 | 36.97 | +2.67 |
| cell_cell_communication | 3.39 | 58.99 | 50.63 | +9.90 |
| anticancer | 6.44 | 57.44 | 46.97 | +9.06 |
| drug_delivery_vehicle | 6.26 | 56.25 | 47.68 | +8.96 |
| antimicrobial | 62.81 | 95.85 | 93.57 | +2.21 |
| antiviral | 19.78 | 80.53 | 74.11 | +9.00 |
| antihypertensive | 10.73 | 94.13 | 90.09 | +1.23 |
| antibacterial | 18.58 | 62.63 | 52.94 | +4.66 |
| antiparasitic | 1.84 | 45.00 | 32.09 | +16.92 |
| toxic | 26.13 | 82.90 | 81.23 | +4.32 |

### 5.4 与历史 seed0 的可比性事实

- cosine horizon 从 120 改成 240，因此 LR 轨迹不同（本 run epoch 120 时 LR 仍为 1.55e-04，
  历史 run epoch 120 时 LR 已降到 1.00e-05）。上述差值**包含学习率轨迹变化**，不能解释为
  「单纯增加 epoch」的因果收益。
- 两个 run 的初始权重完全相同（`init_state_hash` 相同、参数量相同），但**不是逐位可复现**：
  本 run epoch 1 的 train loss 0.67557 / Validation AP 32.00 %，与历史 seed0 的
  0.67495 / 31.35 % 不同。仓库已记录 CUDA `index_add_` 归约不保证逐位确定；
  本轮未进一步追查该数值差异（超出工程执行范围）。

---

## 6. 缺失、失败或阻塞

| 项目 | 状态 | 说明 |
|---|---|---|
| P0.1 / P0.2 | 完成 | 无缺失 epoch；未读取 Test 标签 |
| P0.3 | 完成 | 3/3 checkpoint 可用；无 `CHECKPOINT_UNAVAILABLE` |
| P1 | 完成 | 行身份门禁 PASS；无 `ROW_ALIGNMENT_BLOCKED`；未训练第二套分类器、未尝试其他 fingerprint family |
| P2 | 完成 | 240/240 epoch；未早停；无 OOM / 非有限梯度 / 实现错误；Test 零访问 |
| GPU 预算 | 未超 | 1.25 A100 GPU-hours 预算，实际 0.653 GPU-hours |

其他事实记录：

1. 历史 Fast v1 seed0 第一次尝试曾因 `PER_TASK_TIME_LIMIT` 阻塞（属上一轮历史，非本轮失败）；
   本轮的延长训练在 4200 s 上限内自然结束（2350.8 s），未触发时间上限。
2. 为执行 `rr deploy`，res 上有 26 个 Fast v1 分析/结果文件处于「未跟踪但内容与本地已提交版本
   **逐字节相同**」的状态，被移动到 `/tmp/rr-untracked-backup-20261104T010328Z/` 后由部署检出；
   历史文件内容未被修改（26 个文件 SHA-256 在移动前逐一核对为 OK）。
3. P2 `run.json` 中 `environment.git_dirty = true` 的原因是运行过程在 `runs/` 下新建了未跟踪的
   输出目录；`rr` 在启动时记录 `git_dirty=false`、`git_diff_hash=e3b0c44298fc1c14`（空），
   运行结束后 res 上 `git status --porcelain` 无任何被跟踪文件改动。见 `analysis/p2_provenance.json`。

---

## 7. 资源消耗、版本、哈希与复现

### 7.1 CPU 消耗（本地，`analysis/cpu_steps.json`）

| 步骤 | 墙钟秒 |
|---|---|
| P0.1 学习曲线审计 | 1.62 |
| P0.2 标签统计 | 1.23 |
| 行身份核验（15,535 行逐原子/逐键） | 43.47 |
| P0.3 三个 checkpoint 重评（CPU） | 48.18 |
| P1 ECFP4 + LightGBM | 14.45 |
| 汇总 | 1.12 |
| 合计 | **110.07** |

### 7.2 GPU 消耗（P2）

| 项 | 数值 |
|---|---|
| GPU | 1× A100-SXM4-40GB（`res` / `a100-2`，`CUDA_VISIBLE_DEVICES=0`） |
| run.json 计时 | 2350.79 s（train 2036.69 s + Validation 300.16 s + 收尾） |
| rr 墙钟 | 2362 s（01:04:12Z → 01:43:34Z） |
| **GPU-hours** | **0.653**（预算 1.25） |
| 平均每 epoch | 9.74 s（240 epoch） |
| GPU peak allocated / reserved | 357.85 MB / 773.85 MB |
| grad clip 触发次数 / 最大梯度范数 | 2918 / 229.676 |
| checkpoint SHA-256 | `22643f093cc96dd3d8a81f916d7a6cc6e4bb38a91faf405eaf222fb7bf57634b` |
| curve.csv SHA-256 | `f023ca5f39afbf650f0cbad6b53ab6f624ca23ae21f2263aa6ea69b76e5e5948` |

### 7.3 版本

| 环境 | 版本 |
|---|---|
| 仓库 GPU 环境（P0.3 CPU / P2 GPU） | python 3.12.14/3.12.15，torch 2.5.1+cu124（CUDA 12.4），numpy 2.1.3，scikit-learn 1.5.2，torch-geometric 2.6.1 |
| ECFP 隔离 CPU 环境 | scikit-fingerprints 2.1.0，lightgbm 4.7.0，rdkit 2026.3.5，scikit-learn 1.9.1，numpy 2.5.4，pandas 3.1.0，scipy 1.18.1 |
| 说明 | ECFP/LightGBM 装在独立 venv（`/home/calendar/.cache/research-venvs/peptides-ecfp`），未改动仓库 `pyproject.toml` / `uv.lock` 与 GPU 环境 |

### 7.4 数据与模型哈希

| 对象 | SHA-256 |
|---|---|
| SMILES CSV `peptides_func.csv` | `9d2e253d2c3f9770edda82ba467117d17b9f119297c8166046de07b0046db98f` |
| 官方 split JSON | `39ca1b1400761460c381fa407a234881bb617c8df06f7bd39fe11c70fcf85b14` |
| Fast v1 `peptides_func_y_train.npy` | `fd527b3318b3b7d6a8fa98af8424b66704f32cae5d5e7e7f9e5278af884c3ba9` |
| Fast v1 `peptides_func_y_valid.npy` | `ef29af2c48869c35422ba62c75f575c4e45a40ed2e7951c72135bf2fcd45ecb0` |
| Train cache `peptides_func_train.pt` | `7b57289e8a7cc6174c6c356124c333d03d1cbeff11cf0fca0ef66826608b0ad1` |
| Valid cache `peptides_func_valid.pt` | `6c1d66677effdac1fa8ca8f25d457e962b89a837c5416985c6c10b717418be6c` |
| Fast v1 checkpoint seed0 / 1 / 2 | `1da869b87358…` / `5140a1668446…` / `7c7dcf89e0eb…`（全量在 `analysis/history_runs.csv`） |
| P2 checkpoint | `22643f093cc96dd3d8a81f916d7a6cc6e4bb38a91faf405eaf222fb7bf57634b` |
| split 索引哈希（skfp 顺序） | `617d4130c379aa8c…`（Train）/ `042c32ffa7aed06f…`（Valid），全量见 `analysis/ecfp_manifest.json` |
| 行身份 digest | 每 split 的 `row_digest_sha256` 见 `analysis/row_identity.json` / `analysis/resource_ledger.json` |

### 7.5 复现命令

```bash
# 0) 代码版本（正式 GPU 训练 / 本报告分析）
git checkout f2f2d1b0292f77cbbd9b3e3d6fa9b0b7d1af08b1

# 1) CPU 全链路（P0.1 / P0.2 / 行身份 / P0.3 / P1 / 汇总），约 110 s
bash tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/code/run_cpu_steps.sh

# 1a) 其中 P1 使用隔离 CPU 环境（一次性创建）
uv venv --python 3.12 /home/calendar/.cache/research-venvs/peptides-ecfp
uv pip install --python /home/calendar/.cache/research-venvs/peptides-ecfp/bin/python \
    scikit-fingerprints==2.1.0 lightgbm==4.7.0

# 1b) 单独运行 P0.3（冻结 checkpoint，Train+Validation only）
uv run python tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/code/diag_p0_checkpoint_eval.py \
    --seeds 0 1 2 --device cpu \
    --out-dir tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/runs/p0_frozen_eval

# 2) P2：单 GPU 延长训练（1× A100，约 39 min）
rr doctor res && rr deploy res
rr run res lcrc-pepdiag-p2-seed0 --pool res-gpu0 --gpus 1 --cpus 4 --mem 32G --time 01:15:00 \
    --result tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/runs/peptide_c1_seed0_240ep \
    -- uv run python tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/code/diag_p2_train.py \
        --device cuda --max-seconds 4200 \
        --out-dir tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/runs/peptide_c1_seed0_240ep
rr status lcrc-pepdiag-p2-seed0 && rr pull lcrc-pepdiag-p2-seed0

# 3) 正确性测试
uv run pytest -q tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/tests
```

`rr` 记录（`analysis/p2_provenance.json`）：run id `lcrc-pepdiag-p2-seed0-20261104-090409-586b7651`，
pool `res-gpu0`，node `a100-2`，commit `f2f2d1b0292f`，exit code 0。

---

## 8. Validation 对比表（现有 LCRC / 固定 ECFP 基线 / 长训 LCRC）

全部数值都是 **Validation**，不涉及 Test。ECFP 行训练只用 Train；LCRC 行按 epoch 选
Validation 最佳 checkpoint。

| 方法 | seed | 训练 epoch（配置） | cosine horizon | 最佳 epoch | Validation macro AP % |
|---|---|---|---|---|---|
| LCRC-C1（Fast v1） | 0 | 120 | 120 | 116 | 53.73 |
| LCRC-C1（Fast v1） | 1 | 120 | 120 | 119 | 53.94 |
| LCRC-C1（Fast v1） | 2 | 120 | 120 | 113 | 54.79 |
| LCRC-C1（Fast v1）3-seed 均值 ± 样本 std | 0,1,2 | 120 | 120 | — | 54.15 ± 0.56 |
| Count ECFP4 + LightGBM（Train-only） | 0 | 500 trees / 标签 | — | — | **72.98** |
| 常数得分基线（= Validation 阳性率均值） | — | — | — | — | 16.48 |
| LCRC-C1 长训（前 120 epoch 段） | 0 | 240 | 240 | 120 | 57.72 |
| LCRC-C1 长训（121–240 epoch 段） | 0 | 240 | 240 | 231 | 60.62 |
| LCRC-C1 长训（全程最佳） | 0 | 240 | 240 | 231 | **60.62** |

逐标签 Validation AP 对比：

| label | Valid pos %（基线） | Fast v1 seed0 % | Fast v1 seed1 % | Fast v1 seed2 % | ECFP4+LGBM % | 长训 LCRC seed0 % |
|---|---|---|---|---|---|---|
| antifungal | 8.88 | 34.30 | 34.00 | 33.45 | 50.30 | 36.97 |
| cell_cell_communication | 3.39 | 40.73 | 42.48 | 40.64 | 73.51 | 50.63 |
| anticancer | 6.44 | 37.91 | 39.18 | 39.75 | 66.17 | 46.97 |
| drug_delivery_vehicle | 6.26 | 38.72 | 36.67 | 41.18 | 69.23 | 47.68 |
| antimicrobial | 62.81 | 91.36 | 91.39 | 91.81 | 97.23 | 93.57 |
| antiviral | 19.78 | 65.11 | 63.11 | 66.86 | 86.71 | 74.11 |
| antihypertensive | 10.73 | 88.86 | 89.20 | 89.36 | 92.04 | 90.09 |
| antibacterial | 18.58 | 48.28 | 50.57 | 49.51 | 68.59 | 52.94 |
| antiparasitic | 1.84 | 15.17 | 16.30 | 18.09 | 35.32 | 32.09 |
| toxic | 26.13 | 76.91 | 76.51 | 77.33 | 90.66 | 81.23 |

机器可读版本：`results/validation_comparison.csv`、`results/label_ap_comparison.csv`、
`analysis/report_tables.md`。

---

## 9. 正确性门禁与测试

`analysis/gates.json`（全部为真，除标注计数器）：

| 门禁 | 结果 |
|---|---|
| 历史 curve / checkpoint SHA-256 与历史 run.json 一致 | 通过 |
| 历史 best epoch 由曲线 argmax 复现 | 通过（3/3） |
| split 尺寸 10873 / 2331 / 2331、互斥且完整 | 通过 |
| 标签二值、Train/Valid 标签逐行等于 Fast v1 npy | 通过 |
| P0.2 是否读取 Test 标签 | `false`（未读） |
| 行身份 PASS / 原子特征不一致 / 键特征不一致 / 标签不一致 / canonical SMILES 不一致 | PASS / 0 / 0 / 0 / 0 |
| 冻结 checkpoint 全部可用、重算 macro AP 与历史一致、helper 重算差 0 | 通过 |
| P0.3 读取的 Test 产物数 | 0 |
| ECFP 特征形状 10873×2048 / 2331×2048、purity 检查 | 通过 |
| ECFP macro AP 独立重算差 | 0.0 |
| ECFP 读取的 Test 行数 | 0 |
| P2 参数量 / 初始化 hash / spec / norm 与 Fast v1 seed0 一致 | 通过 |
| P2 `test_evaluated` / `guard.attempts` | `false` / `[]` |
| P2 checkpoint 重载复现 Validation macro AP 差 | 5.4e-5 |

单元/一致性测试：`uv run pytest -q tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/tests`
→ **8 passed**（包括从保存的预测矩阵独立重算逐标签 AP、从曲线重算分阶段最佳值、
P2 源码不引用任何 Test 产物的静态检查）。

---

## 10. 产物索引

| 路径 | 内容 |
|---|---|
| `PROTOCOL.md` | 冻结执行协议 |
| `configs/protocol.json` | 机器可读配置（数据源、P1/P2 配方、budget） |
| `history/peptide_seed{0,1,2}/` | 历史曲线、run.json、selection.json、冻结 checkpoint（只读副本） |
| `analysis/learning_curves.csv` / `learning_curve_full.csv` / `learning_curve_summary.csv` / `learning_curves.png` | P0.1 |
| `analysis/history_runs.csv` / `learning_curve_audit.json` | P0.1 元数据与校验 |
| `analysis/label_statistics.csv` / `label_statistics.json` | P0.2 |
| `analysis/row_identity.json` | P1 行身份门禁（逐原子/逐键） |
| `analysis/ecfp_manifest.json` | P1 版本、行索引哈希、特征形状、耗时、purity |
| `results/ecfp_validation.csv` / `ecfp_per_label.csv` / `ecfp_gain_importance.csv` / `ecfp_valid_predictions.npz` | P1 结果 |
| `results/p0_frozen_checkpoints_summary.csv` / `_per_label.csv` / `analysis/p0_frozen_checkpoints_audit.json` | P0.3 结果 |
| `runs/p0_frozen_eval/seed{0,1,2}/` | P0.3 每 seed 的 eval/structural 记录与 Train/Valid 概率 |
| `runs/peptide_c1_seed0_240ep/` | P2：curve.csv（240 epoch）、checkpoint、run.json、selection.json、Train/Valid 概率 |
| `analysis/p2_provenance.json` | P2 rr provenance 与被拉取产物的哈希 |
| `analysis/gates.json` | 全部门禁结果 |
| `analysis/resource_ledger.json` / `cpu_steps.json` | 资源、版本、哈希 |
| `results/validation_comparison.csv` / `label_ap_comparison.csv` | 对比表机器可读版 |
| `analysis/report_tables.md` | 本报告表格的生成版本 |
| `code/` | P0/P1/P2/汇总脚本 + `run_cpu_steps.sh` |
| `tests/test_diag_core.py` | 8 个一致性测试 |
