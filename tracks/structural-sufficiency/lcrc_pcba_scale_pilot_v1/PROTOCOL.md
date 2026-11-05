# PROTOCOL — LCRC PCBA 规模试运行（`SCALE_PILOT`）

Round: `lcrc-pcba-scale-pilot-v1`
结果标签：`SCALE_PILOT`
说明：本文件逐项照录执行指令给定的实验定义。除"实现映射"一节外不新增研究内容。

---

## 1. 科学定义

* 数据集：`ogbg-molpcba`。
* 使用 OGB 官方 `GraphPropPredDataset`、官方 `get_idx_split()`。
* 预期数据规模：437,929 个分子、128 个二分类任务。
* 模型：**现有 LCRC-C1 / NO_BOND_MEAN**，即保留显式 S1/S2 和 global readout，仅删除原模型中的 `b_mean` 块。
* 禁止更换架构、添加 attention、预训练、额外分子描述符或任何新模型模块。
* **不得直接把已有 `bp_mol.py` 当作已经支持 PCBA。** 原代码只覆盖 MolHIV、MolBACE、MolTox21，并存在单任务/12 任务假设；必须在实验目录中正确适配 128 任务。

## 2. 模型与训练配置

沿用历史 `bp_common.py` 的 LCRC-C1 配置和 OGB 训练配方：

| 项 | 值 |
|---|---|
| seed | `0`，且仅此一个 seed |
| 模型宽度 | 复用已有 `MODEL_CONFIG`（width 64 / stat_rank 32 / local_hidden 192 / atom_hidden 128 / atom_out 96 / bond_hidden 160 / bond_out 96 / rho_hidden 64） |
| 优化器 | AdamW |
| peak learning rate | `3e-4` |
| weight decay | `1e-3` |
| dropout | `0.10` |
| batch size | `128` |
| gradient clipping | `5.0` |
| 学习率调度 | 保留历史 per-step cosine 配方及其定义（`steps_per_epoch × 360` 更新，`eta_min = 1e-5`） |
| 其他训练技巧 | 不使用 class reweighting、focal loss、数据增强或其他新训练技巧 |
| 输出 | 128 维 logits |
| 损失 | 对实际观测标签执行 masked `BCEWithLogitsLoss`；`NaN` 严格从损失中排除，不改为 0 |
| 选择指标 | OGB 官方 `Evaluator("ogbg-molpcba")` 的 Validation `ap` |
| checkpoint | 保存 Validation AP 最高的 checkpoint；平分时选更早 epoch |

探索性训练资源约束：

* `max_epochs = 40`
* `min_epochs = 12`
* `patience = 10`（以 Validation AP 严格改善为依据）
* GPU0 总预算不超过 **4 A100 GPU-hours**，包括训练 smoke。
* 设定外层远端超时保护，确保优雅保存和记录先于进程被强制终止。

若资源预算先于预定 epoch 耗尽，保存现有最优 checkpoint、训练曲线和资源记录，标记
`RESOURCE_LIMITED`；**不得自行加时或续跑**。

本轮是 `SCALE_PILOT`，训练轮数不足的结果不得解释为模型能力上限。

## 3. 数据与正确性检查（正式训练前必须完成）

1. 核验数据量、128 个标签列及官方 train/valid/test 划分。
2. 核验每个任务的已观测、正例、负例及缺失标签数量。
3. 核验结构缓存不包含目标标签。
4. 使用既有 LCRC 结构特征定义，不能截断、采样或近似 S2。
5. 验证 S2 小图计数与历史实现一致。
6. 验证 C1 特征形状、输出 `[batch,128]`、loss finite、梯度 finite、checkpoint 重载一致。
7. 检查单图/批量推理一致性和原子重编号不变性。
8. 用人工构造含 `NaN` 的小型标签矩阵验证 masked loss。
9. 验证官方 OGB AP 计算可以忽略缺失标签，独立重算与官方评估器一致。
10. 核验缓存、split、checkpoint 的哈希及可复现信息。

数据处理可以使用分块或流式缓存以避免一次性存储 43 万个 Python 分子对象，但**不得改变数学特征定义或样本集合**。

若官方 SMILES 转换失败，记录失败的行号、原因和数量；不得静默丢弃、替换或改变标签与图的行号对应关系。

## 4. 执行顺序

1. CPU 数据预检查及结构缓存可行性检查。
2. 使用官方 Train 内固定索引子集做最小模型 smoke（仅测试正确性、吞吐量、显存和数值稳定性；不通过小样本 AP 选择超参数）。
3. 基于实际的预处理、单 epoch 吞吐及显存实测，检查完整训练能否在预算内推进；不能保证完成时也启动有上限、可保存 checkpoint 的单 seed 探索性训练，而不是擅自改变模型。
4. 使用 `rr run res ... --pool res-gpu0` 启动正式 PCBA seed0。
5. 训练期间保存每个 epoch 的：Train loss、Validation mean AP、128 个任务的有效标签/有效 AP 数、Learning rate、epoch 耗时/累计耗时、GPU 显存峰值、已执行 epoch 与最优 epoch。

**本轮禁止访问或评估官方 Test 标签、禁止 Test 指标、禁止依据 Test 调整模型。**

## 5. 交付物

`PROTOCOL.md`、`analysis/data_audit.json`、`analysis/feature_cache_audit.json`、
`analysis/smoke.json`、`runs/.../train_curve.csv`、`runs/.../checkpoint.pt`（最优
Validation）、`runs/.../run.json`、`REPORT.md`。

`REPORT.md` 必须写明：是否完整训练/是否资源受限；实际总 GPU-hours、峰值显存、数据
预处理耗时；最优 Validation AP 与 epoch；Train loss、Validation AP 趋势；可评估任务
数和任务标签覆盖率；是否存在训练发散、性能平台期或明显欠训练；与历史 LCRC
benchmark 的实现继承关系、具体修改及所有正确性检查结果。若未完成可比训练，不得将
AP 与论文 Test AP 直接比较。

## 6. 实现映射（不是新的科学定义）

| 指令项 | 本目录实现 |
|---|---|
| 模型 | `pc_model.build_c1` → 历史 `bp_model.build_c1`（`cont_slice=None`），并在 `pc_common.assert_frozen_inheritance` 中断言 `MODEL_CONFIG` 与 `bp_common.OGB_RECIPE` 未漂移 |
| 数据 | `pc_data.load_official`（`GraphPropPredDataset` + `get_idx_split()` + 官方 `mapping/mol.csv.gz`） |
| 128 任务 | `n_out=128` 由 `pc_data` 的加载断言保证；损失/评估都是 `[N,128]` |
| 特征 | `pc_data.build_shards`：每个分子走历史链路 `mb2_rdkit.smiles_to_raw_row_relaxed` → `raw_row_to_graph` → `lcrc_data.build_molecule`；分块并行只为省内存，不改变定义或样本集合 |
| masked loss | `pc_metrics.masked_bce_loss`（`torch.isfinite` 掩码） |
| 官方 AP | `pc_metrics.official_ap`（`Evaluator("ogbg-molpcba")`），独立重算 `pc_metrics.per_task_ap` |
| per-step cosine | `pc_train`：`CosineAnnealingLR(T_max = steps_per_epoch × 360, eta_min = 1e-5)` |
| 选择 | `pc_train.Tracker`：严格提升才更新，平分保留更早 epoch |
| 超时保护 | `pc_train.Watchdog`（进程内硬限，命中后优雅收尾）+ `--soft-seconds`（epoch 间检查）+ 外层 `timeout` 作为最后手段 |
| Test 隔离 | `pc_data.build_label_artifact` 把官方 Test 行写成 NaN；`pc_train.load_task` 断言 Test 行全为 NaN；训练代码只索引 train/valid |
| 资源记录 | `runs/pcba_seed0/run.json` 的 `gpu_seconds`/`gpu_hours`/`gpu_peak_*` |
