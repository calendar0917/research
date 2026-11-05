# PROTOCOL — LCRC PCBA 学习率平台诊断（`lcrc-pcba-lr-probe-v1`）

Round: `lcrc-pcba-lr-probe-v1`
类型：**优化诊断**，不是新架构实验，也不是完整基准复现
数据集：`ogbg-molpcba`（官方 `GraphPropPredDataset` + 官方 `get_idx_split()`）
模型：既有 **LCRC-C1 / NO_BOND_MEAN**，245,184 参数，128 任务输出
上游（只读）：`tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1`
数据传输：复用 `tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.py` 的 `PinnedStager`

本文件在**第一次正式训练之前**固定实验定义。除“实现映射”和“执行偏差”两节外，
不新增研究内容。

---

## 1. 研究问题

已有 LCRC-C1 在 `ogbg-molpcba` 上于 epoch 29 达到最佳 Validation AP
`0.23449277039313338`（报告写作 `0.23449`），后续 10 epoch 无新高，学习率仍接近
`3e-4`（历史 per-step cosine 只走到 360-epoch 周期的约 11%）。

本轮只检验：

> 从相同的最佳模型权重出发，使用较低学习率是否比继续使用较高学习率带来更好的
> **短期** Validation AP？

这是优化诊断；不声称统计显著性，不声称模型能力上限，不做超参数搜索。

---

## 2. 复用资产（只读，不修改、不覆盖）

| 资产 | 路径 | 冻结身份 |
|---|---|---|
| 官方特征缓存 | `lcrc_pcba_scale_pilot_v1/cache/pcba_v1` | manifest sha `caad3e90…`，4.74 GB，23 文件 |
| 官方 split | `analysis/pcba_split.json` | partition sha `c872bd63…`；350343/43793/43793 |
| 标签工件 | `analysis/pcba_labels_trainvalid.npy` | sha `5c3b5fbe…`；Test 行全 NaN |
| 特征规范 | `analysis/pcba_spec.json` | hash `7270f767…` |
| Train-only normalization | 由官方 train 行重新拟合 | fingerprint `6ce50f98…` |
| 历史最佳权重 | `runs/pcba_seed0/checkpoint.pt` | sha `b2affec7…`，epoch 29，AP 0.23449277 |
| 数据传输 | `lcrc_pcba_pipeline_perf_v1/code/pf_pipeline.py` | 已通过 batch/label 等价与一步更新门槛 |

历史 `epoch29` checkpoint、`runs/pcba_seed0/` 与 pilot 的任何文件都不会被覆盖。
本轮所有输出写入 `tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/`。

---

## 3. 实验矩阵

严格运行两组；两组只有固定学习率不同。

| 组 | 角色 | 初始权重 | AdamW LR | epochs | 计划 GPU |
|---|---|---|---|---|---|
| **H** | High LR 对照 | 历史 epoch29 checkpoint | `3e-4` | 12 | `res-gpu0` |
| **L** | Low LR 实验 | **完全相同**的历史 checkpoint | `5e-5` | 12 | `res-gpu1` |

共同配置：

| 项 | 值 |
|---|---|
| 模型 | LCRC-C1，128 输出，宽度/层结构与历史 `MODEL_CONFIG` 完全一致 |
| 数据划分 | 官方相同 Train/Validation split；Test 不可访问 |
| batch size | 128 |
| 优化器 | AdamW，weight decay `1e-3`，betas `(0.9, 0.999)`，eps `1e-8` |
| dropout | `0.10`（继承历史模型定义） |
| gradient clipping | `5.0` |
| 学习率调度 | **无 scheduler**；两组固定 LR |
| 精度 | FP32；不启用 autocast / mixed precision |
| 随机种子 | `0`；两组显式重置到同一训练起始 RNG |
| minibatch 顺序 | 同一 CPU generator：`manual_seed(0)` + 每 epoch `randperm`，共 12 epoch |
| 数据传输 | `PinnedStager(device, depth=2)`；不进一步优化 collate |
| 训练轮数 | 12 个完整 epoch；**不使用 early stopping** |
| 超参数搜索 | 无 |

**AdamW 重新初始化（明确限制）**：历史 checkpoint 只含模型 `state_dict`，
不含 AdamW moments、step count 或 scheduler 状态。两组都从零 AdamW 状态开始。
因此本轮不能宣称“无缝续训”，只能宣称“从相同权重出发的新优化器探针”。
该限制写入每个 checkpoint、`run.json`、`initialization_audit.json` 和 `REPORT.md`。

---

## 4. 训练前正确性门槛

正式任务提交前必须全部通过；任一失败即停止正式训练，不得通过修改模型、数据或指标绕过。

1. **初始权重相同**：H/L 加载后的参数 fingerprint 完全相同，且逐 tensor bitwise
   等于历史 checkpoint 的 `state_dict`。
2. **配置匹配**：模型 `MODEL_CONFIG`、128 输出、feature-spec hash、
   Train-only norm fingerprint 与历史 checkpoint/run 记录一致。
3. **epoch0 AP 重现**：加载后、任何更新前，两个 GPU 上的官方 Validation AP
   都重现 `0.23449`，容差 `1e-4`。
4. **训练顺序一致**：两组每个 epoch 的训练行号顺序完全相同；保存每 epoch 的
   row-order hash 与 batch-plan hash。
5. **`PinnedStager` 真实接入**：batch/label 等价、loss/梯度/一次 AdamW 更新 smoke
   通过；不是只测空转。
6. **Test 隔离**：官方 Test 行在标签工件中全 NaN；训练代码只索引 train/valid；
   Test 不可访问、不可评价。
7. **数据契约**：gids 为官方行序、128 任务、NaN mask 语义、标签二值性、
   Train/Validation/Test 行不重叠、Train-only normalization 与历史一致。

预检在两块 GPU 上分别运行；本地 `lr_audit.py` 汇总两份记录。只有
`initialization_audit.json.passed == true` 且 `paired_training_order.json.passed == true`
才允许正式训练。

---

## 5. 正式训练与评价

- 本地实现、测试、提交代码后，`rr deploy res` 部署同一 commit。
- 使用 `rr run res ... --pool ...` 启动两组。
- 正式训练固定 12 个完整 epoch，不 early stop。
- 在 epoch `0、3、6、9、12` 计算官方 OGB Validation AP：
  - epoch 0 = 加载历史权重后、更新前；
  - epoch 12 = 第 12 个完整 epoch 结束后。
- 每个 epoch 记录：Train loss、实际 LR、step/epoch 耗时、GPU 峰值显存；
  eval epoch 额外记录 Validation AP、逐任务 AP、有效标签/正例/负例数。
- 记录 NaN、梯度异常、数据访问异常；任何 non-finite 事件使该组标记为
  `INCOMPLETE`。
- 保存：
  - 每组最终 epoch12 checkpoint；
  - 每组最佳 Validation checkpoint（严格提升才更新；平分保留更早 eval epoch）；
  - 完整随机状态（Python/NumPy/CPU torch/每 GPU CUDA）、优化器状态、
    checkpoint SHA、初始化方式。
- 不读取、不评价官方 Test。

---

## 6. 资源与停止条件

| 项 | 值 |
|---|---|
| 总预算 | **≤ 1.5 A100 GPU-hours**，包含两块 GPU 的 preflight/smoke 与正式训练 |
| 每组 soft wall limit | 2050 s |
| 每组 hard wall limit | 2200 s（进程内 watchdog，优雅保存后退出） |
| 每组 outer kill | 2300 s（最后手段） |
| 预算记账 | 每组 `gpu_seconds / 3600` + 两份 preflight 的 `gpu_seconds / 3600` |

若任一组无法完成预定 12 epochs，标记 `INCOMPLETE`；不把未完成结果与完整组
直接作为主实验比较。禁止延长训练、增加 seed、增加 LR 候选、改 batch、
使用混合精度、继续优化 collate 或尝试其他模型。

---

## 7. 固定评价与决策标签

主比较：

```
D_LR = AP_L(epoch12) - AP_H(epoch12)
D_old = AP_L(epoch12) - 0.23449
```

判定（逐字执行）：

| 标签 | 条件 |
|---|---|
| `LR_SIGNAL` | `D_LR >= 0.005` 且 `D_old >= 0.005` |
| `TRAINING_SIGNAL_NOT_LR_SPECIFIC` | 两组 epoch12 均比旧最佳提高至少 `0.005`，但 `D_LR < 0.005` |
| `NO_CLEAR_INCREMENT` | 其他达到门槛的完整运行结果 |
| `INCOMPLETE` | 资源、环境或正确性原因导致无法完成配对实验 |

无论标签如何，报告两组完整曲线、最佳/最终 Validation AP、逐任务差异及其局限。
这些只是资源决策标签，不代表统计显著性，不代表模型能力上限。

---

## 8. 交付物

新目录 `tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/` 至少包括：

- `PROTOCOL.md`
- `code/` 实现
- `tests/` 针对性正确性测试
- `analysis/initialization_audit.json`
- `analysis/paired_training_order.json`
- `analysis/comparison.json`
- `runs/H_high_lr/`、`runs/L_low_lr/` 的训练曲线、checkpoint、`run.json`
- `REPORT.md`

---

## 9. 实现映射（不是新的科学定义）

| 指令项 | 本目录实现 |
|---|---|
| 模型 | `lr_model.build_from_checkpoint` → 历史 `bp_model.build_c1(cont_slice=None)`；断言 `MODEL_CONFIG` 未漂移、无 `b_mean`、245,184 参数、dropout 0.10、FP32 |
| 数据 | `lr_data.load_task` → pilot `pc_data.load_cache` + 官方 split + 标签工件；拒绝任何有限 Test 标签 |
| Train-only norm | `bp_data.fit_norm_rows(cache, spec, train_rows)`；断言 fingerprint 等于历史 |
| 128 任务 | `N_TASKS = 128`；输出/loss/AP 都是 `[N,128]` |
| masked loss | `pc_metrics.masked_bce_loss`（`torch.isfinite` mask；NaN 不填 0） |
| 官方 AP | `pc_metrics.validation_report` → `Evaluator("ogbg-molpcba")`；独立逐任务 AP 同时保存 |
| 顺序 | `lr_data.make_order_plan`：与 `pc_train._batches` 完全相同的 CPU generator 序列 |
| 数据搬运 | `pf_pipeline.PinnedStager`；训练 step 使用 `stager.batch` + `stager.labels` |
| 固定 LR | `lr_model.make_optimizer`：组 LR；**不创建 scheduler** |
| 选择 | `Tracker`：严格提升才更新，平分保留更早 eval epoch |
| 资源保护 | `Watchdog` + epoch 间 soft check；逐 epoch 落盘；`checkpoint_best.pt` 在训练中原子保存 |
| Test 隔离 | `load_task` 断言 Test 行全 NaN；训练只索引 `train_rows`/`valid_rows` |
| 预检 | `lr_preflight.py` + 本地 `lr_audit.py` |
| 比较 | `lr_compare.py`；逐字执行第 7 节规则 |

---

## 10. 执行偏差与局限（预先声明）

- 计划 H 使用 `res-gpu0`、L 使用 `res-gpu1`。如果 `res-gpu0` 被外部共租任务占用，
  按研究者授权，两组可**顺序**在 `res-gpu1` 上运行；实际 `CUDA_VISIBLE_DEVICES`、
  GPU 型号、GPU-hours 和耗时全部记录。设备分配变化是执行偏差，不改变数据、模型、
  LR、batch 顺序或评价规则。
- 只运行一个 seed、12 个 epoch；结果不能外推为长周期或多 seed 结论。
- AdamW 重新初始化；不能与历史 optimizer 轨迹做无缝比较。
- 不使用 scheduler；两组 LR 固定。
- 官方 Test 从未读取；报告中的 AP 不可与论文 Test AP 直接比较。
- 资源决策标签不是统计显著性检验。

---

## 11. 执行后记录（post-run execution record）

- 正式 H/L 均从 commit `9b5a61a9fead1c2276c864f073e571fc454e68d4` 部署并运行：
  H `lr-probe-H12-20261105-193541-317733c4`（`res-gpu0`，CUDA_VISIBLE_DEVICES=0），
  L `lr-probe-L12-20261105-193608-6506e97b`（`res-gpu1`，CUDA_VISIBLE_DEVICES=1）。
  两组各完成 12/12 epoch，状态 `COMPLETED`。
- `res-gpu0` 在正式运行期间确有外部共租进程（用户 `wujiayu`，PID 3841932，约 16 GB），
  但仍有约 24 GB 空闲；预检和正式训练均正常完成。因此**没有启用第 10 节的
  “两组顺序在 res-gpu1 运行”回退**，而是保留主分配 H=GPU0、L=GPU1。
  这是执行选择偏差：H 的 step p50 为 43.30 ms，L 为 33.04 ms；H 的吞吐较低，
  但 AP 评价规则、数据、模型、LR、batch 顺序均未改变。
- 第一次 preflight（`lr-probe-preflight-H-20261105-192223-9e2bf4d7` /
  `lr-probe-preflight-L-20261105-192253-da8d74a1`）因 NaN 标签比较与 dropout RNG
  重置问题失败；修正后重跑，只有第二轮 preflight 通过后才启动正式训练。
  被拒绝的 run 不进入主比较。
- 正式训练后新增本地审计：四个 checkpoint 的 CPU AP 重算、主比较、资源记账与
  逐任务差异；结果见 `analysis/checkpoint_audit.json`、`analysis/comparison.json`。
- 官方 Test 在所有正式/预检/审计路径中均未读取或评价。
