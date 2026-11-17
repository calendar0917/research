# PROTOCOL — lcrc-readout-context-pilot-small-v1

本文件是本轮的**规则声明**（跑之前冻结）。机器可读版本在
`protocols/lcrc-readout-context-pilot-small-v1.yaml`（控制面协议）与
`configs/experiment.yaml`（工程配置）；跑完之后冻结的全部常量与 hash 在
`analysis/protocol_freeze.json`（含 `freeze_hash`）。

变化规则：本文件与 freeze 一经正式运行即不得为了结果好看而修改。工程性修复只允许
修实现（索引/dtype/路径/账目字段），并写入 `analysis/engineering_log.md`。

---

## 1. 问题

在官方 `ogbg-molesol`（1128 行）与 `ogbg-mollipo`（4200 行）上，**保持现有 C1/LCRC 局部编码不变**，
把原读出换成「局部基础贡献 + 单项全局统计」的加性读出（B），再加两项
「局部结构 × 全局上下文」的显式交互（C），在**同一个读出参数预算**下回答四件事：

1. C 相对 A（原 C1 读出）保住多少性能？
2. C 相对 B 的上下文增量是否存在、是否稳定？
3. 交互是否真的被使用，还是只有常数/规模效应？
4. 每个预测能否从导出的账目逐项重建（TRACE）？与强参照 GINE 还差多少？

不在本轮范围：文献调研、架构/LR/epoch 搜索、新损失、蒸馏、SSL、额外标签、MMP 挖掘、分子生成。

## 2. 数据与划分

* 只用官方 OGB `PygGraphPropPredDataset` + `get_idx_split()` 产生的 Train/Val/Test（scaffold split）。
* 四个冻结缓存按**真实文件字节** SHA-256 在每个阶段、每台机器校验：
  `ogbg_{molesol,mollipo}_sq-pilot-cache-v1.npz` 与 `..._sq-pilot-lipo-raw-v1.npz`。
* 划分尺寸固定：molesol 902/113/113，mollipo 3360/420/420；split 互斥且覆盖全表（prepare 阶段断言）。
* 不删行、不合并 Train/Val、不做自造 80/10/10。

## 3. 允许的只读复用

| 用途 | 路径 |
| --- | --- |
| 图缓存/载入/C1 计算图/GINE | `tracks/structural-query-learning/{code,data,results/sq-pilot-small-v1}` |
| raw→特征、norm、`DissectLCRC` | `tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code` |
| C1 对照模块 | `tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code` |

这些目录本轮**只读**，不作任何修改；它们的文件 hash 进入 `protocol_freeze.json`。

## 4. 模型

三臂共用同一个局部 trunk（逐位等于 C1 的局部通路；同一份规范初值）：

* **A**：C1 原读出，`sum_v f(h_v) + rho([z_g, mean h, mean h^2, log1p n])`，**不改**；
* **B**：`b_global + sum_v c0(h_v) + sum_j u_j(z_j)`；
* **C**：B + `a_size(z)·sum_v c_size(h_v)` + `a_ring(z)·sum_v c_ring(h_v)`。

细节：

* `M` = Train 上跨度 > `1e-8` 的原始 global 列数（molesol 34 / mollipo 36），常量列在 B/C 中不建随机头。
* 每个单变量族是 `q(z) = w2·[SiLU(w1 z + b1) − SiLU(b1)]`（48 参数，无输出 bias，`q(0)=0`）。
* 上下文只有两个**预先命名**字段：`log1p_n_atoms`、`ring_atom_fraction`。
* 读出参数预算 = A 的 15298；`H` 取使名义读出参数量落在**一个宽度步长**内的整数
  （molesol：B 207/15296、C 200/15330；mollipo：B 206/15326、C 198/15290）。
* 初始化：规范初值子种子 0；新读出用三条独立子流（global 1001 / context 1002 / local 1003）。
  这些是**一个模型配置内的模块初始化**，不是额外训练 seed。

## 5. 训练与每 epoch 评估（一套配方，三臂相同）

* seed 0；360 epoch 跑满，无 early stopping；
* AdamW：peak 3e-4 → min 1e-5，betas (0.9,0.999)，eps 1e-8，wd 1e-3；
* 逐步 cosine：`lr(t) = min + 0.5(peak−min)(1+cos(π t/T))`，`t` 从 0 起，`T = 360·ceil(N_train/128)`；
* MSE（标准化目标），全局梯度范数 clip 5，FP32，无 AMP/EMA/SWA/soup，无数据增广/重采样/蒸馏；
* batch 顺序：每条轨迹 `np.random.default_rng(0)` 逐 epoch `permutation` Train split-local 行，
  **同任务三臂的 batch-order hash 必须一致**（实测一致）；
* 每 epoch 全量评估 Train/Val/Test 三划分，原始单位 RMSE/MAE，`test_policy=always`；
* 选择：**最小 Val RMSE**，严格更小才替换，相同保留更早 epoch；epoch 0 只作初始化诊断，不参与选择。

## 6. Test 的角色

Test 每 epoch 全量评估（用户明确覆盖 terminal-only 默认）。它**从不**参与 checkpoint、宽度、
LR、上下文选择或任何判读量；报告中 Test 一律标 `DEVELOPMENT_TEST`。

## 7. 回溯（TRACE）与容差

* 容差 `tol_raw(x) = 1e-5 + 1e-6·max(1,|x|)`，原始单位；失败不得事后放宽。
* 账目：`intercept_raw = mu + sigma·b_global`；`base_root_raw = sigma·c0(h_v)`；
  `global_j_raw = sigma·u_j(z_j)`；`interaction_{v,k}_raw = sigma·a_k(z_k)·c_k(h_v)`；总和 = `pred_raw`。
  **只有 intercept 带 mu。**
* 导出范围：B/C 全量 Val/Test 的分子级与 root 级账目；Train 只用于 `c̄_k` 参考均值与汇总。
* 诊断 1：全量预测重建（max/median 误差、失败 row_id）。
* 诊断 2：固定局部输入干预（每任务最多 32 个 Val 分子，删第 0 条 typed-S2 occurrence；
  不改 raw graph / S1 / scalars / global；变化必须由被影响 root 的 base + 交互重建；A 的 rho 响应照实记录）。
* 诊断 3：固定 `h` 的上下文条件响应（观测值 → Train Q75；闭式 `Δ = Δu_k + Δa_k·S_k` 与整前向比较；
  主效应与交互效应分列）。
* 诊断 4：交互退化（Train 根等权均值 `c̄_k` 精确分解；常数根项 vs 结构偏离项；根间 std/RMS 中位数）。
  读数：Val RMS ≥0.01 且干预 RMS ≥0.01 → `RESOLVED_FUNCTIONAL_RESPONSE`；根间中位数 <0.05 →
  `LOW_ROOT_SPECIFICITY_AT_5PCT`，否则 `NONUNIFORM_ROOT_RESPONSE`。0.01 只是 1% 目标标准差的读数分辨率。
* 诊断 5：等价尺度（`a_k×2 / c_k÷2` 预测与每根交互不变；因此参数范数不是重要性指标）。
* checkpoint 重载后预测与全部账目必须重放通过。

## 8. 统计

* 配对分子 bootstrap：2000 次、RNG 1703、同一 split 内所有模型共用同一次重采样、每次重算 RMSE/MAE，
  报差值的 2.5/97.5 分位。
* 条件于「当前单 seed 选中状态」；不估计 seed 方差，不修复 Val 选择，不消除 Test 开发暴露。
* 逐行 drop-one 敏感性：报 Δ 范围与翻号行数；不得用删除后的数字替换官方分数。

## 9. 四个判读轴（分别输出，不合并成一个 PASS）

| 轴 | 规则 |
| --- | --- |
| A 回溯 | 预测重建 + 两类干预 + 尺度/批量/重编号等价 + checkpoint 重放全部通过 = `TRACE_PASS`；否则 `TRACE_FAIL` |
| B 性能保留 | C/A 的 Val 与 Test RMSE 比值都 ≤1.05 = `PRESERVED_POINT_ESTIMATE`；last20 两比值也 ≤1.05 追加 `HORIZON_CONSISTENT`，否则 `CHECKPOINT_SENSITIVE`；两 split 分居阈值两侧 = `SPLIT_SENSITIVE`；两者都 >1.05 = `PERFORMANCE_LOSS_THIS_RECIPE` |
| C 上下文增量 | C 相对 B 的 Val 与 Test 均改善 ≥2% 且 last20 两 split 不反向 = `CONTEXT_PERFORMANCE_POINT_SIGNAL`（区间含 0 追加 `UNCERTAIN_FIXED_STATE`）；只有一个 split 改善或 best/late 翻转 = `SPLIT_OR_SELECTION_SENSITIVE`；未达 2% = `NO_MATERIAL_CONTEXT_GAIN_THIS_RECIPE` |
| D 对 G 的竞争力 | C 的 Val 与 Test 都 ≤1.05·G = `WITHIN_5PCT_OF_THIS_GINE`，否则 `BEHIND_THIS_GINE` 并逐 split 列差距 |

5% 是**本轮实用性容差**的单 seed 开发判断，不是等价性检验（区间跨 0 不能写「等价」）。
若 C 保住 A 却仍落后 G，必须同时报告，不能只给「保留成功」的正面标题。

## 10. 资源与执行

* 正式 GPU 软预算 2 A100-h、硬预算 3 A100-h（含 smoke 与 GPU 侧解释），墙钟目标 4 h。
* 静态调度：`res-gpu0` = molesol worker（A→B→C），`res-gpu1` = mollipo worker（A→B→C）；
  只启动这两个 worker，每卡只有一个训练进程，逻辑设备 `cuda:0`。
* 正式轨迹数据置 6 条；仅当 G 复用门禁失败时允许新增对应 G（上限 8）。
* 全部远端执行经 `rr deploy / run / status / logs / pull`；不手写 ssh+nohup/sbatch。
* 记录 `CUDA_VISIBLE_DEVICES`、GPU 型号、driver、torch/CUDA、python、commit、uv.lock hash。
* 单臂/单任务独立输出目录 + 原子完成标记；resume 先核对配置/数据/代码/seed 签名再跳过。

## 11. 交付

`REPORT.md` / `DECISION.md`（中文）、`analysis/{protocol_freeze,dataset_audit,checks_*,run_manifest,
resource_ledger,engineering_log}.{json,md}`、逐臂 `results/<task>/<arm>/{trajectory.json,curve.csv,
best/final 预测,checkpoint}`、`analysis/ledgers/*`（分子级/root 级）、两类干预与两个诊断 JSON、
曲线图 + 2 个上下文函数图 + 每任务 3 个案例图。大型逐 epoch 预测矩阵与 `checkpoint_final.pt`
留在本轮结果目录（可复现、不入库），README 与 `run_manifest.json` 给出路径与 hash。
