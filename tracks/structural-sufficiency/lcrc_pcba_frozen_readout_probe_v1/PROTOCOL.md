# PROTOCOL — LCRC PCBA 冻结结构表征的快速读出能力探针（`lcrc-pcba-frozen-readout-probe-v1`）

Round: `lcrc-pcba-frozen-readout-probe-v1`
类型：一次性冻结特征提取 + 极小残差头的短程配对比较（诊断，不是架构实验，不是完整训练）
数据集：OGB `ogbg-molpcba`，官方 Train / Validation；**官方 Test 从未读取、从未评分**
冻结模型：LCRC-C1 / NO_BOND_MEAN，245,184 参数，128 任务输出
源 checkpoint：`tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/runs/L_low_lr/checkpoint_best.pt`
源 checkpoint SHA256：`b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5`

本文件在第一次 GPU 运行前冻结。运行期间不修改学习率、宽度、训练长度、损失、
评价点或判定阈值；如需修改，必须先停轮并显式重写本协议。

---

## 0. 冻结参考值

| 项 | 值 |
|---|---|
| 历史 Validation AP `AP0` | `0.2374902968831333`（epoch 9） |
| 冻结诊断 Train macro AP `TRAIN_AP0` | `0.38833559949014146`（`lcrc_pcba_loss_screen_v1` Phase A） |
| 可评分 Validation 任务数 | 127（任务 45 在 Validation 只有 41 个有效标签、0 个正例） |
| 复现容差 | Validation `1e-4`；Train `1e-4` |

---

## 1. 阶段 A：一次性冻结特征提取

模型加载后立即：

- `model.eval()`；
- 所有参数 `requires_grad_(False)`；
- 所有前向在 `torch.inference_mode()` 下执行；
- 不创建优化器，不调用 `backward`。

对官方 Train（350,343）与 Validation（43,793）分子**各编码一次**，提取：

1. 模型原始 128 维 logits（`return_aux=True` 的同一次前向）；
2. 原 `rho` 模块收到的完整输入 `rho_in`，用 `model.rho` 的 **forward pre-hook**
   原样捕获，不修改历史数学实现。

`rho_in` 布局（由 `tu_model.rho_layout` 与 `model.rho.lins[0].weight.shape[1]`
双向核验，运行期决定，不硬编码）：

```
[ g (42) | mean_h (64) | mean_h2 (64) | log1p(n_atoms) (1) ]  =>  d = 171
```

独立内容核验（每个 batch）：

- 从 batch 的 `global` 与 `aux["h"]` 独立重建 `rho_in`，与 hook 捕获张量比较；
- `model.rho(rho_in)` 重放结果与 `aux["rho"]` 比较；
- `aux["contrib"] + aux["rho"]` 与 `logits` 必须逐位相等；
- 另用历史 `pc_model.predict_logits` 路径独立重放 Validation logits。

缓存（`runs/feature_cache/`，git-ignored）：Train/Valid `rho_in`、Train/Valid
原始 logits、Train/Valid 标签（NaN 保留）、Train/Valid 有效掩码、标准化的
mean/scale、`manifest.json`（形状、dtype、SHA256、行顺序哈希、来源）。

标准化：**只用 Train** 计算每列 mean / population std（ddof=0，float64 累加）；
`std <= 1e-12` 的列 scale 置 1.0 并逐列记录。Validation 标签不参与拟合。

### 阶段 A 正确性门槛（任一失败 → `INCOMPLETE`，停止）

| 门槛 | 判据 |
|---|---|
| checkpoint 身份 | SHA256 等于冻结值，元数据审计通过 |
| 权重加载 | 与 checkpoint 逐 tensor bitwise 相等 |
| 参数冻结 | 全部 `requires_grad == False`，`model.training == False` |
| 提取后参数不变 | 每参数 SHA256 与指纹前后一致；无 `.grad` |
| Validation AP 复现 | `|AP_valid - AP0| <= 1e-4` |
| Train AP 复现 | `|AP_train - TRAIN_AP0| <= 1e-4` |
| `rho_in` 维度 | 等于 `model.rho.lins[0].weight.shape[1]` == 171 且布局宽度吻合 |
| `rho_in` 内容 | 独立重建最大绝对差 `<= 1e-5` |
| 原计算路径一致 | `contrib + rho == logits`（逐位）；独立重放 AP 差 `<= 1e-5` 且最大 logit 差 `<= 1e-3` |
| 行完整性 | Train 350,343 / Valid 43,793 唯一且互不相交 |
| Test 隔离 | Test 行有限标签数 0；Train/Valid 与 Test 无交集；从不索引 |
| 标准化 | zero-variance 列 scale 恰为 1.0；拟合行数 = 350,343 |

### 阈值修订（在阶段 A/B 任何正式 GPU 结果之前，2026-11-05T23:02Z）

smoke（`fr-preflight`，run `fr-preflight-20261105-225804-13e5bbf8`）暴露了一条
实现事实而不是数据问题：冻结前向里的 `mean_h` / `mean_h2` 用 CUDA
`index_add_`（atomic）归约，因此**同一输入两次前向的 logits 不可能逐位相等**。
随后用专门的诊断 run（`fr-replay-diag-20261105-230204-165891c2`，8,192 条
Validation 行，`runs/replay_diag/replay_diag.json`）量化：

| 对比 | logits 最大绝对差 | AP 绝对差 |
|---|---:|---:|
| 同代码、同 batch 的两次提取 | `6.10e-05` | `6.40e-06` |
| `predict_logits(512)` 两次调用 | `5.34e-05` | — |
| 提取 vs `predict_logits(512)` | `6.87e-05` | `6.31e-06` |
| 提取 vs `predict_logits(256)` | `7.63e-05` | `6.21e-07` |

`rho_in` 分块：`g` 与 `log1p_n` 块**逐位相同**（0.0），`mean_h` `8.9e-07`、
`mean_h2` `2.4e-06`。同一 batch 内独立重建 `rho_in` 的差为 `1.4–2.1e-06`，
`rho(rho_in)` 重放差为 `0.0`。

因此把“独立重放”门槛改为有依据的双条件：**AP 差 `<= 1e-5`**（真正有意义的
判据，实测 `<= 6.4e-6`）**且 logit 差 `<= 1e-3`**（粗粒度路径/布局错误检测；
真实 bug 会是 O(0.1–1)，比实测扩散大 3–4 个数量级）。所有实测值照样记录。
`rho_in` 重建 / 重放 / `contrib+rho==logits` 的门槛不放宽（`1e-5` / `1e-5` / 逐位）。

这不是判定阈值变更：`AP0`、`0.005`、`64/127`、`0.003`、epoch 8、配方和评价点
全部不变；改的是一条正确性检查的可达容差，且改动发生在任何正式提取或残差头
训练之前，依据同时写入 `analysis/feature_extraction_audit.json`、
`analysis/readout_comparison.json` 与 `REPORT.md`。

---

## 2. 阶段 B：两个极小残差头

原模型始终冻结；残差头只学习对既有预测的修正：

```
new_logits = frozen_logits + head(standardized_rho_in)
```

| 组 | 架构 | 初始化 |
|---|---|---|
| `L` | `Linear(d, 128)` | weight = 0，bias = 0 |
| `N` | `Linear(d, 128) -> SiLU -> Linear(128, 128)` | 第一层 = `torch.manual_seed(0)` 下 PyTorch 默认 `nn.Linear` 初始化；最后一层 weight = 0、bias = 0 |

不引入 attention、消息传递、RDKit 描述符或 Morgan 指纹。

### 共用配方（两组完全相同）

| 项 | 值 |
|---|---|
| 可训练参数 | 仅残差头（`L` 22,016；`N` 38,528） |
| 优化器 | AdamW，`lr=3e-4`，`weight_decay=1e-3`，`betas=(0.9,0.999)`，`eps=1e-8` |
| Loss | 原始 masked BCE，所有有效标签条目等权（NaN 不进损失） |
| Batch | 1024（全 Train 342 步 + 1 步 = 343 步/epoch） |
| Epochs | 8（无 scheduler、无 early stopping） |
| 精度 | FP32 |
| 种子 | 0（初始化与行顺序） |
| 行顺序 | `torch.Generator().manual_seed(0)` + 每 epoch `torch.randperm`；两组共用同一 plan（记录 plan 哈希） |
| 梯度裁剪 | 无（不在冻结配方内） |
| 选择 | 固定 epoch 8 为主指标；最佳 Validation AP 仅辅助 |
| 评价点 | epoch `0, 2, 4, 8`（epoch 0 = 更新前） |

禁止：对原始 LCRC 模型做 `backward` 或 optimizer update；正类权重、focal loss、
任务加权、重采样；任何额外 seed 或额外架构。

阶段 B 门槛（任一失败 → `INCOMPLETE`）：

- 阶段 A `all_passed`，缓存文件哈希与 manifest 一致；
- 标准化后 Train/Valid 特征 SHA256 与阶段 A 记录**逐位一致**；
- 行 plan 与 Validation/Test 行完全隔离，两组 plan 哈希相同；
- 两组 epoch 0 残差恰为 0，`new_logits` 与 `frozen_logits` 逐位相等；
- 两组完成 8 epoch 与 4 个评价点；
- 冻结模型参数指纹不变、无梯度；冻结 logits 缓存哈希不变；
- 无 NaN/Inf 事件；总 GPU 时间在预算内。

---

## 3. 每次评价记录

每臂每次评价写：Train masked BCE（全量）、Train macro AP、官方 Validation
macro AP、127 个可评分任务的逐任务 AP（`per_task_ap.csv`）、与该组 epoch 8
主值的差、与原 checkpoint 的 AP 差、GPU 耗时与显存峰值。跨臂的“与另一个
残差头 AP 差”在 `analysis/readout_curve.csv` 与
`readout_comparison.json` 中给出（两组共用同一行顺序与评价点）。

---

## 4. 固定判定

```
AP0   = 0.2374902968831333
AP_L  = Linear head epoch8 AP
AP_N  = Nonlinear head epoch8 AP
D_L   = AP_L - AP0
D_N   = AP_N - AP0
D_NL  = AP_N - AP_L
```

| 标签 | 规则（两组都必须完整运行） |
|---|---|
| `READOUT_SIGNAL` | 至少一个头：`D >= +0.005` **且** 127 个可评分任务中 `>= 64` 个 AP 改善 |
| `NONLINEAR_EXTRA_SIGNAL` | 在 `READOUT_SIGNAL` 成立的前提下 `D_NL >= +0.003`（附加标志，不替换主标签） |
| `WEAK_OR_MIXED` | `max(D_L, D_N) > 0` 但未满足 `READOUT_SIGNAL` |
| `NO_FAST_READOUT_SIGNAL` | 两组完整运行且 `D_L <= 0` 且 `D_N <= 0` |
| `INCOMPLETE` | 正确性/资源/数据隔离门槛失败、正式结果缺失或不可比较 |

“改善”= 该任务在 epoch 8 的 AP 严格大于冻结基线 AP（任务 45 不计入分母）。
这些是**资源决策标签**，不是统计显著性结论。

---

## 5. 资源与执行

| 项 | 值 |
|---|---|
| GPU 预算 | **<= 0.4 A100 GPU-hour**（smoke + 提取 + 两个头，全部计入） |
| 执行主机 | `res`（`a100-2`），pool `res-gpu1`（`CUDA_VISIBLE_DEVICES=1`） |
| 运行顺序 | `fr_preflight` → `fr_extract` → `fr_heads`（同一 commit，顺序执行） |
| 工作流 | `rr doctor res` → `rr deploy res` → `rr run res …` → `rr pull` |
| 超预算处理 | 优雅停止，保留已完成结果，不追加预算 |

每个脚本都带 soft deadline：超时即保存已完成状态并以 `INCOMPLETE` 退出。

---

## 6. 交付

```
PROTOCOL.md
code/        fr_common / fr_metrics / fr_model / fr_extract / fr_heads / fr_compare / fr_preflight
tests/       test_fr_probe.py（CPU-only，12 项）
analysis/    feature_extraction_audit.json
             frozen_baseline_replay.json
             readout_comparison.json
             per_task_deltas.csv
             readout_curve.csv
runs/        preflight/, feature_cache/, heads/{L,N}/（曲线、逐任务 AP、最终 checkpoint）— git-ignored
REPORT.md
```

大规模缓存与 checkpoint 不入 Git；其 SHA256 与来源写入 `analysis/*.json`。

---

## 7. 解释边界（预先声明）

本轮不得声称：

- 读出探针失败 ⇒ S1/S2 表征无效；
- 残差头改善 ⇒ 原模型存在某个特定机制缺陷；
- 单 seed 提升具有统计显著性；
- 已使用过的 Validation 可充当独立最终确认。

本轮不得：完整 LCRC 训练、额外 seed、额外架构、正类权重或新 loss、ECFP 等
外部特征、修改 S1/S2 编码、读取 Test，或自动开启下一轮。
