# REPORT — LCRC PCBA 学习率平台诊断（`lcrc-pcba-lr-probe-v1`）

Round: `lcrc-pcba-lr-probe-v1`
类型：优化诊断（不是新架构、不是完整基准复现）
数据集：`ogbg-molpcba`，官方 Train/Validation split；**官方 Test 从未读取、从未评价**
模型：既有 LCRC-C1 / NO_BOND_MEAN，245,184 参数，128 任务输出
初始权重：`lcrc_pcba_scale_pilot_v1/runs/pcba_seed0/checkpoint.pt`（epoch 29，AP 0.23449277）
正式训练 commit：`9b5a61a9fead1c2276c864f073e571fc454e68d4`
结果目录：`tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/`

---

## 1. 结论摘要

| 项 | H（高 LR，对照） | L（低 LR，实验） |
|---|---|---|
| AdamW 固定 LR | `3e-4` | `5e-5` |
| 训练状态 | `COMPLETED`，12/12 epoch | `COMPLETED`，12/12 epoch |
| 最佳 Validation AP | **0.23449194 @ epoch 0** | **0.23749030 @ epoch 9** |
| epoch12 Validation AP | **0.23245139** | **0.23680089** |
| epoch12 − 历史 0.23449 | −0.00203861 | +0.00231089 |
| 最佳 AP − 精确历史 0.23449277 | −0.00000083 | +0.00299753 |
| 最佳 epoch 后最终回落 | +0.00204055 | +0.00068940 |
| 总 GPU 时间（该组） | 0.40570 GPU-h | 0.31618 GPU-h |

主比较：

```
D_LR = AP_L(epoch12) - AP_H(epoch12)
     = 0.23680089235865226 - 0.23245139250096544
     = +0.004349499857686817

D_old = AP_L(epoch12) - 0.23449
      = 0.23680089235865226 - 0.23449
      = +0.0023108923586522556
```

**决策标签：`NO_CLEAR_INCREMENT`。**

原因：`D_LR = 0.0043495 < 0.005`，且 `D_old = 0.0023109 < 0.005`。
低 LR 确实比高 LR 的 epoch12 AP 高约 0.00435，但未达到预先注册的 0.005 门槛；
低 LR 相对历史最佳也只高约 0.00231，同样未达到门槛。高 LR 的 epoch12 AP
低于历史最佳，因此也不满足 `TRAINING_SIGNAL_NOT_LR_SPECIFIC` 的“两组都提高 ≥0.005”。

决策按协议使用字面量 `0.23449`；上表“精确历史”行使用实际历史最佳
`0.2344927704` 作对照。

这是一条**接近但未过门槛**的诊断信号，不是统计显著结论，也不是模型能力上限。
本轮没有延长训练、没有增加 seed、没有增加 LR 候选、没有 early stopping、
没有使用混合精度、没有继续优化 collate。

---

## 2. 实验定义与执行状态

两组只差固定学习率；共同配置：

| 项 | 值 |
|---|---|
| 模型 | LCRC-C1，128 输出，`MODEL_CONFIG` 与历史完全一致 |
| batch size | 128 |
| 优化器 | AdamW，wd `1e-3`，betas `(0.9, 0.999)`，eps `1e-8` |
| dropout | `0.10` |
| gradient clipping | `5.0` |
| scheduler | **无**；两组固定 LR |
| 精度 | FP32 |
| seed | `0`；两组训练起始 RNG 显式重置 |
| 顺序 | 同一 CPU generator：`manual_seed(0)` + 每 epoch `randperm` |
| 数据搬运 | `PinnedStager(depth=2)`，未进一步改 collate |
| epochs | 12 个完整 epoch；无 early stopping |
| 评价 epoch | `0, 3, 6, 9, 12` |

**AdamW 重新初始化（明确限制）**：历史 checkpoint 不含 AdamW moments、step count
或 scheduler 状态。两组都从零 AdamW 状态开始；因此这是“从相同权重出发的新优化器
探针”，不是无缝续训。该限制写入每个 checkpoint 的 `history`、两组 `run.json`、
`analysis/initialization_audit.json` 和本报告。

正式运行：

| 组 | `rr` run ID | pool | 实际 GPU | 状态 |
|---|---|---|---|---|
| H | `lr-probe-H12-20261105-193541-317733c4` | `res-gpu0` | `CUDA_VISIBLE_DEVICES=0` | `COMPLETED` |
| L | `lr-probe-L12-20261105-193608-6506e97b` | `res-gpu1` | `CUDA_VISIBLE_DEVICES=1` | `COMPLETED` |

两组并行运行（H 先启动约 27 s）。H 的 GPU0 上有外部共租训练进程
（用户 `wujiayu`，PID 3841932，约 16 GB 显存）；GPU0 仍有约 24 GB 空闲，
预检和正式训练均正常完成。共租只显著影响 H 的吞吐（见第 8 节），不改变
AP 的数学定义或评价规则。

---

## 3. 训练前正确性门槛（全部通过）

预检在两块 GPU 上分别运行，本地 `lr_audit.py` 汇总；两份 preflight 的
`all_passed = true`，`initialization_audit.json` 与
`paired_training_order.json` 的 `passed = true`。

| 门槛 | 结果 |
|---|---|
| 两组加载的参数 SHA/fingerprint 相同 | ✅ `540b27c79b8d75014f2960b806791e282437577fcd1e2c58562071ee71eb34fc` |
| 逐 tensor 与历史 checkpoint `state_dict` 相同 | ✅ bitwise equal |
| 历史 checkpoint SHA | ✅ `b2affec75fdb947417052f11293922d1e504d490ca48e8b2a074355be6d8edf8` |
| 模型配置 / 128 输出 / 无 `b_mean` / dropout 0.10 / FP32 | ✅ |
| feature-spec hash 与历史一致 | ✅ `7270f767…` |
| Train-only norm fingerprint 与历史一致 | ✅ `6ce50f98…` |
| 两个 GPU 的 epoch0 AP 重现 0.23449（容差 1e-4） | ✅ H `0.2344933037`（Δ `5.33e-7`）；L `0.2344917693`（Δ `1.00e-6`） |
| 两组 epoch0 AP 差值 | `1.53e-6`（仅 GPU 浮点顺序噪声） |
| 12 epoch 行号顺序完全一致 | ✅ 同一 `global_plan_sha256 = 1357fc69…`；每 epoch row/batch hash 均相等 |
| 每个 epoch 都是 train 的完整排列，且无 valid/test 行 | ✅ 12/12 |
| `PinnedStager` 真实训练接入 | ✅ batch/label 等价、loss/梯度/一次 AdamW 更新 smoke 通过 |
| 官方 Test 隔离 | ✅ Test 行 finite 数 = 0；未索引、未评价 |
| 数据契约 | ✅ 437,929 分子 / 128 任务 / 官方 gids / train 350,343 / valid 43,793 / test 43,793 |
| Train / Valid 已观测标签 | ✅ 27,162,842 / 3,477,656；标签只取 {0,1} |

`PinnedStager` 预检细节（H / L）：

| 指标 | H | L |
|---|---|---|
| batch 字段逐项相等 | ✅ | ✅ |
| label NaN mask + finite 值相等 | ✅ | ✅ |
| max loss abs diff | 0.0 | `3.7e-9` |
| max grad abs diff | `3.7e-9` | `2.8e-9` |
| 一次更新后 max weight diff | `5.4e-7` | `1.2e-7` |
| max logits abs diff | `5.7e-6` | `3.8e-6` |

正式运行中，两组每个 epoch 记录的 CPU/CUDA RNG state hash 也完全相等
（`comparison.json.paired_integrity.rng_pair.all_cpu_equal = true`，
`all_cuda_equal = true`），说明在相同权重与相同 batch 顺序下，dropout mask 相同。

**预检期间的一次失败（已修正，未进入正式训练）**：第一次 preflight 使用
`torch.equal` 比较含 NaN 的标签，NaN != NaN 被误判为不等；同时在两个模型前向之间
只重置一次 RNG，导致 dropout mask 不同。修正为 NaN-aware 标签比较 + 每次前向
前重置 RNG 后重跑，以上门槛才全部通过。被拒绝的 preflight run ID 记录在
`analysis/run_ids.json`。

---

## 4. 训练曲线

### H（`3e-4`）

| epoch | Train loss | Validation AP | 有效任务 | 说明 |
|---:|---:|---:|---:|---|
| 0 | — | **0.23449194** | 127 | 历史 epoch29 权重，更新前 |
| 1 | 0.03390770 | — | — | |
| 3 | 0.03373332 | 0.23224535 | 127 | |
| 6 | 0.03345592 | 0.23096920 | 127 | |
| 9 | 0.03322699 | 0.23128480 | 127 | |
| 12 | 0.03304152 | **0.23245139** | 127 | 最终 |

H 的 epoch0 复现值为 0.23449194，与历史最佳相差 −8.3e-7（GPU 浮点噪声）；
从 epoch3 起 Validation AP 明显低于历史最佳，之后在 0.2310–0.2325 之间恢复，
但 12 epoch 内从未超过 epoch0。H 的**最佳 checkpoint 就是 epoch0 的同一组历史权重**
（新目录中的 checkpoint 容器，`checkpoint_best.pt`，sha `312625e9…`）。

### L（`5e-5`）

| epoch | Train loss | Validation AP | 有效任务 | 说明 |
|---:|---:|---:|---:|---|
| 0 | — | **0.23449229** | 127 | 同一历史权重，更新前 |
| 1 | 0.03288760 | — | — | |
| 3 | 0.03267627 | 0.23742690 | 127 | |
| 6 | 0.03254369 | 0.23596188 | 127 | |
| 9 | 0.03244918 | **0.23749030** | 127 | 最佳 |
| 12 | 0.03237538 | 0.23680089 | 127 | 最终 |

L 从 epoch3 起的四个评价点都超过历史最佳；最佳为 epoch9 的 0.23749030，
epoch12 回落到 0.23680089。L 的最佳 checkpoint 为 epoch9
（`checkpoint_best.pt`，sha `b158f166…`）。

### 曲线层面的观察

- 两组 Train loss 都单调下降；无发散、无 NaN、无非有限梯度。
- H 的 grad norm 最大值 `0.3848`，L 为 `0.3874`；两组 `grad_clip_events = 0`。
  （历史从随机初始化训练时 grad norm 可达 11.35；本轮从 epoch29 权重出发，
  梯度尺度已经很小，这是预期差异，不是 bug。）
- H：Train loss 改善，但 Validation AP 从 0.23449 降到 0.23245。
- L：Train loss 改善，Validation AP 也从 0.23449 升到 0.23680。
- L 在 epoch12 同时具有更低 Train loss 和更高 Validation AP；没有出现
  “验证改善但训练损失恶化”的反向组合。详见第 7 节。

---

## 5. 主比较与决策

```
H epoch12 AP = 0.23245139250096544
L epoch12 AP = 0.23680089235865226

D_LR  = +0.004349499857686817
D_old = +0.0023108923586522556   (相对 0.23449)
D_old_vs_historical_checkpoint = +0.0023081219655188834  (相对 0.2344927704)
```

| 判定条件 | 结果 |
|---|---|
| `D_LR >= 0.005` | ❌ `0.0043495` |
| `D_old >= 0.005` | ❌ `0.0023109` |
| 两组 epoch12 都相对旧最佳提高 ≥0.005 | ❌ H 为 −0.0020386 |
| **标签** | **`NO_CLEAR_INCREMENT`** |

补充的“最佳 vs 最佳”比较（不是预注册主指标）：

```
L best (0.23749030 @9) - H best (0.23449194 @0) = +0.0029983568
```

同样低于 0.005。也就是说，低 LR 的方向性优势在 epoch12 和最佳 epoch 上都存在，
但幅度不足以触发预先注册的 `LR_SIGNAL`。

这些标签只是资源决策标签；本轮没有做 seed 重复，也没有统计显著性检验。

---

## 6. 逐任务差异（epoch12，L − H）

127 个可评分任务全部参与比较：

| 统计量 | 值 |
|---|---:|
| 平均 AP 差 | **+0.0043495** |
| 中位 AP 差 | +0.0050513 |
| 标准差 | 0.0177529 |
| 最小（L 最差） | −0.0663581（task 112） |
| 最大（L 最好） | +0.0684661（task 11） |
| L 更好的任务数 | 94 / 127 |
| H 更好的任务数 | 33 / 127 |
| 持平 | 0 |

L 的最大优势任务：11（+0.06847）、105（+0.05293）、31（+0.04882）、
63（+0.03626）、106（+0.03279）。
H 的最大优势任务：112（−0.06636）、62（−0.05668）、15（−0.05454）、
13（−0.04843）、69（−0.04479）。

最佳 epoch 的逐任务比较（L epoch9 vs H epoch0）均值为 `+0.0029984`，
中位 `+0.0040009`，标准差 `0.0239936`；L 更好 85 个、H 更好 42 个。
task 108 出现 −0.2 的极端差（H 0.25 vs L 0.05），说明单任务样本量小、
均值受少数任务影响明显；本轮不据此做任务级结论。

完整逐任务表见 `runs/H_high_lr/per_task_ap.csv`、`runs/L_low_lr/per_task_ap.csv`
与 `analysis/comparison.json.per_task`。

---

## 7. 验证改善 vs 训练损失改善

| 组 | Train loss (epoch1 → 12) | Validation AP (epoch0 → 12) | 是否出现背离 |
|---|---|---|---|
| H | 0.03390770 → 0.03304152（改善） | 0.23449194 → 0.23245139（恶化） | **训练损失改善但验证恶化** |
| L | 0.03288760 → 0.03237538（改善） | 0.23449229 → 0.23680089（改善） | 两者同向改善 |

没有出现“验证改善但训练损失恶化”的情况。
H 的“训练损失改善但验证恶化”是本轮最重要的诊断现象：在固定较高 LR 下，
模型继续拟合训练分布，但 Validation AP 没有恢复；低 LR 则同时降低了 Train loss
并提高了 Validation AP（尽管幅度未过 0.005 门槛）。

---

## 8. 资源与真实吞吐

总预算 `1.5 A100 GPU-hours`，实际使用：

| 项 | GPU-seconds | GPU-hours |
|---|---:|---:|
| H 正式训练 | 1460.51 | 0.405696 |
| L 正式训练 | 1138.25 | 0.316180 |
| H preflight/smoke | 19.71 | 0.005474 |
| L preflight/smoke | 15.59 | 0.004330 |
| **合计** | **2634.06** | **0.731680** |

`0.73168 ≤ 1.5`，预算未超。两组均未触发 soft/hard wall limit。

### 完整 epoch 实测（PinnedStager 真实训练路径）

| 指标 | H（`res-gpu0`，有共租） | L（`res-gpu1`） |
|---|---:|---:|
| Train seconds / 12 epoch | 1405.33 | 1097.91 |
| Validation seconds / 5 次评价 | 44.35 | 32.17 |
| 端到端 epoch 时间（12 epoch 平均，含 valid） | 120.03 s | 93.64 s |
| 端到端吞吐 | **2918.9 molecules/s** | **3741.5 molecules/s** |
| Train-only step p50 | 43.30 ms | 33.04 ms |
| Train-only steps/s | 23.38 | 29.93 |
| Train-only 吞吐 | 2991.5 molecules/s | 3829.2 molecules/s |
| 峰值显存 allocated / reserved | 86.15 MB / 121.63 MB | 同左 |

**吞吐偏差说明**：H 在 GPU0 上与外部共租进程共享 GPU，step p50 从 preflight 的
36.03 ms 升到 43.30 ms；L 在空闲 GPU1 上为 33.04 ms（preflight 33.32 ms）。
因此 H 的端到端吞吐比 L 低约 22%，H 的 GPU-hours 也更高。这是**执行环境偏差**，
不是 LR 的效果；AP 计算本身不依赖 step 时间。报告保留实际测得的 H/L 吞吐，
没有用 L 的空闲 GPU 数字替代 H。

本轮的数据传输管线是已经通过正确性门槛的 `PinnedStager`；没有继续优化 collate，
也没有使用 pageable 路径跑正式训练。上表的吞吐是**完整 epoch 的实测值**，
不是 300-step 外推。

---

## 9. Checkpoint 与复现审计

正式训练程序在训练中原子保存最佳 checkpoint，结束后保存最终 checkpoint；
每个 checkpoint 含模型 state、AdamW state、完整 RNG state、初始化来源、
feature-spec/norm/split/label 身份、order plan hash、recipe 和 environment。

| 组 | checkpoint | epoch | sha256 | CPU 重算 AP | 与 run.json Δ |
|---|---|---:|---|---:|---:|
| H | `checkpoint_best.pt` | 0 | `312625e93874a33538af93a1290ff7f624397a48dc7599b33cc47b8a97ec26a8` | 0.23449465 | 2.71e-6 |
| H | `checkpoint_final.pt` | 12 | `0c7f7ca2db82fc20cf96b7ef06f558d19597b47eed025dd3fcc4bae5efd0fc76` | 0.23245158 | 1.88e-7 |
| L | `checkpoint_best.pt` | 9 | `b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5` | 0.23749401 | 3.72e-6 |
| L | `checkpoint_final.pt` | 12 | `91a2e9d831cc8cc40a14522c8904bf9c4f91208896f978d8ed8a5fa236c8ae8a` | 0.23680425 | 3.36e-6 |

训练程序自身的 best-checkpoint reload 校验：

- H：`0.2344912678` vs stored `0.2344919401`，Δ `6.72e-7`
- L：`0.2374902029` vs stored `0.2374902969`，Δ `9.40e-8`

全部在 `1e-4` 容差内。`analysis/checkpoint_audit.json` 的 `passed = true`。
历史 epoch29 权重未被覆盖；H 的 `checkpoint_best.pt` 是同一组历史权重在新目录中的 checkpoint 容器，
不是对 `lcrc_pcba_scale_pilot_v1/runs/pcba_seed0/checkpoint.pt` 的修改。

每组的 `run.json` 还记录了：

- 初始化方式：历史 epoch29 checkpoint → 新 AdamW；
- `source_checkpoint_sha256 = b2affec7…`；
- `adamw_reinitialised = true`、`not_seamless_continuation = true`；
- 完整 RNG state（`rng_state.pt`）与逐 epoch RNG hash；
- `training_order.json`（每 epoch row/batch hash、首 batch、隔离检查）；
- `test_evaluated = false`。

---

## 10. Git commit、`rr` run ID、测试与偏差

### Git commits

| commit | 内容 |
|---|---|
| `385f8f6df07e` | 初始 protocol、实现、测试 |
| `dd947d00e052` | 预检修复：NaN-aware label 比较 + 每次前向重置 dropout RNG |
| `9b5a61a9fead1c2276c864f073e571fc454e68d4` | 配对预检全通过；正式 H/L 训练使用的 commit |
| 最终分析/报告 commit | 见 `git log --oneline` 最后一条；包含 `analysis/*.json` 与 `REPORT.md` |

### `rr` run IDs

| 用途 | run ID | 状态 |
|---|---|---|
| Preflight H | `lr-probe-preflight2-H-20261105-192920-cb3f9d33` | completed，all gates pass |
| Preflight L | `lr-probe-preflight2-L-20261105-192942-d4cd9f48` | completed，all gates pass |
| Formal H | `lr-probe-H12-20261105-193541-317733c4` | completed，12/12 |
| Formal L | `lr-probe-L12-20261105-193608-6506e97b` | completed，12/12 |
| 被拒绝的 preflight H | `lr-probe-preflight-H-20261105-192223-9e2bf4d7` | failed，未进入正式训练 |
| 被拒绝的 preflight L | `lr-probe-preflight-L-20261105-192253-da8d74a1` | failed，未进入正式训练 |

### 本地测试

- `uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/tests/test_lr_probe.py`
  → **10 passed**（5 fast + 5 slow，约 11.5 s）。
- 测试覆盖：recipe/arm 常量、与 `pc_train._batches` 逐 batch 一致、顺序可重复且
  不消耗全局 RNG、NaN-aware label 比较、决策标签边界、历史 checkpoint bitwise
  加载、epoch0 CPU AP、perf v1 的 320-batch 前缀 hash、完整 12-epoch 行隔离。
- `uv run ruff check tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1` → all checks passed。
- 另用 256 train / 128 valid 的 CPU 子集跑通完整 `lr_train.train` 单 epoch 流程
  （临时目录，不作为结果），确认曲线/checkpoint/RNG/reload 落盘路径无运行时错误。
- 预检 `initialization_audit.json`、`paired_training_order.json`、
  `checkpoint_audit.json` 均 `passed = true`；`comparison.json` 的决策规则按预注册执行。

### 偏差与限制

1. **AdamW 重新初始化**：历史 checkpoint 无优化器状态；两组从零 AdamW 开始，
   不能与历史 optimizer 轨迹做无缝比较。
2. **单 seed、12 epoch、无 scheduler**：短窗口诊断；不能外推为长周期或多 seed 结论。
3. **H 的 GPU0 共租**：外部进程约 16 GB 显存，H step p50 43.30 ms vs L 33.04 ms；
   吞吐差异已记录，AP 评价不受影响。研究者曾授权在 GPU0 被占用时改用 GPU1；
   实际 GPU0 仍有约 24 GB 空闲且正式训练正常完成，因此保留协议分配
   H=`res-gpu0`、L=`res-gpu1`，没有改为单 GPU 顺序运行。
4. **预检第一次失败**：NaN 标签比较和 dropout RNG 重置问题，已修正并重跑；
   被拒绝的 run ID 保留在 `analysis/run_ids.json`。
5. **epoch0 AP 的微小 GPU 差异**：preflight 与正式运行的 epoch0 AP 在 1e-6 量级
   不同（例如 H preflight 0.23449330 vs formal 0.23449194），来自 GPU 浮点归约顺序；
   两者都在历史 0.23449 的 1e-4 容差内。
6. **逐任务差异异质性大**：均值 +0.00435 但标准差 0.01775，存在 −0.066 到
   +0.068 的任务级差异；不能据此声称某些任务稳定受益。
7. **官方 Test 从未读取**：本轮 AP 不可与论文 Test AP 直接比较。
8. **决策标签不是统计显著性**：`NO_CLEAR_INCREMENT` 只表示未达到预注册的
   0.005 资源决策门槛，不表示低 LR 一定无效，也不表示高 LR 一定更差。

---

## 11. 交付物

| 路径 | 内容 |
|---|---|
| `PROTOCOL.md` | 预注册实验定义、门槛、决策规则、执行偏差 |
| `code/lr_common.py` | 冻结常量、路径、哈希、决策规则 |
| `code/lr_data.py` | 只读数据加载、顺序 plan、隔离检查 |
| `code/lr_model.py` | 历史 checkpoint 加载、fingerprint、官方 AP |
| `code/lr_preflight.py` | 单 GPU 预检 + PinnedStager smoke + 吞吐 smoke |
| `code/lr_train.py` | 12-epoch 正式训练、逐 epoch 落盘、最佳/最终 checkpoint |
| `code/lr_audit.py` | 本地 H/L 配对预检汇总 |
| `code/lr_compare.py` | 主比较、逐任务差异、资源与决策 |
| `tests/test_lr_probe.py` | 10 个针对性正确性测试 |
| `analysis/preflight_H.json`、`preflight_L.json` | 两份 GPU 预检完整记录 |
| `analysis/initialization_audit.json` | 初始权重/配置/norm/epoch0 AP 配对审计 |
| `analysis/paired_training_order.json` | 12 epoch 行号顺序配对审计 |
| `analysis/comparison.json` | 主比较、曲线、逐任务、资源、标签 |
| `analysis/checkpoint_audit.json` | 四个 checkpoint 的 CPU 重算 AP |
| `analysis/run_ids.json` | Git commit、`rr` run ID、被拒绝的预检 |
| `runs/H_high_lr/` | H 的 `run.json`、curve、per-task、selection、RNG、best/final checkpoint |
| `runs/L_low_lr/` | L 的同上产物 |

`runs/` 按仓库惯例在本地生成并由 `.gitignore` 排除；文件已拉回本地目录，
可直接审计。官方 Test 在所有路径中均不可达。
