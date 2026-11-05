# REPORT — LCRC / OGB MolPCBA 冻结诊断与最小 loss 筛选（`lcrc-pcba-loss-screen-v1`）

- 数据集：OGB `ogbg-molpcba`，官方 Train / Validation / Test；Test 从未读取、从未评价。
- 模型：既有 LCRC-C1 / NO_BOND_MEAN，245,184 参数；结构未改。
- 源 checkpoint：`tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/runs/L_low_lr/checkpoint_best.pt`
- 源 checkpoint SHA256：`b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5`
- 正式运行 commit：`98bbf66c9f0893fdeebd2843ad59f894f5af6b7b`
- 正式运行日期：2026-11-05（UTC+08:00）
- 最终判决：**`SCREEN_NO_SIGNAL`**

> 本报告只报告预注册的 Phase A 诊断与 Phase B 固定 epoch-6 比较。没有新增实验，
> 没有改任务数、没有换 checkpoint、没有用 Test、没有把“最佳 checkpoint”当作主结果。

---

## 0. 一句话结论

冻结的 epoch-9 模型在 Train 上有明显拟合信号（macro AP `0.3883`），但 Validation
只有 `0.2375`，Train−Valid AP 平均差 `+0.1507`；差距在正例稀疏任务上最大，但在
正例充足任务上也存在。六 epoch 配对筛选中，任务等权 BCE（T）在固定 epoch 6 的
Validation macro AP 为 `0.235504`，低于历史 masked BCE 对照（M）的 `0.237227`，
`D6 = -0.001723`，未达到预注册的 `+0.003` 决策门槛；42/127 个可评分任务 T 更好，
低于 64/127 多数门槛。因此判为 `SCREEN_NO_SIGNAL`，不进入完整训练。

---

## 1. 执行身份与门槛

| 项目 | 结果 |
|---|---|
| Phase A 门槛 | 15/15 `true`，`diagnostic_summary.json.all_passed = true` |
| 模型权重等于源 checkpoint | 逐 tensor bitwise 相等 |
| Phase A 行数 | Train 350,343；Valid 43,793；Test 43,793 |
| Train/Valid/Test 重叠 | 0 / 0 / 0 |
| Test 有限标签数 | 0；`test_never_used = true`，`no_test_metric = true` |
| 历史 Valid AP 复现 | `0.23749175273654682` vs `0.2374902968831333`，绝对差 `1.4559e-06`（容差 `1e-4`） |
| 官方 AP vs 独立逐任务 AP | Train 差 `0.0`；Valid 差 `2.78e-17`（容差 `1e-8`） |
| 所有可计算 AP/AUC | 有限；无 NaN/Inf |
| 配对初始化审计 | 32/32 `true`，`paired_initialization.json.all_passed = true` |
| 配置审计 | `passed = true`，`mismatches = []` |
| 正式两组状态 | 均为 `COMPLETED`，6/6 epochs，评价点 0/2/4/6 齐全 |
| 非有限事件 / clipping | 0 / 0（两组） |
| 总 GPU 预算 | `0.3981` A100 GPU-hour，低于 `1.0` |

数据身份（本次实际核验）：label artifact SHA256
`5c3b5fbe642d80113522832fe01090097b860010868c59f8696bc247d5027d01`；
Train rows SHA256 `16169ef35ce1812eac577af939fa0416e79c670c015ecf5aa7b848a25d412dd9`；
Valid rows SHA256 `fcad7abd938cf3328b6761cd78c8170cf6bd389bef4b6d1f94b5ed0b4aec21ed`；
Test rows SHA256 `9171aa54ed4070dcbd6697b68d57560c7015c118168de4fa63b5e05a02c3cd62`。
特征 spec、split partition、Train-only norm fingerprint、cache manifest 也与冻结值一致
（`loss_definition_audit.json.checks` 全部 `true`）。

---

## 2. 九个必答问题

### Q1. Phase A 是否全部通过？冻结模型在 Train / Validation 上的整体表现是多少？

全部通过。模型在 Train 上有明显预测力，在 Validation 上明显下降：

| 指标 | Train | Validation |
|---|---:|---:|
| 官方 macro AP | **0.388335599** | **0.237491753** |
| macro ROC-AUC | 0.922556862 | 0.857382200 |
| 任务 AP 中位数 | 0.392683470 | 0.203242543 |
| 任务 AUC 中位数 | 0.933709879 | 0.881492397 |
| 正例率 | 0.013252479 | 0.016092448 |
| 有效标签条目 | 27,162,842 | 3,477,656 |
| 正例条目 | 359,975 | 55,964 |
| 可计算 AP 的任务数 | 128 | 127 |

Validation 的任务 45 只有 41 个有效标签、0 个正例，AP/AUC 不可计算；不填 0、不插值。
其余 127 个任务全部可评分。

**解释边界：** Train AP 是样本内拟合，不能单独当作泛化证据；它说明“模型并非连
Train 都拟合不了”，但不能仅凭这一点证明过拟合。它给出的是描述性的 Train→Valid
拟合差距。

### Q2. Train→Valid 差距有多大？是否普遍存在？

普遍存在。127 个可配对任务中：

- AP 差（Train − Valid）均值 **`+0.150702790`**，中位数 **`+0.132681368`**；
- 最小 `-0.114914779`（task 11），最大 `+0.605384644`（task 57）；
- **124/127** 个任务 Train AP 高于 Valid AP；只有 task 10、11、40 为负；
- 109 个任务差距 > `0.05`，38 个任务差距 > `0.20`；
- AUC 差均值 `+0.065249609`，中位数 `+0.045934796`，范围
  `-0.014599686` 到 `+0.541654547`。

所以这不是“少数任务拉低平均”的现象；绝大部分任务都有明显的样本内−样本外差距。

### Q3. 按 Train 正例数分层，差距来自哪里？

按 Train 正例数预先分层（分层只看 Train 标签，不按 Validation 好坏挑任务）：

| 分层 | 任务数 | Train 正例合计 | Train AP 均值 | Valid AP 均值 | ΔAP 均值 | ΔAP 中位数 | ΔAUC 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sparse_le_100`（≤100） | 14 | 730 | 0.367384 | **0.065894** | **+0.301490** | +0.290868 | +0.161622 |
| `medium_101_1000`（101–1000） | 63 | 29,913 | 0.318286 | 0.180141 | +0.136726 | +0.122649 | +0.062977 |
| `ample_gt_1000`（>1000） | 51 | 329,332 | 0.480619 | 0.354317 | +0.126302 | +0.113697 | +0.041558 |

三个分层都存在明显差距；**稀疏任务最严重**（Train AP 0.367，Valid AP 仅 0.066），
但“相对充足”分层的 Valid AP 也只有 0.354，ΔAP 仍为 `+0.126`。因此不能把问题
完全归因于 14 个稀疏任务；这是全谱系的 Train→Valid 下降，稀疏任务放大了它。
14/14 稀疏任务、51/51 充足任务、59/62 中等任务的 AP 差为正。

### Q4. 配对起点是否真的相同：权重、AdamW moments、RNG、顺序、首步输入？

是。`ls_pair_audit.py` 汇总两组 preflight，32/32 项通过：

- 源 checkpoint SHA 一致；两组模型 state_dict 逐 tensor 相同；
- 模型参数 fingerprint 两组均为
  `bd969c6328749735e8a509cd00ab272052b3d47f57ee97b2d576f9b62d91eee1`；
- AdamW state fingerprint 两组均为
  `6409b6b75c9e25c6adf17cc38229023b9e34293f7ec0e372d3cc79bafd143b0c`，
  与源 state 相同；step = `24642`；moments 未重新初始化；
- Python / NumPy / CPU torch / CUDA RNG 起点 hash 两组完全相同
  （combined `19c8fd10c1f80c0573b419a2649dc8401bf93a04691cfeab5ba81f6a0a10eec7`）；
- 六 epoch 预生成顺序的 global plan SHA 两组均为
  `46865ae7d177287a…`；训练行与 Valid/Test 行不重叠；
- first batch 的 rows SHA `182f7db814dba23f89ac0a12f0525e108baf1c65b4a670a29cdaa41e2d6cdb2e`、
  labels SHA `a6f7655704d40b0d6bc95e140bd8984523b334cd5f07d3e877f58443572fd9c9`
  两组一致；
- first-step logits 两组最大绝对差：preflight `5.72e-06`，正式运行 `3.81e-06`，
  均低于 `1e-5`；
- epoch-0（任何更新前）官方 Validation AP：M `0.237489513`，T `0.237492611`；
  两组差 `3.10e-06`（preflight 为 `1.00e-06`），均与历史 `0.237490297` 在 `1e-4` 内；
- 两组 `test_evaluated = false`；Test 标签全 NaN，未进入训练或评价。

这是“同一权重 + 同一 AdamW moments + 同一 RNG + 同一 batch 顺序”的新六 epoch
固定分叉；不是历史 epoch-10+ 轨迹的 bitwise 延续。

### Q5. T loss 是否严格按冻结公式实现？是否真的改变了梯度？

是。T 的公式冻结为：

```
L_T = N / (B*T) * sum_{i,t}( m_it * ell_it / n_t )
```

- `N = 350343`，`T = 128`，`B` 为当前 batch 实际行数；
- `n_t` 只由完整官方 Train 的 `task.labels[task.train_rows]` 计算；
- `n_t` 合计 `27,162,842`，与 Train 有效标签总数一致；128 个 `n_t` 全部 > 0；
  最小 `5,152`，最大 `332,662`，均值 `212,209.703125`；
  `n_task_sha256 = c14cb43150c3ebe4c5734ecc376e1865c2d6c4af1f233e9aaf74b31d75a509ae`；
- 无效标签先用 NaN→0 安全替换，再算 elementwise BCE，最后用 mask；没有按 batch
  动态重算 `n_t`，没有正类权重、focal loss 或权重裁剪；gradient clipping 仍为 `5.0`。

CPU 验证与首步数值：

- `cpu_validation_suite` 10/10 项通过（NaN 矩阵、完全无标签任务、不同 batch 大小、
  等覆盖等价、batch 加权汇总、梯度有限、真实任务权重改变等）；
- `loss_definition_audit.json` 的 12 项数据/公式 identity 检查全部 `true`；
- first-step M loss `0.0262591280043`，float64 numpy 参考 `0.0262591311`，
  绝对差 `3.10e-09`；M 直接调用历史 `pc_metrics.masked_bce_loss`，未重写；
- first-step T loss `0.0333172678947`，numpy 参考 `0.0333172694330`，
  绝对差 `1.54e-09`；
- 同一 batch 上 T−M 的梯度最大绝对差 `0.019809097`（阈值 `1e-8`），
  31 个梯度 tensor 全部有限；`task_weight_changed = true`。

T 的训练目标从 epoch 1 的 `0.042153` 降到 epoch 6 的 `0.040505`，但统一口径的
raw masked BCE 从 `0.032543` 升到 `0.032666`。这不是实现错误：T 优化的是任务等权
目标，raw BCE 只是日志口径；首步两者 raw BCE 完全相等，且 T loss 与独立参考一致。

### Q6. 固定 epoch 6 的主比较结果和预注册标签是什么？

| 量 | 值 |
|---|---:|
| `AP_0`（源 checkpoint 历史 Valid AP） | 0.2374902968831333 |
| `AP_M6`（历史 masked BCE 对照） | **0.23722683307799955** |
| `AP_T6`（任务等权 BCE） | **0.23550376491774172** |
| `D6 = AP_T6 - AP_M6` | **-0.001723068160257829** |
| `AP_M6 - AP_0` | -0.0002634638051337568 |
| `AP_T6 - AP_0` | -0.001986531965391586 |
| T 更好的任务数 / 可评分任务数 | **42 / 127** |
| 预注册标签 | **`SCREEN_NO_SIGNAL`** |

判定条件：两组都完成 6 epochs（是）；配置审计通过（是）；预算未超（是）；
`D6 >= +0.003`（**否**，实际为负）；`AP_T6 >= AP_0`（否）；任务多数 `>= 64/127`
（否）。按预注册规则，只要 `D6 < +0.003` 就是 `SCREEN_NO_SIGNAL`，因此不是
`SCREEN_MIXED`，也不是 `SCREEN_PROMISING`。

辅助最佳 checkpoint 只作工件，不参与主决策：

- M 最佳 epoch 4，`0.238629768`；
- T 最佳 epoch 0（即未更新起点），`0.237492611`；
- 若按“各自最佳”比较会得到 M 更好，但这不是预注册主比较；主比较固定 epoch 6。

### Q7. T 的改善/退化在任务间如何分布？是否集中在少数任务？

任务级分布（127 个可评分任务）：

- ΔAP 均值 `-0.001723068`；中位数 **`-0.002386399`**；
- 最小 `-0.069407970`（task 117，medium）；最大 `+0.055981703`（task 121，medium）；
- T 更好 42 个，M 更好 85 个，tie 0 个；
- 正贡献合计 `+0.481013638`，负贡献合计 `-0.699843294`；
- 正贡献集中度：top-1 `11.64%`，top-3 `29.26%`，top-5 `45.07%`，top-10 `69.53%`。

改善最大的 5 个任务：

| task | 分层 | Train 正例 | Valid 正例 | AP_M | AP_T | ΔAP |
|---:|---|---:|---:|---:|---:|---:|
| 121 | medium | 269 | 58 | 0.297480 | 0.353462 | +0.055982 |
| 114 | medium | 980 | 176 | 0.596675 | 0.640297 | +0.043622 |
| 96 | medium | 441 | 171 | 0.489891 | 0.531014 | +0.041123 |
| 112 | sparse | 81 | 23 | 0.193043 | 0.233176 | +0.040132 |
| 55 | sparse | 71 | 14 | 0.058107 | 0.094046 | +0.035939 |

退化最大的 5 个任务：

| task | 分层 | Train 正例 | Valid 正例 | AP_M | AP_T | ΔAP |
|---:|---|---:|---:|---:|---:|---:|
| 117 | medium | 322 | 9 | 0.405053 | 0.335645 | -0.069408 |
| 31 | sparse | 28 | 6 | 0.140089 | 0.091560 | -0.048529 |
| 120 | medium | 193 | 13 | 0.373636 | 0.333814 | -0.039822 |
| 118 | medium | 514 | 8 | 0.336118 | 0.300124 | -0.035994 |
| 13 | medium | 112 | 20 | 0.201400 | 0.174811 | -0.026590 |

结论：T 的正向改善确实集中在少数任务（top-10 占正贡献约 70%），但退化分布更广
（85 个任务为负），所以 macro 平均为负。不是“普遍小幅改善、被一两个任务拖累”。

### Q8. 分层与覆盖量说明什么？训练曲线是否稳定？

Phase B 分层结果：

| 分层 | 任务数 | 可评分 | T 更好 | M 更好 | ΔAP 均值 | ΔAP 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| sparse ≤100 | 14 | 14 | 6 | 8 | -0.000402 | -0.000187 |
| medium 101–1000 | 63 | 62 | 28 | 34 | -0.0000666 | -0.000652 |
| ample >1000 | 51 | 51 | 8 | **43** | **-0.004099** | -0.004550 |

T 在正例稀疏任务上略好/基本持平，但在正例充足任务上明显更差（8 好 / 43 差，
均值 `-0.0041`）。这与“任务等权会把更多相对权重给稀疏任务”的方向一致，但本轮
没有把这种重分配转化为整体 macro AP 增益。

覆盖量与 ΔAP 的 Spearman 描述性相关（不是显著性结论）：

| 覆盖量 | Spearman ρ | p |
|---|---:|---:|
| Train observed | -0.0211 | 0.814 |
| Train positive | **-0.3203** | 0.000242 |
| Valid observed | +0.0332 | 0.711 |
| Valid positive | **-0.2330** | 0.00840 |

即：Train 正例越少，T 相对 M 的 ΔAP 越偏正；但这一趋势不足以抵消充足任务上的
负贡献。

训练曲线（官方 Validation AP，评价点 0/2/4/6）：

| epoch | M AP | T AP |
|---:|---:|---:|
| 0 | 0.237489513 | 0.237492611 |
| 2 | 0.236642007 | 0.234990585 |
| 4 | 0.238629768 | 0.235943487 |
| 6 | **0.237226833** | **0.235503765** |

Train loss / raw masked BCE：

| 组 | epoch 1 train loss | epoch 6 train loss | epoch 1 raw BCE | epoch 6 raw BCE |
|---|---:|---:|---:|---:|
| M | 0.032426491 | 0.032333115 | 0.032426491 | 0.032333115 |
| T | 0.042153315 | 0.040505470 | 0.032542716 | 0.032665883 |

两组均无 gradient clipping（0 次）、无非有限事件；M 的 grad norm 均值约
`0.0609→0.0623`，T 约 `0.1930→0.1814`（T 的系数更大，量级不同不构成异常）。
峰值显存约 86 MB allocated / 122 MB reserved（两组相近）。曲线没有数值不稳定；
T 的 Validation AP 从起点后持续低于 M。

### Q9. 资源、执行偏差与结论边界是什么？下一步是否值得推进？

总 GPU 时间 `0.398142` A100 GPU-hour（预算 `1.0`，使用约 `39.8%`）：

| 项目 | GPU-hours | 说明 |
|---|---:|---|
| Phase A 只读诊断 | 0.016750 | 60.3 s；res-gpu1 |
| preflight M | 0.005524 | 19.9 s；res-gpu0 |
| preflight T | 0.003842 | 13.8 s；res-gpu1 |
| 正式 M | 0.208506 | 750.6 s；res-gpu0 |
| 正式 T | 0.163519 | 588.7 s；res-gpu1 |
| **合计** | **0.398142** | 约 23.9 min GPU 时间 |

执行偏差与共租：

- 正式运行期间 GPU0 有共租进程（PID `3841932`，`python3`，约 16.08 GB，
  `nvidia-smi`  utilization 100%）。M 的 step p50 为 `43.27 ms`，T 为 `33.36 ms`；
  M 的 train epoch 约 116–118 s，T 约 88–95 s。M 仍在 1200 s 软限内完成。
  该吞吐差只作为执行环境记录，**不得解释为 loss 或模型差异**。
- `runs/M_micro/run.json` 与 `runs/T_task_macro/run.json` 中的
  `gpu_occupancy_start` 在正式运行时是在收尾组装记录时采样，故时间与
  `gpu_occupancy_end` 相同；这是日志仪表错误，不影响训练、评价或指标。
  真实起始占用见 preflight JSON（M 13:17:36Z，T 13:18:01Z）。运行后已修正
  `ls_train.py` 为启动时采样；仅为仪表修复，正式结果未重跑、run.json 未改写。
- 两个正式训练使用同一 commit `98bbf66c`；报告提交只增加分析、报告和上述仪表
  修复，不改变正式运行的训练定义。
- 没有提前停止、没有 scheduler、没有换 checkpoint、没有额外实验、没有 Test 访问。

结论边界：

- 单 seed、六 epoch 短程继续训练；不能外推为多 seed 或长周期结论；
- 源 checkpoint 的 AdamW moments 与 RNG 被恢复，但六 epoch 顺序是新固定分叉；
- Validation 已被多次用于研究决策；本轮不是独立确认；
- `+0.003` 是资源决策门槛，不是统计显著性阈值；
- 本报告没有 Test AP，不能与论文 Test 指标直接比较；
- `SCREEN_NO_SIGNAL` 只说明“在这个预注册短程筛选里，T 没有达到值得购买完整训练
  的信号”，不证明 T 在任何设置下都无效，也不证明 M 是最终最优损失。

**下一步：不推进任务等权 T 到完整训练。** 本轮到此停止；不新增实验。若未来继续，
需要新的预注册轮次，而不是在本轮结果上事后挑任务、改 epoch 或改 loss。

---

## 3. 预注册决策规则（逐字）

```
AP_M6 = M 组 epoch 6 官方 Validation AP
AP_T6 = T 组 epoch 6 官方 Validation AP
D6    = AP_T6 - AP_M6
AP_0  = 0.2374902968831333
```

| 标签 | 条件 |
|---|---|
| `SCREEN_PROMISING` | 两组都完成 6 epochs；`D6 >= +0.003`；`AP_T6 >= AP_0`；127 个可评分任务中 T 优于 M 的任务数 `>= 64` |
| `SCREEN_MIXED` | 两组完整完成；`D6 >= +0.003`；但未满足 `SCREEN_PROMISING` 的其他条件 |
| `SCREEN_NO_SIGNAL` | 两组完整完成；`D6 < +0.003` |
| `INCOMPLETE_OR_INVALID` | 任一组未完成；正确性、数据隔离或 checkpoint 验证失败；或两组存在不能解释的配置差异；或总 GPU 预算超限 |

本轮逐项：`both_completed = true`，`config_valid = true`，`budget_within_limit = true`，
`d6_ge_floor = false`，`ap_t6_ge_ap0 = false`，`task_majority = false`
（42 < 64）。因此标签为 **`SCREEN_NO_SIGNAL`**。

---

## 4. 产物

| 文件 | 内容 |
|---|---|
| `analysis/diagnostic_summary.json` | Phase A 全部指标、分层、门槛、数据身份、资源 |
| `analysis/train_valid_diagnostic.csv` | 128 任务逐任务 Train/Valid AP、AUC、计数、差值、分层 |
| `analysis/loss_definition_audit.json` | `n_t` 冻结值/哈希、公式 identity、CPU 数值验证 |
| `analysis/paired_initialization.json` | 32 项配对起点审计、首步数值、RNG/顺序/指纹 |
| `analysis/paired_comparison.json` | 固定 epoch-6 比较、逐任务 ΔAP、分层、曲线、资源、判决 |
| `runs/M_micro/` | 历史 masked BCE 对照的 run.json、曲线、逐任务 AP、checkpoint、RNG |
| `runs/T_task_macro/` | 任务等权 BCE 的 run.json、曲线、逐任务 AP、checkpoint、RNG |
| `code/`、`tests/` | 可复现实现与 8 个 CPU 测试（全部通过） |

`runs/` 下的 checkpoint 与预测数组按仓库惯例 gitignore；本报告与 `analysis/` 的
JSON/CSV 提交 Git。Phase A 的 `runs/A_diagnostic/preds/*.npy` 保留在远端运行目录，
其 hash 已写入 `diagnostic_summary.json`。

---

## 5. 最终声明

- Phase A：通过，冻结模型 Train AP `0.3883` / Valid AP `0.2375`，存在明显
  Train→Valid 拟合差距，稀疏任务最严重但非唯一来源。
- Phase B：两组完整、配对有效；`AP_M6 = 0.237227`，`AP_T6 = 0.235504`，
  `D6 = -0.001723`，42/127 任务 T 更好。
- 判决：**`SCREEN_NO_SIGNAL`**。
- 停止：不购买完整训练，不新增实验，不访问 Test。
