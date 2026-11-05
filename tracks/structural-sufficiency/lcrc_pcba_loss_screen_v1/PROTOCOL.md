# PROTOCOL — LCRC / OGB MolPCBA 性能瓶颈诊断与最小损失函数筛选（`lcrc-pcba-loss-screen-v1`）

Round: `lcrc-pcba-loss-screen-v1`
类型：冻结诊断（阶段 A） + 单 seed 六 epoch 配对损失筛选（阶段 B）
数据集：OGB `ogbg-molpcba`，官方 Train / Validation / Test 分区
模型：既有 **LCRC-C1 / NO_BOND_MEAN**，245,184 参数，128 任务输出；**结构不改**
源 checkpoint：`tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/runs/L_low_lr/checkpoint_best.pt`
源 checkpoint SHA256：`b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5`
源 checkpoint：epoch 9，Validation AP `0.2374902968831333`
官方 Test：**从未读取、从未评价**

本文件在第一次正式训练之前固定实验定义。阶段 A 的统计结果不用于自行改模型、
选任务、设计新 loss 或派生其他实验；阶段 B 只比较下面两个预注册损失。

---

## 1. 阶段 A：冻结模型 Train / Validation 诊断（不训练）

### 1.1 目标

在同一个 epoch-9 checkpoint 上回答：

> 模型是否在训练集上也缺乏预测能力，还是主要存在 Train→Validation 泛化差距？

阶段 A 不创建 optimizer、不更新任何参数、不训练。模型 `eval()`，所有前向在
`torch.inference_mode()` 下运行。只推理官方 Train 与 Validation；官方 Test 行在
标签工件中全为 NaN，任何有限 Test 标签都会使加载失败。

### 1.2 指标

对每个任务分别计算：

- Train / Validation AP；
- Train / Validation ROC-AUC；
- Train / Validation 有效标签数、正例数、正例率；
- `AP − 本分区正例率`（描述性排序收益）；
- `Train AP − Validation AP`、`Train ROC-AUC − Validation ROC-AUC`。

任务缺少正例或负例时，相关 AP/AUC 记为不可计算（JSON `null` / CSV 空值），
**不填 0、不插值**。

整体报告：macro AP、macro ROC-AUC、任务中位数；并按 **Train 正例数**预先分层：

| 分层 | Train 正例数 |
|---|---|
| 稀疏 `sparse_le_100` | ≤ 100 |
| 中等 `medium_101_1000` | 101–1000 |
| 相对充足 `ample_gt_1000` | > 1000 |

分层只由 Train 标签决定，不按 Validation AP 好坏挑任务。

### 1.3 正确性门槛（任一失败即停止整个任务）

- 源 checkpoint SHA256 与冻结值一致；
- 加载进模型后的 state_dict 逐 tensor bitwise 等于源 checkpoint；
- Train 行数 350,343、无重复/缺失；Validation 行数 43,793；
- Train / Validation 不重叠；
- 官方 Test 行有限标签数 = 0，且从未进入推理或评价；
- Validation AP 复现 `0.2374902968831333`，容差 `1e-4`；
- OGB 官方 AP 与独立逐任务 AP 均值一致（容差 `1e-8`）；
- 所有可计算 AP/AUC 有限。

### 1.4 输出

- `analysis/train_valid_diagnostic.csv`（逐任务指标表）；
- `analysis/diagnostic_summary.json`（整体、分层、数据身份、门槛、资源）。

预测数组写入 `runs/A_diagnostic/preds/*.npy`，按仓库惯例 gitignore；预测 hash
写入 summary。

---

## 2. 阶段 B：最小 loss 筛选

### 2.1 科学问题

现有训练对全部有效标签条目等权，而最终评价对任务等权。本轮只检验：

> 在不改变模型、数据、正负样本权重和学习率的情况下，按任务等权构造训练目标，
> 是否能在短程继续训练中提高 PCBA macro AP？

不检验 focal loss、正类重加权、采样策略或模型容量。

### 2.2 两个固定实验组

| 组 | 名称 | 角色 | loss | 计划 GPU pool |
|---|---|---|---|---|
| **M** | `M_micro` | 对照 | 历史 `pc_metrics.masked_bce_loss`，严格保持原实现 | `res-gpu0` |
| **T** | `T_task_macro` | 实验 | 任务等权 masked BCE | `res-gpu1` |

#### M 组

对全部有效标签条目求平均的 masked BCE：

```
L_M = sum_{i,t}(m_it * ell_it) / sum_{i,t}(m_it)
```

其中 `ell_it = BCEWithLogits(logit_it, y_it)`，`m_it` 为有效标签掩码。
实现直接调用历史 `pc_metrics.masked_bce_loss`，不重写。

#### T 组

对每个任务先按其**整个 Train 分区**的已观测标签数 `n_t` 取平均，再对 128 个任务
等权平均。固定 minibatch 估计为：

```
L_T = N / (B*T) * sum_{i,t}(m_it * ell_it / n_t)
```

- `N = 350343`：官方 Train 分子数；
- `T = 128`；
- `n_t`：完整官方 Train 中任务 t 的有效标签数，预先计算并冻结；
- `B`：当前 batch 的实际分子数；
- `m_it`：标签有效掩码；
- `ell_it`：`BCEWithLogitsLoss(reduction="none")`。

固定要求：

- `n_t` 只来自完整官方 Train；不得使用 Validation/Test；
- 128 个 `n_t` 全部 > 0；
- 不得按当前 batch 临时计算 `n_t`；
- 不得用 batch 内动态归一化替代公式；
- 无效标签先安全替换（NaN→0）再计算逐元素 BCE，随后使用 mask；NaN 不进入损失；
- 不引入正类权重、focal loss、权重裁剪；
- 保留原始 gradient clipping `5.0`。

T 只调整任务之间的权重，**不解决任务内部正负样本不平衡**。

### 2.3 严格配对设置

两组从**同一个 epoch-9 checkpoint** 开始：

- 同一模型 state_dict；
- 同一 AdamW optimizer state（不重新初始化 moments；源 step=24642）；
- 同一冻结 RNG 起点（Python / NumPy / CPU torch / 可见 CUDA generator）；
- 同一预生成六 epoch minibatch 顺序（`torch.Generator().manual_seed(0)` +
  每 epoch `torch.randperm`）。

共同配置：

| 项目 | 值 |
|---|---|
| 来源权重 | L 组 epoch-9 `checkpoint_best.pt` |
| 来源优化器状态 | 同一 checkpoint 的 AdamW state |
| 学习率 | 固定 `5e-5` |
| Weight decay | `1e-3` |
| AdamW betas / eps | `(0.9, 0.999)` / `1e-8` |
| Batch | `128` |
| Dropout | `0.10` |
| Precision | FP32 |
| Grad clipping | `5.0` |
| Scheduler | 无 |
| 训练数据 | 完整官方 Train |
| 训练长度 | **6 epoch** |
| Early stopping | 无 |
| GPU 传输 | `pf_pipeline.PinnedStager(depth=2)` |

本轮将源 checkpoint 视为共同分叉起点，建立新的固定六 epoch 顺序。AdamW 状态与
RNG 起点来自源 checkpoint，但**不宣称**与原历史第 10 epoch 之后的数据顺序或轨迹
bitwise 无缝一致。

### 2.4 训练前正确性检查

#### CPU 数值验证

`code/ls_losses.py::cpu_validation_suite` 在带 NaN 的小型多任务矩阵上覆盖：

- 某任务在 batch 中完全无标签；
- 不同任务有效标签数不同；
- 同一任务同时有正例与负例；
- 不同 batch 大小；
- 所有有效项有限；
- 梯度正确传递；
- 全部任务相同观测覆盖率时的预期等价关系；
- 按实际 batch 大小加权汇总 minibatch 目标可复原完整数据任务平均目标。

结果写入 `analysis/loss_definition_audit.json`。

#### GPU smoke

两组正式训练前必须在各自 GPU pool 跑 preflight；`ls_pair_audit.py` 汇总后
`analysis/paired_initialization.json.all_passed == true` 才允许正式训练。检查：

1. 两组模型初始参数逐 tensor 相同；
2. 两组 AdamW state 完全一致且等于源 checkpoint；
3. 两组初始 Validation AP 一致，并复现 `0.2374902968831333`（`1e-4`）；
4. 两组输入、标签、dropout RNG、训练顺序一致（含 first-batch logits 对比）；
5. M 组 loss 与原历史函数一致；
6. T 组 loss 数值和梯度有限，且匹配独立 float64 参考；
7. 新 loss 对任务权重的改变真实发生（M/T 梯度差 > `1e-8`），不是只改日志；
8. 不访问官方 Test；
9. `PinnedStager` batch/label/loss/grad/update 等价性 smoke 通过。

任一失败即停止正式训练。

### 2.5 固定训练与评价

两组各训练完整 6 epochs。仅在 epoch `0、2、4、6` 评价官方 Validation AP：

- epoch 0 = 加载 checkpoint 后、任何更新前；
- epoch 6 = 主比较点。

主比较固定 epoch 6，**不得**事后改为“各自最佳 epoch”。最佳 Validation
checkpoint 只作为辅助工件保存，不与 epoch 6 结果混为一谈。

每个评价点保存：

- 官方 OGB macro Validation AP；
- 每任务 Validation AP；
- 每任务有效标签数、正例数、负例数；
- Train loss（实际优化目标）；
- 统一口径的原始 masked BCE（日志用，不参与优化）；
- gradient norm 与 clipping 次数；
- GPU 时间、吞吐、显存、非有限数值事件；
- 最终 checkpoint、RNG 状态、AdamW 状态、数据指纹。

### 2.6 输出

`runs/M_micro/`、`runs/T_task_macro/` 各包含 `run.json`、`train_curve.csv`、
`per_task_ap.csv`、`checkpoint_final.pt`、`checkpoint_best.pt`（辅助）、
`selection.json`、`rng_state.pt/json`、`training_order.json`、首步记录。
checkpoint 与预测数组按仓库惯例 gitignore，不提交 Git。

---

## 3. 预先固定的决策规则

定义：

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

报告必须给出原始数值，不得只给标签。额外报告：

- 逐任务 AP 差的中位数；
- T 比 M 改善和退化的任务数；
- 改善是否集中在极少数任务（top-1/3/5/10 正贡献占比）；
- 按 Train 正例数分层的 AP 变化；
- 有效标签覆盖量与改善的 Spearman 描述性相关；
- 两组 Train loss、raw masked BCE、Validation AP 走势。

阈值 `+0.003` 只用于决定是否值得购买下一阶段完整训练，**不是显著性阈值**，
也不是对最终 AP 增益的保证。本轮是单 seed、短程继续训练、已多次使用
Validation 做研究决策的探索结果，不是独立确认。

---

## 4. 资源限制与执行顺序

总预算：**≤ 1.0 A100 GPU-hour**，包含阶段 A、GPU smoke 和阶段 B 两组训练。

| 项 | 软限制 | 硬限制 |
|---|---|---|
| 阶段 A 只读诊断 | 600 s | 720 s（进程内 watchdog） |
| 每组 preflight | 600 s | — |
| 每组正式训练 | 1200 s | 1350 s（优雅保存）；outer `timeout 1500 s` |

执行顺序：

1. 阅读 `AGENT.md` 与 `skills/remote-research-runner/SKILL.md`；
2. 检查 Git worktree、`rr doctor res`；
3. 核验缓存、checkpoint、hash、环境；
4. 本地实现 + CPU 测试 + loss audit；
5. 提交代码，`rr deploy res`；
6. 阶段 A 在**一张**可用 GPU 上只读运行；
7. 阶段 A 门槛通过后，在 `res-gpu0` / `res-gpu1` 跑两组 preflight；
8. 本地 `ls_pair_audit.py` 通过后，两张 GPU 同时运行 M / T；
9. `rr pull` 拉回，运行 `ls_compare.py`，写 `REPORT.md`；
10. 提交代码、分析与报告，停止。

两个正式训练任务使用**同一代码 commit**；正式运行期间不重新部署新代码。
若 GPU 不可用、checkpoint 丢失或资源预算不足，输出阻塞报告；不切换 checkpoint、
不改任务数、不自行加预算。

GPU0 若有其他进程共租，记录实际状态与吞吐差异；不得把共租速度差解释为模型
性能差异。

---

## 5. 实现映射（不是新的科学定义）

| 指令项 | 本目录实现 |
|---|---|
| 模型 | `ls_checkpoint.build_model_from_checkpoint` → 历史 `pc_model.build_c1`；断言 C1、245,184 参数、无 `b_mean`、dropout 0.10、FP32 |
| 数据 | `lr_data.load_task` → pilot 缓存 + 官方 split + 标签工件；拒绝有限 Test 标签 |
| Train-only norm | `lr_data.load_task` 内部重新拟合并断言历史 fingerprint |
| n_t | `ls_losses.task_counts`，只对 `task.labels[task.train_rows]` 计算 |
| M loss | `pc_metrics.masked_bce_loss`（直接调用） |
| T loss | `ls_losses.task_macro_bce_loss`；NaN 先替换后 mask；固定 `n_t` |
| 官方 AP | `pc_metrics.validation_report` → `Evaluator("ogbg-molpcba")` |
| 逐任务 AP/ROC-AUC | `sklearn`，任务缺类时 None |
| 顺序 | `ls_common.make_order_plan`；同一 CPU generator，六 epoch |
| RNG | `ls_common.restore_rng_state` 恢复 Python/NumPy/CPU/CUDA；记录 hash |
| AdamW | `ls_checkpoint.load_optimizer_state`；断言逐 tensor 等于源 state，step=24642 |
| 数据搬运 | `pf_pipeline.PinnedStager(depth=2)` |
| 资源保护 | `ls_common.Watchdog` + epoch 间 soft check；逐 epoch 落盘；原子 checkpoint |
| Test 隔离 | 标签工件 Test 全 NaN；所有加载/推理只索引 train/valid；报告 `test_evaluated=false` |
| 比较 | `ls_compare.py`；固定 epoch 6；逐字执行第 3 节规则 |

---

## 6. 预先声明的局限

- 单 seed、六 epoch 短程继续训练；不能外推为多 seed 或长周期结论。
- 使用 Validation 多次做研究决策；本轮结果不是独立确认。
- 源 checkpoint 的 AdamW moments 与 RNG 起点被恢复，但六 epoch 顺序是新分叉；
  不能宣称与原历史 epoch-10+ 轨迹 bitwise 无缝。
- T 只改变任务间权重，不改变任务内正负样本不平衡。
- `+0.003` 是资源决策门槛，不是统计显著性阈值。
- 官方 Test 从未读取；报告中的 AP 不可与论文 Test AP 直接比较。
- GPU0 共租只影响吞吐，不改变数据、模型、loss 或评价规则；报告分开记录。

---

## 7. 执行记录（运行后填写）

- 正式运行 commit：`98bbf66c9f0893fdeebd2843ad59f894f5af6b7b`；`rr deploy res` 从
  `9b5a61a9fead` 增量部署到该 commit。正式 M/T 使用同一 commit。
- Phase A（只读）：`lcrc-loss-A-diag-20261105-20261105-211307-9ff5486b`，
  pool `res-gpu1`（node `a100-2`），2026-11-05 21:13:13–21:14:19 (+08:00)；
  15/15 门槛通过；`0.016750` GPU-hour。
- Preflight M：`lcrc-loss-preflight-M-20261105-20261105-211726-f4dbe16a`，
  pool `res-gpu0`，21:17:32–21:17:57；`all_passed=true`；`0.005524` GPU-hour。
- Preflight T：`lcrc-loss-preflight-T-20261105-20261105-211751-12a31a19`，
  pool `res-gpu1`，21:17:58–21:18:16；`all_passed=true`；`0.003842` GPU-hour。
- 配对审计：32/32 项通过，`analysis/paired_initialization.json.all_passed=true`。
- 正式 M：`lcrc-loss-M6-20261105-20261105-212242-bf3b9a59`，pool `res-gpu0`，
  21:22:50–21:35:30；`COMPLETED`，6/6 epochs，评价点 0/2/4/6 齐全；
  `0.208506` GPU-hour；epoch-6 AP `0.23722683307799955`。
- 正式 T：`lcrc-loss-T6-20261105-20261105-212307-b1a33e08`，pool `res-gpu1`，
  21:23:15–21:33:15；`COMPLETED`，6/6 epochs，评价点 0/2/4/6 齐全；
  `0.163519` GPU-hour；epoch-6 AP `0.23550376491774172`。
- 总 GPU 时间：`0.398142` A100 GPU-hour（预算 `1.0`）。
- 判决：`D6 = AP_T6 - AP_M6 = -0.001723068160257829 < +0.003`；
  T 更好任务数 42/127；预注册标签 **`SCREEN_NO_SIGNAL`**。
- 偏差记录：正式运行期间 GPU0 有共租进程（约 16.08 GB，100% utilization），
  M 的 step p50 `43.27 ms`、T `33.36 ms`；仅影响吞吐，不作为模型/loss 差异。
  `runs/*/run.json` 的 `gpu_occupancy_start` 在正式运行时为收尾采样，属于日志
  仪表错误；真实起点见 preflight JSON。运行后修正 `ls_train.py` 为启动时采样，
  仅为仪表修复，未重跑、未改写 run.json。无其他协议偏差。
- 没有访问官方 Test；`test_evaluated=false`。没有提前停止、没有 scheduler、
  没有换 checkpoint、没有额外实验。
