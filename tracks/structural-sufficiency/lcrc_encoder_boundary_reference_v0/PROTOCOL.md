# PROTOCOL — LCRC 编码器能力边界对照（lcrc-encoder-boundary-reference-v0）

- Round：`tracks/structural-sufficiency/lcrc_encoder_boundary_reference_v0/`
- 协议 id：`lcrc-encoder-boundary-reference-v0`
- 本文件在任何新训练之前写定。训练开始后只允许修实现并记录到 REPORT 的工程日志段；
  不允许改模型输入、层数/宽度、预算口径、评价口径或判定阈值。
- 数据：只使用官方 `ogbg-mollipo` scaffold split。Train 3360 / Valid 420。**Test 不参与本轮**：
  本轮代码只构造/评价 Train 与 Valid，不计算任何 Test 指标，选择只用 Valid。
- seed 0；最多两个新训练单元；GPU·h 硬上限 0.6；整体墙钟硬上限 60 min；排队 ≤ 5 min。

## 1. 问题与目的

FIXED1 的局部计算扩展、角色分离和结构 dropout 均未取得持续性能推进。本轮停止
FIXED1 局部计算改进，只回答一个能力边界问题：

> 在共享输入信息来源、匹配参数预算、相同图级读出和相同训练条件下，传统 GINE
> 编码器（4 层迭代消息传递）是否仍比固定来源结构编码器（CSC v0 / FIXED1）具有
> 更好的官方 Valid 预测性能？

GINE 仅作能力边界参考，不是本项目的新主模型。

## 2. 两个编码器

### A：FIXED1（只读复用，不重训）

- 完全复用 `context_conditioned_structural_computation_v0`（CSC v0）的 mollipo A
  best 状态：`runs/mollipo_A_seed0/checkpoint_best.pt`（SHA-256 与 `run.json` 一致）。
- 架构：`BSModel` 1× 宽度（d=64）+ LocalContent + 两个 `MessageFFN` 阶段（FIXED1
  `Q=0`）；总参数 **294,135**。
- 配方：seed 0、360 epochs、batch 128、AdamW 3e-4→1e-5 per-step cosine、wd 1e-3、
  clip 5、dropout 0.10、标准化目标 MSE、同一 Train-only 目标标准化、同一 batch
  顺序计划（plan sha256 `f9c1398c…`）、按最小 Valid RMSE 选 checkpoint。
- 复用条件核对：checkpoint、Train/Valid 预测、`train_curve.csv`、`run.json` 与
  `training_order.json` 均存在且自洽；`run.json` 的 `best_epoch=140`、
  Valid RMSE 0.726678… 与 `best_checkpoint_eval.json` 一致。因此**满足本轮要求，
  只读复用，不重训**。

### B：GINE_ENCODER（本轮唯一新训练单元）

- 基础实现：`tracks/structural-query-learning/code/sq_gine.py` 的真实 GINE 栈
  （`GINEConv` + MLP + BatchNorm + SiLU + dropout 0.10 + residual，`train_eps=True`）。
- 输入：同一真实 OGB 分子图与原始 OGB 原子/键属性表
  `ogbg_mollipo_sq-pilot-cache-v1.npz`（SHA-256 `fe002287…`）：9 个原始原子离散列
  （atomic_num, chirality, degree, formal_charge, num_hs, num_radical_electrons,
  hybridization, is_aromatic, is_in_ring）+ degree embedding + min-ring embedding，
  3 个原始键列（bond_type, bond_stereo, bond_is_in_ring）。不增加外部化学数据。
- 结构范围：4 层传统迭代消息传递（每层真实有向键），输出 **64 维/原子**；
  128 维隐藏表示经一个 `Linear(128→64)` 投影到共享读出的宽度。
- 层数/宽度的确定规则（训练前冻结，不看性能）：沿用已有实现的 hidden=128；
  取“总参数量落在 A 的 ±15% 内”的最大层数 L∈{5,4,3,…}。L=5 时总参数 ≈390.7k
  （+32.8%，超预算），L=4 时 ≈332.7k（+13.1%，在预算内）→ **L=4，hidden=128，
  out=64，总参数约 332.7k**（以构建后实测为准，必须落在 [250,015, 338,255]）。
- 图级预测：完全复用 A 的 `post_local`（`f`、`rho`、四头 attention readout、全局
  descriptor、`global_mean/global_scale` buffer）。不使用 GINE 自己的预测头。
  实现上直接调用 `BSModel.post_local`（未复制公式），head 模块与 buffer 从
  `BM.build_model(scale=1, seed=0)` 的参考模型原样取出，因而与 A 的 head 初始化
  **逐位相同**（check 校验）。
- 初始化：head 与 A 相同；GINE 编码器使用 `torch.manual_seed(0)` 的默认 PyTorch
  初始化。不同编码器无法逐参数匹配初始化，本轮**不**声明完全配对初始化。

### 可对照性判定（训练前）

同时满足以下条件才允许把结果称为同一预算下的编码器对照：

1. 同一官方 split、同一分子集合；
2. 两个编码器只消费同一原始 OGB 属性表与同一真实图，不引入外部化学数据；
3. 图级读出是同一个 `post_local` 实现与同一初始化；
4. B 总参数量在 A 的 ±15% 内。

以上在 `analysis/checks.json` 中逐项验证。若任何一项失败 → 报告
`REFERENCE_NOT_COMPARABLE` 并停止。注意：这不是“只差一个算子”的严格消融；两个
编码器消费的原始列子集与派生结构不同，这一点写入 REPORT，不把差异归因于单一操作。

## 3. 训练（继承 CSC v0 配方，逐项冻结）

- 360 epochs、batch 128、AdamW（betas 0.9/0.999、eps 1e-8）、peak LR 3e-4、
  per-step cosine 至 1e-5（无 warmup）、wd 1e-3、dropout 0.10、grad clip 5、FP32。
- 相同 Train-only 目标标准化（mean 2.1621607144740187、std 1.211539840098397）；
  标准化 MSE；同一 batch 顺序计划（与 A 的 `plan_sha256` 必须相同）。
- 全部参数端到端训练（一个参数组）。
- 每 epoch 全量评价 Train/Valid RMSE、MAE（不评价 Test）。
- checkpoint：最小 Valid RMSE（严格更小才替换，平局取更早 epoch）。
- 计时预检：前 2 个 epoch 后按实测 epoch 时间外推 360 epochs；若外推超过
  1500 s（GPU·h 0.417）或硬预算 0.6 GPU·h，则停止并记 `TIMING_PRECHECK_FAIL`，
  不临时缩短训练长度。

## 4. 分析（只在 Train/Valid）

- 主表：参数量、计算成本（训练/评价秒、GPU·h、推理吞吐）、best/endpoint/last-20 的
  Train/Valid RMSE 与 MAE、best epoch。
- 配对误差：同一分子（official row id）上的 `dse = (pred−y)^2`；Δdse = dse_B − dse_A；
  ΔRMSE = RMSE_B − RMSE_A；改善/退化分子比例。
- 分子级 bootstrap：Valid 分子有放回重抽样 2000 次（seed 20261126），ΔRMSE 的
  95% 百分位区间；同一组重抽样同时用于相对差。
- 敏感性：去掉 A 的 Valid 平方误差最大的 5%（21 个分子）后重算两者 RMSE。
- 逐分子误差分布：Valid 的 |error| 分位数、直方图（图与 JSON）。
- 一项轻量诊断（训练后、用已训练模型，仅探索解释）：在 Train 上冻结阈值
  （各结构描述量的 Train 中位数；局部环境稀有度用 Train 上的原子签名频率），
  比较“B 优于 A”与“其余”的分子在
  （i）原子数、（ii）图直径 `diameter_max_finite`、（iii）环原子比例、
  （iv）局部环境稀有度（原子签名 `(atomic_num, degree, min_ring)` 的 Train 频率的
  −log 均值）上的分布，并报告高/低组的 ΔRMSE。不根据结果寻找新阈值。

## 5. 判定（预注册）

记 `rel = (RMSE_B − RMSE_A)/RMSE_A`（Valid，best checkpoint）。B 更好时 rel<0。

- `ENCODER_ADVANTAGE_SIGNAL`：B 相对 A 的 Valid RMSE 改善 ≥2%（rel ≤ −0.02），
  且 ΔRMSE 的分子 bootstrap 95% CI 完全 <0。
- `NO_CLEAR_ENCODER_ADVANTAGE`：|rel| < 2%，或 CI 含 0（且不满足上一条）。
- `INCONCLUSIVE`：配方/输入/预算/结果不支持明确判断（含数据或实现不一致）。
- `REFERENCE_NOT_COMPARABLE`：§2 可对照性四项任一失败，停止分析。

即使出现 `ENCODER_ADVANTAGE_SIGNAL`，也只表明该编码器整体预测能力更好，不能证明
迭代消息传递、表示范围或某个具体结构操作是唯一原因。

## 6. 交付

`PROTOCOL.md`、`configs/experiment.json`、`code/`、`runs/gine_B_seed0/`（曲线、
best checkpoint、预测、run.json、SHA-256 manifest）、`analysis/`（checks、主表、
配对比较、bootstrap、逐分子误差、诊断、图）、中文 `REPORT.md`。
不设计新模型、不追加数据集、不启动下一轮。
