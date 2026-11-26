# LCRC 编码器能力边界对照 — REPORT

- Round：`tracks/structural-sufficiency/lcrc_encoder_boundary_reference_v0/`
- 协议：`PROTOCOL.md`（训练前冻结）；协议 id `lcrc-encoder-boundary-reference-v0`
- 数据：官方 `ogbg-mollipo` scaffold split，Train 3360 / Valid 420。**Test 未使用**：
  本轮代码不构造 Test batch、不评分、不报告任何 Test 指标（`test_scored=false`，
  `splits_used=[train, valid]`）。
- seed 0；新训练单元 1 个（B）；A 只读复用 CSC v0 的完整 360-epoch 运行。
- 实际资源：B 训练 **485.2 s = 0.1348 GPU·h**（硬上限 0.6）；整体墙钟约 38 min
  （硬上限 60 min，含排队与一次 deploy 排障）。运行环境 `res` / A100-SXM4-40GB / driver 550.163.01 /
  torch 2.5.1+cu124。

**一句话判读**：在共享 `post_local`、匹配参数预算（+13.1%）和相同配方下，GINE
的 Valid RMSE 点估计比 FIXED1 低 **6.69%**（0.6781 vs 0.7267），但按预注册判据
**未达到** `ENCODER_ADVANTAGE_SIGNAL`：配对分子 bootstrap 的 ΔRMSE 95% CI 为
[−0.0994, +0.0073]，包含 0；且去掉 A 误差最大的 5% Valid 分子后，B 反而差 4.9%。
因此判定 **`NO_CLEAR_ENCODER_ADVANTAGE`** —— 这不是“差距消失”，而是“点估计优势
真实存在、但不足以在 420 个 Valid 分子上稳健成立”。

---

## 1. A 的只读复用核对

`context_conditioned_structural_computation_v0/runs/mollipo_A_seed0` 满足本轮要求：

| 项 | 值 |
| --- | --- |
| checkpoint | `checkpoint_best.pt`，epoch 140，SHA-256 `52722a1c…` |
| 配方 | seed 0、360 epochs、batch 128、AdamW 3e-4→1e-5、wd 1e-3、clip 5、dropout 0.10、Train-only 标准化 MSE、最小 Valid RMSE 选择 |
| batch 顺序 | `plan_sha256=f9c1398c…`，与本轮 `D.batch_order_plan` 逐位相同 |
| 初始化可复现性 | 用当前代码重建 CSC A，`base_state_sha256=539e0dbe…` 与 A `run.json` 记录逐位相同 |
| 预测可复算 | 从 `predictions_best.npz` 重算 Train/Valid RMSE/MAE，与 `run.json` 相差 <1e-12 |

因此 A **未重训**，只读复用（其历史运行每 epoch 也评分过 Test，但本轮不读取、
不使用任何 Test 数组）。

## 2. B 的构造与对照条件

B 复用仓库真实 GINE 栈（`structural-query-learning/code/sq_gine.py`）：
9 个原始原子离散列 embedding + degree + min-ring embedding，3 个原始键列
embedding，4 层 `GINEConv`（MLP 128→256→128、SiLU、BatchNorm、dropout 0.10、
residual、`train_eps=True`），128 维隐藏表示经 `Linear(128→64)` 输出 64 维/原子，
随后**原样调用** `BSModel.post_local`（`f`、`rho`、四头 attention readout、
`global_mean/global_scale` buffer、全局 descriptor），不使用 GINE 自己的预测头。

层数规则在训练前冻结：沿用已有实现的 hidden=128，取使总参数量落在 A ±15% 内的
最大层数。L=5 → 约 390.7k（+32.8%，超预算）；L=4 → **332,731（+13.1%，在预算内）**。
不做任何性能搜索。

| 对照条件 | 结果 |
| --- | --- |
| 同一官方 split / 同一分子集合 | 是（Train 3360 / Valid 420，逐分子 row id 对齐） |
| 同一原始属性表 + 同一真实图，无外部化学数据 | 是（sq-cache SHA-256 `fe002287…`） |
| 同一 `post_local` 实现与初始化 | 是；B 的 head init 指纹 `6f9290cb…` 与 A 的**记录初值**逐位相同 |
| B 总参数在 A ±15% 内 | 是（332,731 vs 294,135，+13.12%；预算 [250,014, 338,255]） |

**参数量与结构范围**

| 模型 | 编码器参数 | 共享 head 参数 | 总参数 | 结构范围 |
| --- | --- | --- | --- | --- |
| A FIXED1 | 247,680 | 46,455 | 294,135 | LocalContent（S1/S2 局部内容，含 2-hop 路径统计、环/键类分数）+ 2 个固定来源消息阶段 |
| B GINE | 286,276 | 46,455 | 332,731 | 4 层真实键迭代消息传递（可达 4 跳），无显式 S1/S2 2-hop 统计 |

两个编码器消费同一原始 OGB 属性表的不同子集/派生：A 额外使用 S1/S2 派生量
（`log1p_n_paths2`、`closure_ratio`、incident bond-class fraction 等），B 额外使用
原始 `chirality`、`num_radical_electrons`、`hybridization`、`bond_stereo`。因此本轮
是**匹配预算的编码器级对照**（`MATCHED_BUDGET_ENCODER_REFERENCE`），不是“只差一个
算子”的严格消融；不能把差异单独归因于迭代消息传递。

不同编码器无法逐参数匹配初始化：head 配对初始化已验证；GINE 编码器用
`torch.manual_seed(0)` 的默认初始化，**不声明**编码器层面的完全配对初始化。

## 3. 训练

360 epochs 跑满（无早停）；timing precheck 在 epoch 2 后外推 654.5 s，低于
1500 s 硬限，继续训练。所有参数端到端训练；每 epoch 全量评价 Train/Valid；
checkpoint 只看最小 Valid RMSE（严格更小才替换）。

| 模型 | best ep | best Train RMSE | best Valid RMSE | best Valid MAE | endpoint Valid RMSE/MAE | last-20 Valid RMSE/MAE |
| --- | --- | --- | --- | --- | --- | --- |
| A FIXED1 | 140 | 0.3080 | 0.7267 | 0.4984 | 0.7415 / 0.5061 | 0.7407 / 0.5062 |
| B GINE | 202 | 0.2326 | **0.6781** | **0.4873** | 0.6950 / 0.4902 | 0.6963 / 0.4911 |
| Δ (B−A) | — | −0.0754 | **−0.0486（−6.69%）** | −0.0111（−2.23%） | −0.0465（−6.27%） | −0.0444（−6.00%） |

best 与 last-20 方向一致，endpoint 也未反转；Train–Valid gap 两边接近
（A 0.4187、B 0.4455），B 的优势不是只靠“拟合得更狠”。训练时间几乎相同
（A 390 s / B 399 s；A 的历史 GPU·h 0.167 不计入本轮，B 本轮 0.135 GPU·h），B 的评价更快（A 三 split 197.9 s；B 两 split 80.6 s，评价更快主要因为本轮少评一个 split）。

完整主表：`analysis/main_table.csv`；曲线：`analysis/curves.csv`、
`analysis/figures/curves_and_errors.png`。

## 4. 配对误差分析（Valid，best checkpoint）

Δdse = dse_B − dse_A（同一分子；负值 = GINE 更好）。2000 次分子有放回 bootstrap
（seed 20261126）。

| 量 | Valid | Train |
| --- | --- | --- |
| ΔRMSE (B−A) | −0.0486（−6.69%） | −0.0754（−24.47%） |
| 95% CI（分子 bootstrap） | **[−0.0994, +0.0073]**（含 0） | [−0.0843, −0.0671] |
| P(B 更好)（bootstrap 重抽样） | 0.9555 | 1.0000 |
| B 更好的分子比例 | **50.7%** | 60.7% |
| 去掉 A 误差最大 5% 后 ΔRMSE | **+0.0252（+4.88%，B 更差）** | −0.0464（−18.57%） |

逐分子 |error| 分位数（Valid）：

| 分位 | 50% | 75% | 90% | 95% | 99% |
| --- | --- | --- | --- | --- | --- |
| A | 0.3794 | 0.6600 | 0.9858 | 1.3128 | 1.9698 |
| B | 0.3793 | 0.6568 | **1.0411** | 1.3044 | **1.8322** |

读法：B 明显压缩了 A 的极端误差（99% 分位 1.97→1.83），但在中高分位（90%）更差；
Valid 上 B 只在 50.7% 的分子上更好。官方 RMSE 由大误差分子主导，所以总 RMSE 的
−6.69% 主要来自少数 A 的灾难性分子；把 A 最差的 21 个 Valid 分子（5%）去掉后，
B 在剩余 399 个分子上反而差 4.9%。这与 bootstrap CI 跨 0 一致。

逐分子明细：`analysis/paired_errors.csv`；配对统计与敏感性：
`analysis/paired_comparisons.json`；散点与误差分布：
`analysis/figures/paired_error_scatter.png`、`analysis/figures/curves_and_errors.png`。

## 5. 轻量结构诊断（仅探索解释）

在已训练模型上，使用 Valid 分子，把“B 的平方误差更小”与其余分子分组；阈值全部
取 Train 中位数（未根据结果搜索）。原子局部环境签名 =
`(atomic_num, degree, min_ring)`，稀有度 = 原子签名 Train 频率的 −log 概率的分子均值。

| 描述量 | B 更好组（均值 / 中位） | 其余组（均值 / 中位） | 高组 ΔRMSE | 低组 ΔRMSE |
| --- | --- | --- | --- | --- |
| 原子数 | 27.69 / 27 | 26.90 / 26 | −0.0818 (n=194) | −0.0123 (n=226) |
| 图直径 | 14.23 / 14 | 13.46 / 13 | −0.0724 (n=178) | −0.0276 (n=242) |
| 环原子比例 | 0.7256 / 0.72 | 0.7188 / 0.72 | −0.0266 (n=244) | −0.0738 (n=176) |
| 局部环境稀有度 | 2.524 / 2.474 | 2.589 / 2.517 | −0.0429 (n=230) | −0.0578 (n=190) |

方向上，B 更好的分子略大、图直径略大，局部环境并不更“少见”；高原子数/高直径组
的 B 优势更大。这个诊断与“B 压缩大分子的极端误差”一致，但**不替代官方 RMSE**，
也不构成机制证明。完整数值：`analysis/diagnostic_groups.json`。

## 6. 判读

按 `analysis/verdict.json` 的预注册规则：

- 观察到的 Valid 相对差：**−6.69%**（超过 2% 参考）；
- 但 ΔRMSE 的分子 bootstrap 95% CI **包含 0**（上界 +0.0073）；
- 敏感性：去掉 A 最差 5% 分子后优势翻号；B 只在 50.7% 的 Valid 分子上更好。

→ **`NO_CLEAR_ENCODER_ADVANTAGE`**。这里必须区分两件事：

1. **点估计层面**：GINE 在共享 head/匹配预算/同配方下确实取得了更低的官方
   Valid RMSE（−6.69%）和 MAE（−2.23%），Train 上差距更大（−24.5%），且 best、
   endpoint、last-20 方向一致。这不是“差距消失”。
2. **判定层面**：该优势在 420 个 Valid 分子上不稳健（CI 跨 0、drop-top-5% 翻号、
   分子级胜率仅 50.7%），因此不能按预注册标准升级为 `ENCODER_ADVANTAGE_SIGNAL`。

即使判为有优势，也只能说明“该 GINE 编码器整体预测能力更好”，不能证明迭代消息
传递、表示范围或某个具体结构操作是唯一原因——两个编码器的派生输入、结构范围和
参数分布都不同。

### 对历史 GINE 的复核（NO_CLEAR 分支要求的检查）

| 历史参照 | 预测头 | 层数/宽度 | 参数 | 配方差异 | Valid best |
| --- | --- | --- | --- | --- | --- |
| `sq-pilot-small-v1` G (3e-4) | 自己的 `concat(sum,mean,log1p n)→257→128→1` | 5 层 / 128 | 377,350 | wd 1e-5、5-epoch warmup | 0.6309 (ep165) |
| 同上 (1e-3) | 同上 | 同上 | 377,350 | 同上，peak 1e-3 | 0.6091 (ep168) |
| 本轮 B | **A 的 `post_local`** | 4 层 / 128→64 | 332,731 | A 的 3e-4、wd 1e-3、无 warmup | 0.6781 (ep202) |

历史 GINE 的 Valid 明显更低（0.609–0.631 vs 0.678），但它使用了不同的预测头、
更强的参数预算（+28% vs A）、不同的 wd/warmup 和 5 层深度；它不是本轮严格匹配
对照，不能把 0.609→0.678 的差距全部归给编码器。这里只记录“预测头/训练配方/容量
差异”确实是历史优势的一部分来源；本轮不追加实验去拆分它们。

## 7. 正确性、工程记录与交付

- **Test 边界**：继承的 bundle 构建器会为缓存/身份校验在内存中构建 Test split；
  本轮代码不把它传给任何模型、不构造 Test batch、不评分、不产生任何 Test 分析量。
  本轮新增 run 的 `run.json` 记录 `test_scored=false`、`splits_used=[train,valid]`。
- **训练前检查 23/23 通过**（`analysis/checks.json`）：split/缓存身份、Train-only
  scaler、sq-cache 字节 SHA、逐分子原子/键/SMILES 对齐、GINE 词表与历史一致、
  batch plan 与 A 逐位相同、A 参数量/预测复算、A 初始化指纹复现、B head init
  与 A 初值一致、参数预算、前向形状/有限性、`post_local` 逐位复用、同分子同 logit、
  Test 未使用。
- **训练后独立复算**（`analysis/verification.json`）：从 B `checkpoint_best.pt`
  重新加载并在 CPU 上评分，Train/Valid RMSE/MAE 与 `run.json` 的最大差为
  5.8e-9；与保存预测的最大差 2.0e-6（GPU/CPU float32 归约顺序差异），全部通过。
- **Checkpoint manifest**：`analysis/checkpoint_manifest.json`（A/B best/final 的
  SHA-256、epoch、参数分组、预测与曲线 hash）。
- **工程偏差（如实记录）**：`res-2` 的 Slurm 队列等待约 11 min，超过“排队 ≤5 min”
  的资源指引；期间 `res` 的 deploy 因上一轮遗留的未跟踪结果文件冲突而失败。把
  那些未跟踪文件移到 checkout 外的备份目录
  （`/home/hxy/untracked_backup_ebr_20261126T102319Z`；内容仍在新 commit 中）后，
  `rr deploy res` 成功，B 在 `res`/`res-gpu0` 以 process backend 完成训练，并取消了
  仍在排队的 Slurm 作业。整体墙钟仍在 60 min 内，GPU 消耗 0.1348 h。
- **未做**：Test 评分、PCBA/PCQM4Mv2、蒸馏/集成/超参搜索/额外监督、下一轮模型设计。
- **关键文件**：`PROTOCOL.md`、`configs/experiment.json`、
  `code/{eb_common,eb_data,eb_model,eb_train,eb_checks,eb_analyze,eb_verify}.py`、
  `analysis/{checks,main_table,paired_comparisons,paired_errors,diagnostic_groups,verdict,verification,checkpoint_manifest}`、
  `analysis/figures/`、`runs/gine_B_seed0/`。
