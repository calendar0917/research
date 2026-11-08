# PROTOCOL — LCRC-W2 TopRank v1

Round id: `lcrc-pcba-toprank-v1`
Track: `structural-sufficiency`
Status: **frozen before the first formal training step**

---

## 1. 问题与假设

当前 MolPCBA 主线中 Test 表现最好的单模型是历史 **LCRC-W2**
（994,324 参数，best Valid epoch 24，Test AP `0.253616479`）。本轮不新增任何
结构模块，只检验一个纯目标函数问题：

> 在完全相同的 W2 架构、初始化、数据、batch 顺序与优化配方下，把训练目标从
> 纯 masked BCE 改成 **BCE + per-assay hard-negative ranking**，能否改善
> Validation 与官方 Test 的 macro AP？

假设是“正负样本排序”（而不是任务等权、focal、标签重加权）才是 MolPCBA 的
监督目标缺口。若排序项不能带来增益，则说明“单纯换 loss”不足以改善泛化，
不能据此支持“监督目标不匹配”是主要瓶颈。

本轮只训练 **一条** 轨迹，不搜索 lambda，不跑 seed 1，不重训 W2，不训练 S3，
不做 ensemble，不融合历史预测，不使用 S3 结构。

---

## 2. 唯一模型（完全原版 W2）

```text
base            = LCRC-W2
parameters      = 994,324
atom/bond encoder, S1/S2 local, f, rho, four-head attention readout,
128 assay logits = 全部原版，不新增参数
```

实现上直接调用历史只读代码：

* `bs_model.build_model("W2", task.spec, task.norm, 128, seed=0)`；
* 不复制、不改写 forward；不添加模块、残差、归一化或任务参数。

必须从 seed 0 原始初始化重新训练 40 epoch，**不是**在 W2 epoch24 上继续。
新旧模型使用相同的初始化、输入、训练行、batch 顺序、dropout 与优化器；
唯一差异是训练目标新增排序项。

---

## 3. 目标函数

### 3.1 BCE

```python
loss_bce = pc_metrics.masked_bce_loss(logits, y)
```

直接调用历史实现，不重写、不改变 NaN masking 与归一化。

### 3.2 排序项（逐 assay）

对每个 assay `t`：

```text
P_t = batch 内该任务已观测正例（y == 1）
N_t = batch 内该任务已观测负例（y == 0）
```

仅当 `|P_t| > 0` 且 `|N_t| > 0` 时计算：

```python
negative_reference_t = logsumexp(logits[N_t, t]) - log(|N_t|)

rank_loss_t = mean_{i in P_t} softplus(
    negative_reference_t - logits[i, t]
)

loss_rank = mean(rank_loss_t for valid t)
```

无有效任务时：

```python
loss_rank = logits.sum() * 0.0
```

约束：

* 不跨 assay 比较分数；
* NaN 标签既不是正例也不是负例；
* 不补零、不按 Test 正例率加权、不随机挑 hard negatives；
* 不新增 task-specific 可训练参数，不使用外部正例记忆库，不读其它模型预测；
* 不使用 Validation/Test 标签；
* 向量化 `logsumexp` + mask + 归约，复杂度 `O(B × 128)`，不构造全部正负对
  的稠密张量；
* 完全无正例、完全无负例、完全缺失标签的列都返回有限值且梯度精确为零。

**术语：** 这是 *hard-negative ranking surrogate*（log-sum-exp 软最大负例 +
softplus 正例间隔），**不是**精确 Average-Precision loss，也不是 LibAUC 的
单任务 AP loss 直接套到 128 个缺失标签任务上。

### 3.3 组合与系数冻结

```python
loss_total = loss_bce + lambda_rank * loss_rank
```

`lambda_rank` 只估计一次，不搜索。使用历史 W2 best epoch24 checkpoint
（仅用于损失梯度比例，不作为正式初始化），在预先固定的 16 个 **Train**
batches 上：

```text
g_bce  = median(|| d loss_bce  / d logits ||_2)
g_rank = median(|| d loss_rank / d logits ||_2)
lambda_rank = clamp(0.25 * g_bce / g_rank, 1e-4, 0.1)
```

冻结规则：

* 16 个 batch 固定为 epoch 1 的 `floor(k * 2738 / 16)`，`k = 0..15`；
* 记录每个 batch 的 row SHA256 与合并 SHA256；
* 模型 `eval()`（dropout 关闭），只对 detached logits 求梯度，不更新模型；
* 校准阶段不得改变全局 RNG（模型构建之后、校准之前取 RNG 基线）；
* 只读 Train rows；不读 Valid/Test 标签；
* 无效排序 batch 不进入中位数；
* 若 `g_rank` 非正/非有限，或有效 batch 少于 8，停止并报告预检失败；
* 记录实际 lambda、`ratio_raw`、是否边界裁剪及裁剪侧。

---

## 4. 训练协议

与历史 W2 完全相同：

```text
dataset      = ogbg-molpcba
split        = official scaffold
Train/Valid  = 350,343 / 43,793
seed         = 0
epochs       = 40
batch size   = 128
steps/epoch  = 2,738
optimizer    = AdamW
lr peak      = 3e-4
warmup       = 1 epoch (3e-5 -> 3e-4)
cosine final = 1e-5
weight decay = 1e-3
betas        = (0.9, 0.999)
eps          = 1e-8
dropout      = 0.10
grad clip    = 5.0
precision    = FP32
```

冻结 batch order：

```text
实际冻结值 = 06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6
```

说明：任务书写作 `06f43729fa451e449202a990d28f3b8d101e7aec...`，比历史
W2/S3/SP run.json 中记录的值少一个字符 `9`。本轮以历史实际 plan 为准，并与
W2/S3/SP 的 `global_plan_sha256` 逐字相同；若执行为任务书字符串则无法与历史
batch 顺序一致。

每个 epoch 完整评估官方 Validation AP。best checkpoint 在 1–40 epoch 之间按
Validation AP 选择；浮点完全相等时取**较早** epoch（严格 `>` 更新）。

保存：

```text
checkpoint_best.pt
checkpoint_epoch40.pt
checkpoint_last.pt
train_curve.csv
per_task_valid.csv
rank_diagnostics.json
rank_task_coverage.csv
run.json
```

---

## 5. 训练前审计（必须全部通过）

1. W2 参数量严格 994,324。
2. 原始 W2 参数初始状态逐 tensor 与历史相等（state SHA256 + trunk/full
   fingerprint + 与历史构建器逐 tensor 比较）。
3. epoch0 输出与历史 W2 在 `1e-4` 容差内一致（Valid AP 与 Valid BCE）。
4. 只改 loss，不改 forward（模块调用次数：atom/bond/local/f/rho/readout 各 1；
   forward 源码 SHA 记录）。
5. `lambda_rank=0` 时新训练步的 loss/gradient 与历史 BCE 步逐 tensor 精确一致。
6. NaN 标签完全忽略（扰动 NaN 行的 logits 不改变 loss/grad；NaN 行梯度为 0）。
7. 排序只比较同一 assay 的正负分子（跨 assay 扰动不影响其它 assay 的项）。
8. 单任务缺正例/缺负例/全缺失时安全跳过（值有限、梯度精确为 0）。
9. 排序与独立 NumPy 循环参考一致（容差 `1e-6`）。
10. 向正例 logits 加正常数，排序损失不增加。
11. 向负例 logits 加正常数，排序损失不减少。
12. 所有 loss 与梯度有限（含极端 logits）。
13. 16 batch 校准只使用 Train rows（与 Valid/Test 行交集为 0）。
14. 校准不改变正式训练 RNG。
15. 同一 batch order、dropout 及优化器配方（plan SHA、LR 端点、dropout、
    clip、AdamW 参数审计）。
16. checkpoint 可恢复（4 步分支：中途保存/恢复后，model、AdamW moments、
    RNG 与继续训练逐 tensor 一致）。

整个 epoch 记录：

```text
rank-active task count mean/median
no-positive task fraction
positive/negative pair coverage
loss_bce
loss_rank
lambda_rank * loss_rank
gradient norm
```

若排序有效任务覆盖率低，报告必须明确这是该算法在稀疏多任务数据上的局限。

---

## 6. GPU 预算

只运行一条正式训练轨迹，使用 `rr`，优先 `res-gpu1`，不影响其它运行任务。

```text
target <= 2.5 A100 GPU-hours
hard   <= 4.0 A100 GPU-hours
```

先做真实 batch 的 timing/显存预检。若 40 epoch 预算不可行（超过 hard cap），
停止并报告；不缩减数据、不改训练设置。不跑 seed1，不重训 W2，不训练 S3，
不试第二个 loss。

---

## 7. 官方 Test 评价（冻结后读取）

1. 训练完成并保存 `checkpoint_best.pt` 后，重放 Best Valid AP（容差 `1e-4`）。
2. 读取历史 W2 已冻结的 Test AP 与 checkpoint 身份
   （`lcrc_pcba_s3_selective_v1/analysis/test_scores.json`）。
3. 使用与历史相同的官方 Test row ID、FeatureSpec、Train-only normalization。
4. 生成新模型完整 Test logits，保存 `.npy` 与 SHA256；同时用同一路径重算历史
   W2 Test logits，记录与冻结数组的最大绝对差（路径审计）。
5. **先保存预测和 SHA256，再读取官方 Test 标签。** 只有 `tr_test.py score`
   通过继承的 `cc_data.read_official_test_labels` gate 打开
   `raw/graph-label.csv.gz`。
6. 使用官方 `Evaluator("ogbg-molpcba")`，同时用独立 sklearn
   `average_precision_score` 逐任务检查。
7. 逐任务记录 Test AP 与 W2 的差异。

Test 标签不得进入训练、校准、checkpoint 选择或参数更新。

Test 角色标记：

```text
EXPOSED_TEST_PERFORMANCE_BENCHMARK
```

研究负责人可以用其表现决定后续工程方向，但不得称为未经模型搜索影响的盲测
泛化性能。

---

## 8. 判定规则（预注册）

主对比：

```text
W2 Best Valid = 0.265273316
W2 Test AP    = 0.253616479

TopRank Best Valid AP = measured
TopRank Test AP       = measured
```

工程继续信号（资源决策，不是统计显著性）：

```text
TopRank Test AP >= 0.256616
high-support Test mean delta >= +0.002
high-support improved tasks >= 60/108
```

其中 high-support 定义为官方 Test 正例数 `>= 20` 的 assay（历史 108 个）。

判定标签：

| 条件 | 标签 |
|---|---|
| Test target 达到且 Best Valid AP > W2 | `VALID_AND_TEST_GAIN` |
| Best Valid AP > W2，Test target 未达到 | `VALID_ONLY` |
| Test target 达到但 Best Valid AP 未超过 W2 | `EXPOSED_TEST_ONLY_SIGNAL` |
| 两者都未达到 | `NO_CLEAR_GAIN` |

若只有 Valid 提升而 Test 无收益，报告 `VALID_ONLY`。若 Test 提升但 Valid 下降，
报告 `EXPOSED_TEST_ONLY_SIGNAL`，不得声称确认性泛化改进。若两者都没有实质
改善，报告 `NO_CLEAR_GAIN`，直接停止本轮；不自动调 lambda，不新增训练。

不追加消融、超参数搜索、ensemble、新 scaffold split 或第二 seed。

---

## 9. 已知限制（执行前声明）

* Test split 已暴露；本轮 Test 数字是 benchmark，不是盲测。
* 排序项是 per-assay surrogate，不是精确 AP；AP 依赖完整负例排序。
* MolPCBA 每个 batch 只有少数 assay 同时含正负例；排序项覆盖率必须报告。
* 单 seed、单轨迹；不做显著性声明。
* 历史 W2 baseline 的数字来自既有 run，不重训，以冻结 checkpoint SHA 为身份。
* 若排序项覆盖率过低，任何 Test/Valid 差异都不能单独归因于“排序目标”。
