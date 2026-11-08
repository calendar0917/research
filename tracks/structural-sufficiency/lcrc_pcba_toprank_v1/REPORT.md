# REPORT — LCRC-W2 TopRank v1: per-assay hard-negative ranking on the frozen W2 trunk

Round id: `lcrc-pcba-toprank-v1`
Track: `structural-sufficiency`
Protocol: `PROTOCOL.md`（训练前冻结，commit `409766ab`）
Training commit: `21097d8c` · Selection/freeze commit: `1451a88b` · Score/compare commit: `b9f61e78`
Test role: **`EXPOSED_TEST_PERFORMANCE_BENCHMARK`** — MolPCBA Test 不是盲测泛化估计
Decision: **`NO_CLEAR_GAIN`**

---

## 1. 结论（Answer）

| model | parameters | best Valid epoch | **Best Valid AP** | epoch40 Valid AP | **Test AP** | Test BCE | GPU-h | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W2（历史，2026-11-06） | 994,324 | 24 | **0.265273316** | 0.261866918 | **0.253616479** | 0.048583948 | 1.3949 | 124.9 |
| **TopRank（本轮）** | **994,324** | **33** | **0.256844174** | 0.255541366 | **0.250193112** | 0.048729783 | 1.5022 | 134.5 |

| contrast | Validation | Test（已暴露） |
|---|---:|---:|
| TopRank − W2 | **−0.008429142** | **−0.003423367** |
| TopRank − 工作目标 `0.256616` | — | **−0.006422888** |

三个事实：

1. **排序目标没有提升 Validation。** Best Valid AP `0.256844` 比 W2
   `0.265273` 低 `0.00843`（−3.18 % 相对）。这不是某一个坏 epoch：40 个 epoch
   中 TopRank 只有第 1、10 个 epoch 短暂领先，其余 38 个 epoch 都落后；前 20
   个 epoch 平均差 `−0.00348`，后 20 个 epoch 平均差 `−0.00797`，差距随训练
   扩大而不是收敛。
2. **Test 也没有提升。** Test AP `0.250193` 比 W2 `0.253616` 低
   `0.003423`，比预注册目标 `0.256616` 低 `0.006423`。在 108 个 Test 正例
   `>= 20` 的稳定任务上，59 个改善、49 个退化，平均差 `+0.000192`，远低于
   `+0.002` 和 `60/108` 两个条件。整个 Test 的负差主要来自两个低支持任务
   （6 和 4 个正例），不是高支持任务的普遍退化。
3. **模型确实在优化排序 surrogate，但没有转化为目标指标。** 排序损失从
   epoch 1 的 `0.6532` 降到 epoch 40 的 `0.2040`（−68.8 %），但同期的
   Train AP@40 `0.574234` 低于 W2 的 `0.581875`，Train BCE 也略高
   （`0.025257` vs `0.025215`）。所以这不是“排序项完全没被优化”，而是
   “优化了这个 per-batch surrogate，却没有改善 AP”。

按预注册判定表：Validation 没有改善，Test 没有达到 `0.256616`，因此
**`NO_CLEAR_GAIN`**，本轮停止；不搜索 lambda，不跑 seed 1，不重训 W2，不训练
S3，不做 ensemble。

---

## 2. 改了什么：只改目标函数，没有新增结构或参数

```text
TopRank = 原版 LCRC-W2（994,324 参数）
        + masked BCE
        + lambda_rank × per-assay hard-negative ranking loss
```

- 模型完全复用 `bs_model.build_model("W2", task.spec, task.norm, 128, seed=0)`。
- S1/S2、atom/bond encoder、local、f/rho、four-head attention readout、128 个
  assay logits 全部不变。
- 新增可训练参数 **0**；总参数严格 `994,324`。
- 从 seed 0 原始初始化重训 40 epoch，**不是**从 W2 epoch24 继续。
- 数据、Train 行、batch 顺序、dropout、优化器、LR 计划与 W2 相同；唯一差异
  是 `loss_total` 中的排序项。

模型身份审计（`analysis/preflight.json`）：

| property | measured |
|---|---|
| 参数量 | `994,324`（base 848,096 + readout 146,228） |
| 初始 state SHA256 | `6a3df38f00a5a4754c310a82d4ad5f71d7f89e3392f1c914aacc4113a3d6f53b` |
| trunk init fingerprint | `5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d` |
| full trunk init fingerprint | `109756bd24b2ab8589012031e3cf0b0389b0d275227b35044e9381a45c00a5db` |
| 与历史构建器逐 tensor 比较 | 57/57 state tensors，max abs diff `0.0` |
| epoch-0 Valid AP | `0.029063073`（W2 `0.029063332`，差 `2.6e-7`） |
| epoch-0 Valid BCE | `1.939158691`（W2 `1.939158691`，差 `6.0e-10`） |

这些不是形式检查：初始化和 epoch-0 输出都对齐后，后续差异只能来自训练目标，
而不是“换了一个模型”或“换了初始化”。

---

## 3. 排序损失是什么，不是什么

对每个 assay `t`，在一个 batch 内定义：

```text
P_t = {i : y[i,t] == 1}      # 已观测正例
N_t = {i : y[i,t] == 0}      # 已观测负例
```

仅当 `P_t` 和 `N_t` 都非空时：

```python
negative_reference_t = logsumexp(logits[N_t, t]) - log(|N_t|)

rank_loss_t = mean_{i in P_t} softplus(
    negative_reference_t - logits[i, t]
)

loss_rank = mean(rank_loss_t for valid t)
```

没有有效任务时返回 `logits.sum() * 0.0`。实现是向量化 mask + `logsumexp`，
复杂度 `O(B × 128)`，不构造稠密正负对。NaN 标签既不是正例也不是负例：
完全无正例、完全无负例、完全缺失的列都返回有限值，且这些列的梯度精确为零。
单元测试覆盖了 NumPy 循环参考、NaN 扰动、跨 assay 隔离、单侧缺失、全缺失、
正/负 logits 单调性、极端 logits 有限性，以及 `lambda=0` 时与
`pc_metrics.masked_bce_loss` 的逐值/逐梯度一致。

**术语必须准确：** 这是 *per-assay hard-negative ranking surrogate*
（softmax 负例参考 + softplus 正例间隔），**不是**精确 Average-Precision
loss，也不是 LibAUC 的单任务 AP loss 直接套到 128 个缺失标签任务上。AP 依赖
完整负例排序；这里每个 batch 只看到一个子集，而且只有同时含正负例的 assay
参与。

BCE 始终直接调用继承的 `pc_metrics.masked_bce_loss(logits, y)`，没有重写、
没有改 NaN masking、没有改归一化。

---

## 4. lambda 如何冻结：一次校准，不搜索

校准只做一次，使用历史 W2 best epoch24 checkpoint（仅用于估计梯度比例，**不**
用于初始化），在 16 个预先固定的 **Train** batches 上：

```text
g_bce  = median(|| d loss_bce  / d logits ||_2)
g_rank = median(|| d loss_rank / d logits ||_2)
lambda_rank = clamp(0.25 * g_bce / g_rank, 1e-4, 0.1)
```

| item | value |
|---|---:|
| calibration batches | 16（epoch 1 的 `floor(k × 2738 / 16)`） |
| valid ranking batches | 16 / 16 |
| `g_bce` median | `8.381133e-4` |
| `g_rank` median | `3.533112e-2` |
| raw ratio `0.25 × g_bce / g_rank` | `5.930418e-3` |
| **frozen `lambda_rank`** | **`5.930418e-3`** |
| clipped? | **否**（不在 1e-4 / 0.1 边界） |
| calibration file SHA256 | `4c1a17246ae8d3394da33331cc8f3969be913d41d9afc6b9cdefc4b43e260abe` |
| 16-batch row SHA256 | `3b87e305939988488911aa21f039655a40c293343b7b06c5cde83a8b163caa84` |

校准只读 Train rows；不读 Valid/Test 标签，不更新模型；校准前后全局 RNG
哈希不变（`rng_unchanged=true`）。`tr_audit.py preflight` 在 GPU 上
`all_pass=true` 后才写出 `FROZEN` 校准文件，训练器只接受 `FROZEN` 且
`preflight_all_pass=true` 的记录。

> 说明：前两次 GPU 预检把“逐 bit 相等”作为硬条件，在 FP32/CUDA 的
> scatter/reduction 噪声上失败（最大差约 `1e-7`）。正式训练前把这类检查改成
> 记录实测差并使用 `1e-5/1e-6` 容差；损失、模型和预测实现未改。最终预检
> `all_pass=true`。

---

## 5. 训练协议与训练前审计

| item | value |
|---|---|
| host / pool | `res`（`a100-2`）/ `res-gpu1`，`CUDA_VISIBLE_DEVICES=1` |
| GPU / driver / torch | A100-SXM4-40GB / 550.163.01 / 2.5.1+cu124 |
| run id | `toprank-TR1-20261108-163726-6417a4ef` |
| epochs / steps | 40 / 109,520（2,738 × 40） |
| batch plan | `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`（与 W2/S3/SP 相同） |
| optimizer | AdamW，peak lr `3e-4`，1 epoch warmup from `3e-5`，per-step cosine to `1e-5`，wd `1e-3`，β `(0.9, 0.999)`，ε `1e-8` |
| dropout / clip / precision | `0.10` / `5.0` / FP32 |
| clip events / non-finite | 14 / 0 |
| epoch-0 AP | `0.029063073` |
| epoch 40 BCE / Valid AP | `0.026266` / `0.255541` |
| train AP@40 | `0.574234`（W2 `0.581875`） |
| train BCE@40 | `0.025257`（W2 `0.025215`） |
| compute | `1.5022` GPU-h，`134.50` s/epoch，21.82 steps/s，peak 237 MB alloc / 453 MB reserved |

`analysis/preflight.json` 的全部检查通过，核心项：

| check | result |
|---|---|
| 参数量 / 初始 state / trunk fingerprints | 全部等于历史值 |
| 逐 tensor 初始化 | 57/57，max abs diff `0.0` |
| forward 只调用一次 | atom 1 / bond 1 / local 1 / f 1 / rho 1 / readout 1 |
| `lambda=0` 与 BCE | **同一 logits** 上 loss 与 gradient 逐 bit 相等；端到端差 `loss 1.2e-7`、`grad 6.0e-8`、`logits 3.8e-6`（FP32/CUDA） |
| 排序 unit invariant | 全部通过（NumPy 参考、NaN、跨 assay、单侧缺失、单调性、有限性） |
| checkpoint resume | 4 步分支：loss 差 `0.0`，model max `1.23e-7`，AdamW moments max `7.45e-9`，RNG 完全一致 |
| plan / LR / dropout / clip | 与冻结协议一致 |
| Test 标签 | 训练和校准阶段 `test_labels_read=false`，未打开 |

Checkpoints：`checkpoint_best.pt`（epoch 33，
`bfbaa90cdadf5f64601fa767dc8847a14feb8208489170e6e56ca9b94bb4cb4f`）、
`checkpoint_epoch40.pt`（`417d776f83c512d5738dc65f8412e8bb631f02e949f3935d276a4aa1d75811c6`）、
`checkpoint_last.pt`（`05e512cefeba009991de63fd4a880aebac59f0d00133f1ec150627ca76bedef9`）。

`run.json` 的 `git_state.dirty=true` 只来自远端两个**预先存在的未跟踪**文件
（S3 轮的 `architecture_failed.json` / `cost_failed.json`）；tracked tree 正好
在 `21097d8c`，没有 tracked 文件被远端修改。批次计划字符串在任务书中少了一个
字符 `9`；本轮使用历史实际 64 字符值（见 `PROTOCOL.md` §4），否则无法与
W2/S3/SP 的 batch 顺序逐字一致。

---

## 6. 排序覆盖率与优化行为

| metric（40 epoch 平均） | value |
|---|---:|
| 每 batch 有效 assay 数 | `45.31 / 128` = **35.4 %**（median 45） |
| 每 batch 无正例 assay 比例 | **64.35 %** |
| 每 batch 无负例 assay 比例 | 0.94 % |
| 正例覆盖率（active task 中的正例条目占比） | 99.69 % |
| 负例覆盖率（active task 中的负例条目占比） | **37.41 %** |
| implicit pos–neg pairs / batch | 10,502 |
| implicit pairs / positive | 79.9 |
| 每个 epoch 至少激活一次的 assay | 128 / 128 |

这里的“正例/负例覆盖率”是**条目覆盖率**，不是任务覆盖率；任务覆盖率是
`45.31/128 = 35.4 %`。也就是说，一个 batch 里约 64 % 的 assay 根本没有正例，
排序项对它们没有贡献；负例也只覆盖了 batch 内约 37 % 的负例条目。每个 epoch
所有 assay 都会至少激活一次，但梯度信号高度稀疏且偏向正例较多的 assay。
个别 batch 甚至 `active_tasks_min=0`，此时走 `logits.sum()*0` 的安全零损失。

这不是一个“没有生效”的损失：

| epoch | `loss_bce` | `loss_rank` | `lambda × rank` | `λ×rank / BCE` | grad norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.061498 | 0.653236 | 0.003874 | 6.30 % | 0.3115 |
| 20 | 0.029660 | 0.256081 | 0.001519 | 5.12 % | — |
| 40 | 0.026266 | 0.203999 | 0.001210 | 4.61 % | 0.0723 |

排序项确实被优化了（`0.6532 → 0.2040`），但它占总损失的比例很小（校准目标
是把排序梯度设为 BCE 梯度的 25 %，实际损失占比只有约 5–6 %）。同时，
BCE 本身也没有比 W2 更好：Best epoch 的 Valid BCE 为 `0.045946`，W2 为
`0.045774`；Train AP@40 和 Train BCE 也都略差。因此更准确的描述是：
**排序项改变了优化轨迹，但没有改善 likelihood 或 AP，反而略微拖慢了 BCE 的
下降**。单 seed、单 lambda 下不能把原因进一步拆成“lambda 太小”还是
“surrogate 与 AP 不匹配”，本轮也不允许搜索 lambda 来区分。

---

## 7. Validation 结果

| epoch | W2 Valid AP | TopRank Valid AP | Δ（TopRank−W2） |
|---:|---:|---:|---:|
| 1 | 0.095825 | 0.099531 | +0.003706 |
| 8 | 0.233455 | 0.231811 | −0.001644 |
| 16 | 0.256102 | 0.250005 | −0.006096 |
| 24 | **0.265273** | 0.252396 | −0.012877 |
| 33 | 0.263806 | **0.256844** | −0.006962 |
| 40 | 0.261867 | 0.255541 | −0.006326 |

- Best epoch：W2 24，TopRank **33**（无并列；严格 `>` 取较早 epoch）。
- TopRank 只在 epoch 1、10 领先；其余 38 个 epoch W2 更高。
- 前 20 epoch 平均 Δ `−0.00348`，后 20 epoch `−0.00797`；最大差距出现在
  epoch 24（`−0.01288`）。
- Best Valid AP：`0.256844174`，比 W2 低 `0.008429142`。
- epoch40 Valid AP：`0.255541366`，比 W2 的 `0.261866918` 低 `0.006326`。
- 选择规则：只按 epoch 1..40 的官方 Validation AP 取 argmax；没有使用 Test，
  没有使用 epoch40 指标，没有 early stopping。

所以 Validation 的负结果不是“最后一个 epoch 掉点”或“选错 checkpoint”：
TopRank 的整条曲线在 epoch 16 之后持续低于 W2，best 也比 W2 低约 0.0084。

---

## 8. Test 评价：先冻结，再读标签

1. 训练完成后，`tr_compare.py select` 只读 `run.json` 和 Validation 曲线，
   重算 argmax，写出 `analysis/selected_checkpoint.json`（`test_read=false`，
   `selection_uses_test=false`），选出 epoch 33。
2. `toprank-test-freeze`（job `toprank-test-freeze-20261108-181051-c2a80b21`，
   33 s GPU）恢复 epoch33 和 W2 epoch24 两个 checkpoint，重放 Validation AP，
   对 43,793 个官方 Test 行生成 logits，保存 `.npy` 和 SHA256，然后用继承的
   `cc_data.freeze_seal` 封存 manifest。
   - manifest seal：`a4d0e49e4ba57e5008def66d8014462c15656c0883b660e5562c3cc0203ca158`
   - `test_labels_read=false`，`test_labels_read_before_freeze=false`
   - TopRank Test logits：`fa80811c0c42b72858facdc8d911e5b435ad977ac2233f93954f72424dd0ad43`
   - W2 reference（base prediction path）：`184930ab68b0c3bd6105ccf919c0d7ebc0095bc6e4d576cc2caf91f32db85072`
   - Validation replay：TopRank `0.256846551` vs recorded `0.256844174`（差
     `2.4e-6`）；W2 `0.265273145` vs `0.265273316`（差 `1.7e-7`）
3. `tr_test.py score` 在**本地**运行（Test 标签只在本地），先验证 seal、选中
   manifest 的 SHA、每个预测文件的 SHA，再通过
   `cc_data.read_official_test_labels` 打开官方 `graph-label.csv.gz`；它先校验
   官方 Train/Valid 块与冻结标签 artifact 对齐，然后才取 Test 行。
4. 评分同时用 OGB 官方 `Evaluator("ogbg-molpcba")` 和独立 sklearn 逐任务
   `average_precision_score`；两者差 `5.6e-17`。127/128 个任务可评分。

Test 标签统计：43,793 行，3,376,672 个已观测条目，56,698 个正例
（observed rate `1.6791 %`）。

| model | Test AP（official） | Test AP（independent） | Test BCE |
|---|---:|---:|---:|
| W2（frozen reference） | `0.253616479` | `0.253616479` | `0.048583948` |
| **TopRank** | **`0.250193112`** | `0.250193112` | `0.048729783` |

### W2 路径审计

用本轮的 base prediction path 重算历史 W2：

- 与冻结 W2 `.npy` 的最大绝对差 `6.866e-5`，平均 `6.10e-7`；
  99.99 % 的 logits 在 `7.63e-6` 以内；超过 `1e-5` 的条目只有 `0.0062 %`。
- 由重算 logits 得到的 W2 Test AP 为 `0.253615369`，与冻结数组的
  `0.253616479` 差 `1.11e-6`；**W2 的正式分数使用冻结数组**，重算精确复现
  `0.2536164787414065`。
- 这个差异是独立预测过程之间的 FP32 累加/归约顺序差，不是不同的模型、数据或
  特征规范；`analysis/w2_prediction_path_audit.json` 记录了逐项数值。
- 该路径审计**不影响** TopRank 的比较：TopRank 的 logits 在读标签前已经冻结，
  W2 的 per-task AP 来自历史冻结数组。

---

## 9. Test 差异来自哪里

逐任务 Test Δ（TopRank − W2，127 个可评分任务）：

| stratum（Test positives） | tasks | improved | degraded | mean Δ | contribution to macro Δ |
|---|---:|---:|---:|---:|---:|
| ≥ 20 | **108** | **59** | **49** | **+0.000192** | +0.000164 |
| 5–19 | 18 | 10 | 8 | −0.016226 | −0.002300 |
| < 5 | 1 | 0 | 1 | −0.163479 | −0.001287 |
| all | 127 | 69 | 58 | −0.003423 | −0.003423 |

- **稳定任务上基本是平局，不是普遍退化。** 108 个高支持任务 59 上 / 49 下，
  平均 `+0.000192`，中位数 `+0.001070`；二项检验（仅描述）`p=0.387`。这远低于
  预注册的 `+0.002` 和 `60/108`。
- **整个 macro 负差几乎全部来自两个低支持任务。** task 31（6 个 Test 正例）
  Δ `−0.255156`，task 57（4 个正例）Δ `−0.163479`；两者之和
  `−0.418635`，占 127 个任务总差 `−0.434768` 的 **96.3 %**。去掉这两个任务
  后，其余 125 个任务的平均 Δ 为 `−0.000129`，仍不是正收益。
- 低支持 stratum 的 19 个任务整体 `10 上 / 9 下`、平均 `−0.023976`，单个任务
  的 AP 可以跳变 0.16–0.26；这正是不能把 macro 差直接解释为“排序目标伤害了
  稳定任务”的原因。
- 在 Validation 上，按 Valid 正例 `>= 20` 的 107 个任务同样是
  `53 上 / 54 下`、平均 `−0.002795`、中位数 `−0.000183`。稳定任务在两个 split
  上都没有改善。
- 高支持任务不是“被低支持噪声掩盖的正收益”：它的均值只有 `+0.000192`，
  贡献 `+0.000164`，远不足以抵消低支持的 `−0.003587`。

结论：TopRank 没有在稳定任务上带来可复用的 AP 增益；Test macro 的负差主要是
低支持任务的偶然翻转。但由于高支持均值也只是 0，不能说“排序目标改善了稳定
任务，只是被噪声拖累”。

---

## 10. Valid → Test 与判定

| model | Best Valid | Test | Valid→Test drop |
|---|---:|---:|---:|
| W2 | 0.265273 | 0.253616 | −0.011657 |
| TopRank | 0.256844 | 0.250193 | **−0.006651** |

TopRank 的 Valid→Test 落差比 W2 小（`−0.00665` vs `−0.01166`），但这是因为
两个绝对值都更低，不能解释为更好的泛化。逐任务 Valid Δ 与 Test Δ 的相关性：
Pearson `r=0.199`（`p=0.025`），Spearman `ρ=0.169`（`p=0.057`），同号
77/127；方向是弱正相关，不是“Valid 升、Test 降”的反转。

预注册判定：

```text
Validation gain  −0.008429142  <  0        -> 未改善
Test gain        −0.003423367  <  +0.002    -> 未改善
Test AP            0.250193112  <  0.256616 -> 未达目标
high-support mean  +0.000192    <  +0.002   -> 未达
high-support count  59 / 108    <  60 / 108 -> 未达
label: NO_CLEAR_GAIN
```

- 不采用 TopRank；W2 仍是当前主线中 Test 最好的单模型。
- 本轮不授权后续：不调 lambda，不跑 seed 1，不重训 W2，不训练 S3，不添加
  第二个排序 loss，不做 ensemble，不根据 Test 结果改 recipe。
- 如果未来还要继续“监督目标”方向，必须先解决本轮暴露的覆盖率问题（每 batch
  64 % assay 无正例、负例覆盖 37 %），而不是在这个 surrogate 上继续扫 lambda；
  但这不是本轮授权的工作。

---

## 11. 成本、provenance 与 artifacts

### 成本

| item | value |
|---|---:|
| formal training | `1.5022` GPU-h |
| final preflight | `0.0214` GPU-h（77 s） |
| Test freeze | `0.0092` GPU-h（33 s） |
| **operational total** | **`1.5328` GPU-h** |
| 含两次被过严 bitwise 检查拒绝的预检 | `1.5758` GPU-h |
| budget target / hard cap | `2.5` / `4.0` GPU-h |
| Test score（本地 CPU） | 17 s |
| W2 training（历史） | `1.3949` GPU-h，`124.9` s/epoch |
| 额外训练时间 | `+0.1073` GPU-h（`1.077×`） |

正式训练在 target 和 hard cap 内；排序项带来的额外计算很小（主要成本仍是
W2 主干）。

### Provenance

| step | job / commit | wall |
|---|---|---|
| protocol + code | `409766ab` | — |
| audit tolerance fixes | `3b60a673`, `b13fce25` | — |
| GPU preflight（final） | `toprank-audit-20261108-163144-04de480e` / `b13fce25` | 77 s |
| calibration + training | `21097d8c` / `toprank-TR1-20261108-163726-6417a4ef` | 1 h 30 min |
| Validation selection | `1451a88b` | — |
| Test freeze | `1451a88b` / `toprank-test-freeze-20261108-181051-c2a80b21` | 33 s |
| Test score + compare | `b9f61e78`（本地） | 17 s |

`run.json` 的 tracked tree 是 `21097d8c`；`git_state.dirty=true` 仅因远端两个
S3 轮遗留的未跟踪失败 JSON，和本轮无关。

### Artifacts

```text
analysis/preflight.json                 全部预检检查、timing、RNG、resume
analysis/calibration.json               FROZEN lambda / g_bce / g_rank / batch hash
analysis/selected_checkpoint.json       Validation-only 选择（test_read=false）
analysis/test_prediction_manifest.json  seal + 两个模型的 Test logits SHA
analysis/test_scores.json               官方/独立 Test AP、逐任务 AP、strata
analysis/per_task_valid_test.csv        逐任务 Valid/Test AP 与 Δ
analysis/comparison.json                W2 vs TopRank、成本、strata、decision
analysis/decision.json                  NO_CLEAR_GAIN + 预注册阈值
analysis/per_task_test_deltas.csv       逐任务 Test Δ
analysis/w2_prediction_path_audit.json  W2 重算 vs 冻结数组的路径审计
analysis/report_numbers.json            本报告使用的曲线/覆盖/统计数值
analysis/test_predictions/*.npy         冻结 Test logits（git-ignored）
runs/TR1/                               checkpoints、run.json、曲线、覆盖率（git-ignored）
```

Test logits 和 checkpoints 保持本地 / git-ignored；`analysis/` 下的 JSON 和 CSV
是证据。所有 Test 数字的角色都是
**`EXPOSED_TEST_PERFORMANCE_BENCHMARK`**，不是盲测泛化性能。
