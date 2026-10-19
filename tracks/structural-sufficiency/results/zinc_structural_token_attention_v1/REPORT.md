# REPORT — zinc structural-token attention v1

Protocol: `zinc-structural-token-attention-v1`
Round: `tracks/structural-sufficiency/results/zinc_structural_token_attention_v1/`
Status: complete; one official-valid label read after all six predictions and the
mechanism record were frozen.

## 0. 一句话结论

**本轮没有检测到 object-token attention 相对 P3 的实际增益。** P3 seed-mean valid
MAE 是 0.079844；S / T / R 分别是 0.084673 / 0.089460 / 0.093237。三个 arm 的点估计
都更差；T 和 R 相对 P3 的配对 95% CI 整个落在 0 以下。R 的结构关系偏置没有表现出
相对 T 的可靠优势。按预注册决策规则，关闭这个 object-token attention 架构筛选，
不通过加层、加头、加关系桶或加种子来救援。

这不是“object attention 等价于 P3”，也不是“结构关系无用”的一般性结论；这是在本轮
冻结的 P3 基底、R3 recipe、两个种子和 4.0 A100-hour 预算下的**未检测到增益**。
官方 test 从未加载。

## 1. 直接回答四个问题

| 问题 | 回答 | 证据 |
|---|---|---|
| 比 P3 更好吗？ | **没有。** | S/T/R 的 seed-mean MAE 都高于 P3；T、R 的 paired CI 上界 < 0。 |
| 比 S 更好吗？ | **没有。** | T vs S mean gain −0.004786（CI 跨 0）；R vs S mean gain −0.008563（CI 上界 < 0）。 |
| 结构偏置比普通 object attention 更好吗？ | **没有检测到。** | R vs T mean gain −0.003777，95% CI [−0.012779, +0.003761]；点估计方向为负。 |
| 值得额外代价吗？ | **不值得。** | R 每个 run 的 wall time 约为 S 的 2.8 倍；T/R 的 analytic forward FLOPs 约为 S 的 1.24 倍；没有换来精度。 |

## 2. 主结果

指标：official PyG ZINC `subset=True` valid，1000 个分子；raw-y MAE；先按 seed 计算，
再对两个 seed 取平均。没有 prediction ensemble，没有 valid-based selection。

| model | seed 0 MAE | seed 1 MAE | seed mean MAE | seed mean RMSE |
|---|---:|---:|---:|---:|
| P3 (frozen reference) | 0.07910150 | 0.08058640 | **0.07984395** | — |
| S (capacity/readout control) | 0.08649923 | 0.08284775 | 0.08467349 | 0.314318 |
| T (ordinary object attention) | 0.08974954 | 0.08916967 | 0.08945960 | 0.268714 |
| R (T + structural bias) | 0.09087395 | 0.09559912 | 0.09323653 | 0.336036 |

`gain = MAE(reference) − MAE(candidate)`；正数才表示 candidate 更好。

| contrast | seed 0 gain | seed 1 gain | mean gain | paired 95% CI | ordinary (y>−3) | drop P3 worst-10 | without row 172 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S vs P3 | −0.007398 | −0.002261 | −0.004830 | [−0.013840, +0.002222] | −0.001147 | −0.003652 | −0.001381 |
| T vs P3 | −0.010648 | −0.008583 | −0.009616 | [−0.015817, −0.004120] | −0.008561 | −0.010892 | −0.009242 |
| R vs P3 | −0.011772 | −0.015013 | −0.013393 | [−0.022471, −0.006382] | −0.008628 | −0.009550 | −0.010044 |
| T vs S | −0.003250 | −0.006322 | −0.004786 | [−0.013097, +0.004172] | −0.007414 | −0.007240 | −0.007862 |
| R vs S | −0.004375 | −0.012751 | −0.008563 | [−0.014524, −0.003494] | −0.007481 | −0.005898 | −0.008663 |
| R vs T | −0.001124 | −0.006429 | −0.003777 | [−0.012779, +0.003761] | −0.000067 | +0.001341 | −0.000801 |

Bootstrap: 5000 resamples of molecules, seed 20261019, two seeds resampled synchronously.
CI 只描述分子抽样波动，不覆盖全部训练随机性；只有两个训练实例。

补充子组（描述性，不用于选择）：ordinary n=935、tail(y≤−3) n=65、drop P3 worst-10
n=990、without row 172 n=999、cycle-penalty proxy n=36。cycle proxy 的 n 很小，且
按顺序敏感，不应当作机制证据。完整数字在 `valid_metrics.json`。

预注册 practical line 的五个条件（mean gain ≥0.003、两个 seed 都为正、ordinary 为正、
drop-P3-worst10 为正、CI 下界 >0）对 S/T/R 全部失败。`valid_decision.json` 的
`primary_label` 是 `NO_OBJECT_ATTENTION_PRACTICAL_GAIN`。

## 3. 机制诊断（label-free，先于 score 冻结）

这些只是响应统计，不是性能、化学、faithfulness 或因果结论。

* T/R 的 attention 接近均匀：归一化熵 layer 0 约 0.82–0.85，layer 1 约 0.956–0.970；
  effective keys layer 0 约 43–49，layer 1 约 73–78；最大权重 layer 0 约 0.11–0.14，
  layer 1 约 0.037–0.045。没有形成明显的稀疏结构选择。
* R 的结构偏置确实学到了非零值：每个 seed 152 个参数中 144 个非零；每层 type 36/36、
  distance 20/24、overlap 16/16。把 R 的 bias 置零后，预测的平均绝对变化约 0.0078
  （s0）和 0.0072（s1），最大约 0.030/0.036；相对 MAE≈0.09 很小。
* S 的 gamma 没有塌缩：layer 0 均值约 0.69/0.70，layer 1 约 0.49/0.33（s1 有接近 0
  的分量）。这支持“S 仍有实际容量/readout 变化”，但 S 本身也没有超过 P3。

因此，至少在这次训练里，R 的 relation bias 太弱、attention 太接近均匀，无法把
结构关系变成可测的 readout 收益。这个观察不能反过来证明“关系没用”；它只说明在当前
substrate、recipe、容量和预算下没有检测到收益。

## 4. 检查与执行记录

| 检查 | 结果 |
|---|---|
| CPU gates | 12/12 pass；`cpu_gates_this_round.json` |
| 参数 | S/T 483,137；R 483,289；P3 413,505；旧 reader 148,481 已删除 |
| P3 bypass | `z_P3` 与原始 Null grouping/composer reader input 完全一致，max abs diff 0.0 |
| T/R step-0 | zero-bias 时 prediction max abs diff 0.0；MHA/common state 相同 |
| 关系 bin 手检 | path/star 图的 type/dist/overlap 全部符合手算；disconnected bin=5 |
| 不变性 | atom renumber 2.38e−7；bond/junction endpoint swap 0；token permutation 9.54e−7；batch composition 3.73e−8；均 ≤2e−5 |
| 边界 | 单原子、无 bond/junction 均 finite，absent junction sum 精确为 0 |
| 梯度 | 两个 update 全部 finite；type embedding/reader/token FFN/mix/R bias 均参与 |
| batch order | 两个 seed 与历史 R3 hash 完全一致：`3c964a71…`、`8710751a…` |
| P3 reuse | branch `A_REUSE_FROZEN_P3`；P3 valid replay max abs diff 1.91e−6 / 2.86e−6，≤5e−5 |
| 正式训练 | 6/6 runs，每个 28,440 updates；batch order 和 LR schedule 断言通过；last-5 soup |
| 预测冻结 | 6 个单模型预测，combined hash `91a4b62b09f2…`；见 `valid_prediction_freeze.json` |
| score | 单次读取；`valid_read.lock` 记录 `read_count: 1`、`reread: false` |

P3 reference 使用分支 A 复用，不重训；执行 regime 与历史 P3 匹配
（A100-SXM4-40GB、driver 550.163.01、torch 2.5.1+cu124、CUDA 12.4、Python 3.12.14），
且 soup hash、init/encoder hash、label-free replay 全部重放通过。

### 4.1 Smoke timing amendment

第一次 smoke 用全部 16 个 update 的 raw mean，投影 4.72 GPU-h，超过 4.0，按规则不能
开始正式矩阵。检查发现每个 arm 的第一个 update 包含 CUDA/library warm-up；在**任何正式
训练开始之前**，把投影估计器改为“每个 arm 去掉前 2 个 update 后的 mean”，margin 1.20、
300 s reserve、epochs/batch/architecture/评分规则和 4.0 h cap 全部不变。第二次 smoke
投影 3.906 GPU-h，通过。两次 smoke 记录和 `smoke_timing_amendment.json` 都保留。这是
一个 timing-estimator amendment，不是 epoch/width 的削减。

### 4.2 GPU ledger

| item | GPU hours |
|---|---:|
| formal training jobs (both GPUs, sum) | 3.383 |
| two smoke runs + two audits + housekeeping + predict/mechanism/freeze | 0.021 |
| **total occupancy** | **3.404** |
| cap | 4.000 |
| remaining | 0.596 |

`gpu_ledger.json` 逐项记录 run id、起止时间和 wall seconds。两个 GPU 并行运行，因此
总 GPU-hours 是占用之和，不是 wall-clock span。没有隐藏的失败重试或额外 GPU 时间。
R 每个 run 约 3411–3539 s；S 约 1172–1339 s；T 约 1266–1388 s。

## 5. 解释与边界

1. **没有检测到增益，不是等价。** S vs P3 的 CI 跨 0，但点估计更差；T/R vs P3 的 CI
   整个在 0 以下。不能写成“和 P3 一样好”。
2. **S 也没有赢。** 这说明仅替换 reader、保留 P3 统计 bypass 并加入同参数量的 token
   FFN 容量，并不足以超过 P3。不能把 T/R 的失败简单归因于“注意力不够强”。
3. **R 没有赢 T。** R 的 relation bias 非零且可消融，但效应小；attention 近均匀。
   这更像“结构偏置没有在这个 recipe 下形成有效选择”，而不是“关系信息被证明无用”。
4. **参数匹配不等于有效容量匹配。** S/T 都是 483,137 参数，R 是 483,289；但 T/R 的
   analytic FLOPs 约为 S 的 1.24 倍，R 的实际 wall time 约为 S 的 2.8 倍。不能把
   S/T/R 称为严格 compute-matched。
5. **valid split 已暴露。** 这是 development split，不是 blind test；CI 只覆盖分子抽样，
   不覆盖训练随机性。只有两个 seed。
6. **没有 official test。** 没有加载、解析或预测 official test；没有与 GRIT/ESA+ 的
   official-test 数字比较。SAT/TokenGT/GPM/ESA/GRIT 等已存在；“structure token +
   attention”本身不是新颖性声明。

## 6. 决策

按预注册规则，`NO_OBJECT_ATTENTION_PRACTICAL_GAIN`：**关闭这个 object-token attention
架构筛选**。不追加 layer/head/relation bucket/seed，不通过 dropout、LR、residual scaling
或 valid 选择来救援。若未来继续，应当作为新的、单独预注册的机制问题，而不是本轮的
补救。

## 7. 交付物

* `PROTOCOL.md` — 冻结协议（含 timing amendment 说明）。
* `configs/zinc_structural_token_attention_v1.yaml` — 架构、recipe、hash、预算。
* `code/zinc_sta_v1/` — 模型、数据、runner、9 个 CPU tests。
* `cpu_gates_this_round.json` — 12/12 gates。
* `p3_reuse_audit.json` — branch A、regime、replay。
* `smoke_this_round.json` / `smoke_this_round_v1_mean_failed.json` /
  `smoke_timing_amendment.json` — 两次 smoke 与修订。
* `train_{S,T,R}_s{0,1}.json` — 6 个正式 run 的完整记录。
* `valid_prediction_freeze.json` / `valid_freeze_*` — 6 个预测的路径与 hash。
* `pre_score_freeze.json` / `pre_score_freeze_addendum.json` — score 前冻结。
* `valid_metrics.json` / `valid_decision.json` — 唯一一次 valid 读取后的指标与决策。
* `run_manifest.tsv` — 每个 run 的 soup/prediction hash、seed MAE、wall time、
  batch-order hash、commit。Reference 行的 `git_commit` 是本轮用于重放/核对 P3 的
  commit；P3 soup 本身的历史 hash 在 `soup_sha256` 列。
* `gpu_ledger.json` — 3.404 GPU-h 的逐项账本。
* `valid_read.lock` / `valid_access_log.json` — 单次读取与访问披露。

Prediction arrays 和 soups 体积较大，默认不进入 git；它们的 SHA-256 已写入上述
freeze/manifest 记录。
