# ERRATA — `lcrc_predictive_trace_benchmark_v1` 的三处统计/文本更正

Track: `tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1/`
Upstream report: `tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/REPORT.md`
基准提交：`6cf9ca4eb3ddd45701c73baea66e80642034b756`
性质：**`POST_HOC_EXPOSED_EVALUATION`**。本轮只重算已保存预测上的离线统计；
没有重训、没有重新推理、没有新的 Test 评分。

原始 `analysis/statistics.json`、每个 seed 的 bootstrap 数值、checkpoint、
冻结 `PROTOCOL.md`、划分和标签访问账本**均未修改**。旧 `REPORT.md` 只增加了
指向本文件的勘误块，并做了最小文本修正。

---

## P0-1 FULL/C1 胜负计数

旧报告第 0 节写 “C1 在 4/10 更好、6/10 更差”。逐数据集重算三 seed 均值后：

* **正确计数：C1 在 6/10 更好、4/10 更差。**
* 分子任务：C1 更好 5 个（ESOL、FreeSolv、BACE、BBBP、Tox21），更差 1 个（MolHIV）。
* TU：C1 更好 1 个（IMDB-MULTI），更差 3 个（NCI1、PROTEINS、IMDB-BINARY）。
* 5 + 1 = 6，因此 “5/10 更好” 与指令自己列出的逐任务方向不一致；
  实际重算为 **6/10 更好、4/10 更差**。

| dataset | metric | A0 mean | C1 mean | Δ C1−A0 | better |
|---|---|---:|---:|---:|---|
| ESOL | RMSE (lower better) | 0.564457 | 0.553387 | −0.011069 | C1 |
| FreeSolv | RMSE (lower better) | 1.092179 | 1.071093 | −0.021085 | C1 |
| BACE | ROC-AUC | 0.816848 | 0.826329 | +0.009481 | C1 |
| BBBP | ROC-AUC | 0.704114 | 0.705977 | +0.001863 | C1 |
| Tox21 | macro ROC-AUC | 0.835259 | 0.837447 | +0.002188 | C1 |
| ogbg-molhiv | ROC-AUC | 0.771606 | 0.761121 | −0.010485 | A0 |
| NCI1 | accuracy | 0.829562 | 0.824088 | −0.005474 | A0 |
| PROTEINS | accuracy | 0.759199 | 0.753341 | −0.005858 | A0 |
| IMDB-BINARY | accuracy | 0.737000 | 0.726000 | −0.011000 | A0 |
| IMDB-MULTI | accuracy | 0.481000 | 0.484333 | +0.003333 | C1 |

来源：`lcrc_predictive_trace_benchmark_v1/analysis/molecular_results.csv`
与 `tu_results.csv` 的 `test_primary` / `test_accuracy`；逐项数值与
`analysis/p0_errata.json::P0_1_full_c1_count` 一致。

---

## P0-2 typed-S2 计数范围

`analysis/mechanism_audit.json` 的 `intervention.rows` 共 **13** 条：

* 12 条 `THIS_ROUND_FROZEN_CHECKPOINT`（6 个分子数据集 × A0/C1）；
* 1 条 `FROZEN_UPSTREAM_CHECKPOINT`（Lipophilicity A0）。

因此 “本轮 12 个” 与 “含历史 13 个” 是两个**范围**，不是互相矛盾的结果。
用固定判据重算：

| 判据 | 本轮 12 个 | 含历史 13 个 |
|---|---:|---:|
| typed-S2 `mean_abs_delta` **严格最大** | **11/12** | **12/13** |
| typed-S2 比次大通道高 **≥5%** | **10/12** | **11/13** |

5% 边际判据下，MolHIV 两行都记为 `comparable`：

* MolHIV A0：typed-S2 4.182，`mean_h2` 4.308（typed-S2 不是最大）；
* MolHIV C1：typed-S2 4.362，`mean_h2` 4.233（高 3.05%，低于 5% 边际）。

所以旧摘要的 `10/12` 与正文的 `11/13` 在“5% 边际 + 不同范围”下同时成立；
严格数值最大则是 `11/12` 与 `12/13`。这不是结果错误，而是**判据与范围未写明**。

**边界**：这是**固定权重下、不同宽度输入块置零**的敏感度比较（typed-S2 200 列，
`g` 42 列，`mean_h2` 64 列），不是同容量重训消融，不能声称 S2 在训练后“必要”或
“充分”，也不能把置零敏感度当作因果效应。

来源：`analysis/p0_errata.json::P0_2_typed_s2_scope`。

---

## P0-3 分子任务置信区间口径（本轮最重要的统计勘误）

### 旧口径是什么

`lcrc_predictive_trace_benchmark_v1/code/ptb_report.py::molecule_summary`：

```python
lo = min(p["boot_lo"] for p in per_seed)
hi = max(p["boot_hi"] for p in per_seed)
```

即对三个 seed 各自的行 bootstrap 区间取**最小值/最大值外包络**。
该区间记为 **`SEED_WISE_CI_ENVELOPE`**，它回答的是“三个 seed 的区间合起来覆盖哪里”，
**不是**“跨三个 seed 的平均 C1−A0 效应”的 95% CI。

### 新定义（固定、可复现）

对每个分子数据集：

* 目标量 `Δ_mean = mean_seed[metric(C1, seed) − metric(A0, seed)]`。
* 三个 seed 的 checkpoint 均固定。
* 每次 bootstrap 从同一 Test 行集合抽样；**同一 replicate 内三个 seed 与 A0/C1
  使用完全相同的行索引**。
* 分别重算每个 seed 的主指标差，再对三个差值取平均。
* 2000 次，seed `20261103`，2.5%/97.5% 分位数。
* 回归按分子行重抽；单任务二分类按正/负样本分层重抽并保持原类别数量；
  Tox21 按分子行重抽，只使用原始 Test 上 12 个可定义任务，若某次重抽使任一任务
  缺少正/负类则整次无效并重抽（上限 10000 次尝试；本次 6 个数据集均为
  2000/2000 有效，无 `UNAVAILABLE`）。

### 结果

`Δ_mean` 的条件 bootstrap CI（不是旧 envelope）：

| dataset | Δ_mean | seed SD of paired Δ | new 95% CI | spans 0 | old seed-wise envelope | per-seed Δ (s0,s1,s2) |
|---|---:|---:|---|---:|---|---|
| ESOL | −0.011069 | 0.001204 | [−0.031114, +0.010763] | yes | [−0.050656, +0.024824] | −0.012171, −0.011253, −0.009784 |
| FreeSolv | −0.021085 | 0.014484 | [−0.062775, +0.029630] | yes | [−0.092106, +0.101738] | −0.011583, −0.037756, −0.013917 |
| BACE | +0.009481 | 0.011518 | **[+0.001569, +0.018721]** | **no** | [−0.025943, +0.045296] | +0.020652, −0.002355, +0.010145 |
| BBBP | +0.001863 | 0.005909 | [−0.007131, +0.011017] | yes | [−0.016922, +0.028335] | +0.006937, −0.004625, +0.003276 |
| Tox21 | +0.002188 | 0.003843 | [−0.000701, +0.004890] | yes | [−0.002338, +0.015347] | +0.006613, −0.000311, +0.000263 |
| ogbg-molhiv | −0.010485 | 0.014741 | **[−0.020692, −0.000391]** | **no** | [−0.051130, +0.009333] | −0.006787, +0.002055, −0.026723 |

关键变化：

* **旧 “6/6 分子任务 CI 跨 0” 不再成立。** 新口径下 **4/6** 跨 0；
  **BACE 与 MolHIV 排除 0**。
* BACE 的正效应很小（+0.0095 AUC），且 seed 1 的配对差为负（−0.0024）；
  MolHIV 的负效应（−0.0105）主要由 seed 2 的 −0.0267 驱动，seed 1 为正（+0.0021）。
  因此区间排除 0 只说明“条件于这 3 个 checkpoint 的平均差”在此行重抽下稳定，
  **没有**覆盖训练 seed 变异；不能据此宣称 C1 普遍更好/更差或等价。
* TU 的旧区间是 **outer-fold bootstrap**，与分子任务的 seed-conditional 定义不同，
  本轮未改写。若把两种区间并列计数，旧 “10/10 跨 0” 变为 **8/10**
  （分子 4/6 + TU 4/4），但**不能把两种口径合并成一个 95% CI**。

### 旧 `statistics.json` 是否修改

**否。** 旧 `statistics.json` 保留每个 seed 的 `boot_lo/boot_hi`、`delta_mean`、
`delta_std_over_seeds`。新结果只写入：
`analysis/mean_seed_bootstrap_ci.json` 与 `.csv`。

来源：`analysis/mean_seed_bootstrap_ci.json`、`analysis/p0_errata.json`。
