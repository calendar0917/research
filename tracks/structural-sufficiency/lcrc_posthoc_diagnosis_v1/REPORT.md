# REPORT — LCRC 事后诊断 v1（P0 勘误 + P1-A/B/C）

Round ID: `lcrc-posthoc-diagnosis-v1`
Track: `tracks/structural-sufficiency/lcrc_posthoc_diagnosis_v1/`
基准提交：`6cf9ca4eb3ddd45701c73baea66e80642034b756`
结果标签：**`POST_HOC_EXPOSED_EVALUATION`**
性质：对上一轮已冻结预测与已保存 Audit 派生量的离线重分析；**不是预注册实验，
不是独立验证**。定义见 `PROTOCOL.md`。

---

## 0. 一页结论

**旧报告修正了什么？**

1. **FULL/C1 胜负计数**：逐数据集三 seed 均值重算为 **C1 6/10 更好、4/10 更差**
   （旧报告写 4/10、6/10；任务指令自己列出的逐任务方向相加也是 6 个更好，
   不是 5 个）。见 §1.1。
2. **typed-S2 计数范围**：`10/12` 是 12 个本轮 checkpoint，`11/13` 是加上 1 个
   历史 Lipophilicity checkpoint；按 5% 边际判据两者同时成立。严格数值最大为
   `11/12` 与 `12/13`。这不是结果错误，是判据/范围未写明。见 §1.2。
3. **分子任务 CI 口径**：旧 “Δ 95% CI” 实际是三个 seed 各自 bootstrap 区间的
   外包络（`SEED_WISE_CI_ENVELOPE`），不是跨 seed 平均 C1−A0 效应的 CI。
   新的 seed-conditional mean-difference bootstrap 下，**4/6 分子任务跨 0；
   BACE 与 MolHIV 排除 0**。旧 “10/10 跨 0” 不再成立；按不同口径并列计数为
   8/10（分子 4/6 + TU 4/4），但两者不能合并成一个 CI。见 §1.3。

**MolHIV 的误差与哪些结构支持指标存在可见关系？**

* 绝对排序性能在“有未见二跳根结构”的 Test 分子上明显更差：
  `fraction_wl2_unseen=0` 组 A0 AUC 均值 **0.9249**，
  `(0,0.25]` **0.7116**，`(0.25,0.50]` **0.6809**；`(0.50,1]` 只有 7 个正例，
  记 `LOW_SUPPORT`。相似度和分子大小也呈同方向的分组差异。
* 但这些分组同时强烈改变正例比例（三组 8.16% / 3.08% / 1.78%），
  且低支持、低相似度、小分子、少正例彼此混杂；不能把差异归因于“低支持”本身。
* C1 相对 A0 的退化不是单调的 support 梯度：`0` 组 −0.003、`(0,0.25]` −0.011、
  `(0.25,0.50]` −0.028（AUC）；`(0.50,1]` 无法估计。

**该关系是否跨 seed 和 FULL/C1 稳定？**

* 总体 C1−A0 = **−0.0105 AUC**，但 seed 间为 −0.0068 / +0.0021 / −0.0267：
  **主要由 seed 2 驱动**，不是三个 seed 一致。
* 唯一在三个 seed 上方向一致、且行 bootstrap CI 排除 0 的结构组是
  **`fraction_wl2_unseen ∈ (0.25,0.50]`**：ΔAUC = **−0.0279**，
  CI [−0.0496, −0.0040]，n=1292，但只有 **23 个正例**。
* 该组在 A0/C1 之间没有“FULL vs C1 的容量机制”解释；它只是描述性集中，
  不能称为 S2 的因果效果。

**LCRC 是否比零变化基线更能预测真实分子局部编辑效应？**

* 是。Audit 74 对 / 26 components 上，component-equal
  `MAE_MODEL` A0 均值 **0.4027**、C1 均值 **0.4133**；
  `MAE_ZERO` **0.6823**。
  `Edit_Gain` A0 **+0.2796**、C1 **+0.2690**；component bootstrap 95% CI
  A0 [+0.1153, +0.4544]、C1 [+0.0968, +0.4492]，2000 次中 >99.9% 为正。
  方向一致率约 **0.70–0.73**（67/74 对 `|Δy|≥0.10`），Spearman **0.60–0.67**。
* 这与旧 TRACE “不能稳定预测 `e_delta`、对零误差修正基线无可靠增益”是**两个不同问题**：
  旧 `NATIVE`（令 `e_delta=0`）不是这里的 `ZERO_CHANGE`（令 `Δpred=0`）。
  新结果说明模型的编辑响应本身有信息，**不**说明 TRACE 解释了它的错误。
* 限制：74 对不是 74 个独立样本，只有 26 个 component；单一数据集、
  Audit 已曝光、事后分析。不能据此宣称通用能力。

**是否有足够证据进入下一轮容量匹配 S2-off 实验？**

`NEEDS_BETTER_EVIDENCE`。存在一个跨 3 seed 一致的结构组差异，但只有 23 个正例、
是多个固定组中的一个、与大小/标签比例混杂，且没有 seen-scaffold 对照；
现有机制证据是固定权重置零敏感度，不是训练后 S2 必要性。见 `DECISION.md` Q4。
**本轮不启动任何新实验。**

---

## 1. 统计口径勘误

### 1.1 FULL/C1 胜负计数（P0-1）

旧报告第 0 节写 “C1 在 4/10 更好、6/10 更差”。逐数据集三 seed 均值：

| dataset | kind | metric | A0 | C1 | Δ C1−A0 | better |
|---|---|---|---:|---:|---:|---|
| ESOL | molecule | RMSE ↓ | 0.564457 | 0.553387 | −0.011069 | C1 |
| FreeSolv | molecule | RMSE ↓ | 1.092179 | 1.071093 | −0.021085 | C1 |
| BACE | molecule | ROC-AUC ↑ | 0.816848 | 0.826329 | +0.009481 | C1 |
| BBBP | molecule | ROC-AUC ↑ | 0.704114 | 0.705977 | +0.001863 | C1 |
| Tox21 | molecule | macro ROC-AUC ↑ | 0.835259 | 0.837447 | +0.002188 | C1 |
| ogbg-molhiv | molecule | ROC-AUC ↑ | 0.771606 | 0.761121 | −0.010485 | A0 |
| NCI1 | TU | accuracy ↑ | 0.829562 | 0.824088 | −0.005474 | A0 |
| PROTEINS | TU | accuracy ↑ | 0.759199 | 0.753341 | −0.005858 | A0 |
| IMDB-BINARY | TU | accuracy ↑ | 0.737000 | 0.726000 | −0.011000 | A0 |
| IMDB-MULTI | TU | accuracy ↑ | 0.481000 | 0.484333 | +0.003333 | C1 |

* **正确计数：6/10 更好、4/10 更差。**
* 分子 5 个更好 + TU 1 个更好 = 6。任务指令中的 “5/10 更好、5/10 更差”
  与它自己列出的逐任务方向不一致；本轮按实际数据记录为 6/4。
* 只改描述，不修改 `molecular_results.csv`、`tu_results.csv` 或任何原始分数。

来源：`analysis/p0_errata.json::P0_1_full_c1_count`；旧报告已加最小勘误指向。

### 1.2 typed-S2 计数范围（P0-2）

`analysis/mechanism_audit.json` 的 `intervention.rows` 共 **13** 条：
12 条本轮新 checkpoint（6 数据集 × A0/C1）+ 1 条历史 Lipophilicity A0。

| 判据 | 本轮 12 个 | 含历史 13 个 |
|---|---:|---:|
| typed-S2 `mean_abs_delta` 严格最大 | 11/12 | 12/13 |
| typed-S2 比次大通道高 ≥5% | **10/12** | **11/13** |

5% 边际判据下 MolHIV 两行都记为 `comparable`：A0 的 typed-S2 4.182 <
`mean_h2` 4.308；C1 的 typed-S2 4.362 仅比 `mean_h2` 4.233 高 **3.05%**。
因此摘要 `10/12` 与正文 `11/13` 是**同一判据下的范围差异**，不是结果矛盾；
严格数值最大则是 `11/12`、`12/13`。

**边界**：这是固定权重下不同宽度输入块（typed-S2 200 列、`g` 42 列、
`mean_h2` 64 列）置零的敏感度比较，不是同容量重训消融，不是化学编辑，
也不能证明 S2 在训练后“必要”或“充分”。

### 1.3 分子任务跨 seed 平均差的 CI（P0-3）

旧 `code/ptb_report.py::molecule_summary`：

```python
lo = min(p["boot_lo"] for p in per_seed)
hi = max(p["boot_hi"] for p in per_seed)
```

这是 **`SEED_WISE_CI_ENVELOPE`**：三个 seed 各自行 bootstrap 区间的外包络，
不是“跨三个 seed 的平均 C1−A0 效应”的 95% CI。

新定义（`PROTOCOL.md` §1，固定）：
`Δ_mean = mean_seed[metric(C1,s) − metric(A0,s)]`；每次从同一 Test 行集合重抽，
**同一 replicate 内三个 seed 与 A0/C1 使用完全相同的行索引**；2000 次，
seed `20261103`；回归按行、单任务二分类按正/负分层、Tox21 按分子行且要求
12 个任务均有正负类（无效重抽重试；6 个数据集均 2000/2000 有效）。

| dataset | Δ_mean | seed SD of paired Δ | new seed-conditional 95% CI | spans 0 | old seed-wise envelope | per-seed Δ (s0,s1,s2) |
|---|---:|---:|---|---:|---|---|
| ESOL | −0.011069 | 0.001204 | [−0.031114, +0.010763] | yes | [−0.050656, +0.024824] | −0.012171, −0.011253, −0.009784 |
| FreeSolv | −0.021085 | 0.014484 | [−0.062775, +0.029630] | yes | [−0.092106, +0.101738] | −0.011583, −0.037756, −0.013917 |
| BACE | +0.009481 | 0.011518 | **[+0.001569, +0.018721]** | **no** | [−0.025943, +0.045296] | +0.020652, −0.002355, +0.010145 |
| BBBP | +0.001863 | 0.005909 | [−0.007131, +0.011017] | yes | [−0.016922, +0.028335] | +0.006937, −0.004625, +0.003276 |
| Tox21 | +0.002188 | 0.003843 | [−0.000701, +0.004890] | yes | [−0.002338, +0.015347] | +0.006613, −0.000311, +0.000263 |
| ogbg-molhiv | −0.010485 | 0.014741 | **[−0.020692, −0.000391]** | **no** | [−0.051130, +0.009333] | −0.006787, +0.002055, −0.026723 |

关键解释：

* **旧 “6/6 分子 CI 跨 0” 不成立**；新口径 **4/6** 跨 0，BACE 与 MolHIV 排除 0。
* BACE 的正效应小（+0.0095），seed 1 的配对差为负；MolHIV 的负效应（−0.0105）
  主要由 seed 2（−0.0267）驱动，seed 1 为正（+0.0021）。区间排除 0 只说明
  “条件于这 3 个 checkpoint 的平均差”在行重抽下稳定，**没有覆盖训练 seed 变异**。
* 不能据此宣称 C1 普遍更好/更差，也不能宣称等价。
* TU 旧区间是 outer-fold bootstrap，与分子任务定义不同；本轮未改写。
  并列计数为 **8/10 跨 0**（分子 4/6 + TU 4/4），但两种区间不能合并。

旧 `analysis/statistics.json` 与每 seed 的 `boot_lo/boot_hi` 未修改；新结果只在
`analysis/mean_seed_bootstrap_ci.{json,csv}`。

---

## 2. MolHIV：结构支持度与排序错误

### 2.1 输入核验

读取 6 个 `runs/mol_ogb-molhiv_s<seed>_<model>/test_predictions.npz`
（`rows`、`pred`、`y_true`）及对应 `test_eval.json`、`run.json`：

* 6 个文件均存在；`rows` 与官方 `split/scaffold/test.csv.gz` 的 4113 行
  **逐位相同**（顺序也相同）；六个文件的 `y_true` 对齐（含 0/1）；
* `pred` 为有限 raw score，形状 [4113, 1]；
* 离线重算 ROC-AUC 与 `test_eval.json`/OGB evaluator **最大差 0.0**；
* 6 个 checkpoint 的实际远程 SHA-256 与 `checkpoint_manifest.json` 全部一致
  （`correctness_gates.json`）。

SMILES 只从 `mapping/mol.csv.gz` 的 `smiles` 列读取；`raw/graph-label.csv.gz`
未打开。普通 RDKit sanitization 在 7 个 OGB 分子上失败（3 Train、2 Valid、
2 Test）；本轮只使用 Train/Test，故实际回退 5 个（3 Train、2 Test），
用 `sanitize=False` + `UpdatePropertyCache` + `GetSymmSSSR` 回退。
2 个 Test 回退分子仍能计算 WL/Morgan 特征。`n_morgan_failures=0`。

### 2.2 全体指标与 rank-loss 恒等式

| unit | ROC-AUC | PR-AUC | positive RL | negative RL |
|---|---:|---:|---:|---:|
| A0:s0 | 0.765349 | 0.246936 | 0.234651 | 0.234651 |
| A0:s1 | 0.773613 | 0.312206 | 0.226387 | 0.226387 |
| A0:s2 | 0.775857 | 0.298405 | 0.224143 | 0.224143 |
| C1:s0 | 0.758562 | 0.250525 | 0.241438 | 0.241438 |
| C1:s1 | 0.775668 | 0.316780 | 0.224332 | 0.224332 |
| C1:s2 | 0.749134 | 0.288104 | 0.250866 | 0.250866 |

* A0 mean AUC **0.771606**（sd 0.005534），C1 mean **0.761121**（sd 0.013451）。
* A0 mean PR-AUC **0.285849**，C1 mean **0.285136**。
* 对每个 unit 校验 `1 − mean_positive(RL) = ROC-AUC`，最大绝对误差 < `1e-12`；
  `mean_negative(RL)` 在同一组上也等于 `1−AUC`（这是数学恒等式；逐样本分布不同）。
* 全体 C1−A0 的分层行 bootstrap：ΔAUC **−0.0105**，CI
  [−0.02069, −0.00039]，2000 次中 97.9% 的 Δ positive RL > 0。

### 2.3 固定结构特征与支持度

* rooted radius-2 WL：原子描述符 `atomic_number, formal_charge, is_aromatic,
  total_num_H, total_degree, is_in_ring`；边描述符 `bond_type, is_aromatic,
  is_conjugated, is_in_ring`；稳定 JSON + SHA-256；两轮更新。
* Train 支持度按**不同 Train 分子**计数（不是原子出现次数）。
* Morgan r=2/2048 到 Train 的最大 Tanimoto，分块计算；canonical SMILES 重复；
  Bemis–Murcko scaffold 交集。
* **Scaffold split 下 Test scaffold 与 Train 交集为 0**（4111 个 Test scaffold，
  0 个 seen）。因此**没有可比较的 seen-scaffold 组**，不构造该分组。
* Test canonical SMILES 与 Train 完全重复：**0**。

分组样本量（固定阈值，看错误前确定）：

| dimension | group | n | pos | neg | positive rate | status |
|---|---|---:|---:|---:|---:|---|
| wl2_unseen | 0 | 527 | 43 | 484 | 8.16% | OK |
| wl2_unseen | (0,0.25] | 1848 | 57 | 1791 | 3.08% | OK |
| wl2_unseen | (0.25,0.50] | 1292 | 23 | 1269 | 1.78% | OK |
| wl2_unseen | (0.50,1] | 446 | 7 | 439 | 1.57% | **LOW_SUPPORT** |
| train_tanimoto | [0,0.30) | 536 | 12 | 524 | 2.24% | OK |
| train_tanimoto | [0.30,0.50) | 2367 | 53 | 2314 | 2.24% | OK |
| train_tanimoto | [0.50,0.70) | 949 | 44 | 905 | 4.64% | OK |
| train_tanimoto | [0.70,1] | 261 | 21 | 240 | 8.05% | OK |
| heavy_atoms | ≤15 | 621 | 9 | 612 | 1.45% | **LOW_SUPPORT** |
| heavy_atoms | 16–25 | 1977 | 37 | 1940 | 1.87% | OK |
| heavy_atoms | >25 | 1515 | 84 | 1431 | 5.54% | OK |

**结构分组与标签比例**：未见比例越高，正例比例越低（8.16% → 3.08% → 1.78% → 1.57%）；
相似度越高、分子越大，正例比例越高（2.24% → 4.64% → 8.05%；1.45% → 1.87% → 5.54%）。
因此任何“低支持组 AUC 更低”的读数都同时带有类别比例和化学难度的差异。

### 2.4 固定分组的绝对排序性能

A0/C1 三 seed 均值的组内 ROC-AUC / PR-AUC：

| dimension | group | A0 AUC | C1 AUC | A0 PR | C1 PR | status |
|---|---|---:|---:|---:|---:|---|
| wl2_unseen | 0 | 0.9249 | 0.9220 | 0.7153 | 0.6926 | OK |
| wl2_unseen | (0,0.25] | 0.7116 | 0.7005 | 0.1672 | 0.1676 | OK |
| wl2_unseen | (0.25,0.50] | 0.6809 | 0.6530 | 0.0446 | 0.0355 | OK |
| wl2_unseen | (0.50,1] | — | — | — | — | LOW_SUPPORT |
| train_tanimoto | [0,0.30) | 0.6297 | 0.6063 | 0.0366 | 0.0363 | OK |
| train_tanimoto | [0.30,0.50) | 0.6751 | 0.6695 | 0.0878 | 0.0865 | OK |
| train_tanimoto | [0.50,0.70) | 0.8720 | 0.8593 | 0.5400 | 0.5256 | OK |
| train_tanimoto | [0.70,1] | 0.8870 | 0.8749 | 0.5788 | 0.5424 | OK |
| heavy_atoms | ≤15 | — | — | — | — | LOW_SUPPORT |
| heavy_atoms | 16–25 | 0.6607 | 0.6498 | 0.0454 | 0.0470 | OK |
| heavy_atoms | >25 | 0.8284 | 0.8157 | 0.4539 | 0.4548 | OK |

* **绝对误差**明显集中在“有未见根结构 / 低相似度 / 中间大小”的组：
  `wl2_unseen=0` 的 A0 AUC 0.925，而 `(0.25,0.50]` 只有 0.681；
  相似度最低组 0.630，最高组 0.887。
* 但这些组的正例比例相差 4–6 倍，不能把绝对 AUC 差异直接归因于支持度。
* `≤15` 重原子组只有 9 个正例，`(0.50,1]` 只有 7 个正例，均不报告独立 AUC。

### 2.5 A0/C1 配对差与 seed 方向

对每个固定组，同一 Test 行上 C1−A0 的 ΔAUC（负值 = C1 更差）：

| dimension | group | n | pos | ΔAUC mean | 95% bootstrap CI | seed SD | 3/3 同向？ |
|---|---|---:|---:|---:|---|---:|---|
| wl2_unseen | 0 | 527 | 43 | −0.0029 | [−0.0083, +0.0019] | 0.0146 | 否 |
| wl2_unseen | (0,0.25] | 1848 | 57 | −0.0111 | [−0.0309, +0.0081] | 0.0237 | 否 |
| wl2_unseen | **(0.25,0.50]** | 1292 | 23 | **−0.0279** | **[−0.0496, −0.0040]** | 0.0167 | **是（3/3 负）** |
| train_tanimoto | [0,0.30) | 536 | 12 | −0.0233 | [−0.0634, +0.0230] | 0.0192 | 是（3/3 负）但 CI 跨 0 |
| train_tanimoto | [0.30,0.50) | 2367 | 53 | −0.0056 | [−0.0254, +0.0138] | 0.0162 | 否 |
| train_tanimoto | [0.50,0.70) | 949 | 44 | −0.0127 | [−0.0264, −0.0019] | 0.0211 | 否 |
| train_tanimoto | [0.70,1] | 261 | 21 | −0.0121 | [−0.0250, −0.0019] | 0.0298 | 否 |
| heavy_atoms | 16–25 | 1977 | 37 | −0.0109 | [−0.0333, +0.0123] | 0.0214 | 否 |
| heavy_atoms | >25 | 1515 | 84 | −0.0127 | [−0.0248, −0.0010] | 0.0152 | 否 |

* 只有 `(0.25,0.50]` 同时满足：三个 seed 方向一致、CI 排除 0。但 n_pos=23，
  是 4 个固定 support 组中的一个，`(0.50,1]` 还因 7 个正例无法估计。
* 该组的 per-seed ΔAUC 为 **−0.0358 / −0.0087 / −0.0392**；C1:s2 最差，
  但 seed 0/1 也负，不是单 seed 造成。
* 其他组即便 CI 排除 0（相似度 0.5–0.7、0.7–1、大小 >25），也不是三 seed 同向；
  不能称为稳定分组退化。

**C1:s2 是否集中在特定组？** 总体 C1:s2 −0.0267；seed 2 相对 seed 0/1 均值的
额外退化在大多数组为负（最大：相似度 [0.70,1] −0.0515、[0.50,0.70) −0.0363、
support (0,0.25] −0.0371），说明 seed 2 的退化是**较广的**，不是单一组；
`(0.25,0.50]` 的 seed2 额外退化只有 −0.0170，反而更接近其他 seed。

### 2.6 相关性与混杂

Per-molecule rank loss 是相对**全体**反类样本计算的，不是组内 rank loss。
因此全体 Spearman 与组内 AUC 不是同一个量，符号可以不同。A0 结果：

| feature | all | positive class | negative class |
|---|---:|---:|---:|
| fraction_wl2_unseen | −0.125 [−0.158, −0.093] | **+0.663** [+0.549, +0.751] | −0.164 [−0.196, −0.131] |
| fraction_wl2_low_support | −0.125 [−0.158, −0.093] | +0.616 [+0.502, +0.705] | −0.165 [−0.199, −0.132] |
| max_train_tanimoto | +0.106 [+0.077, +0.138] | **−0.576** [−0.685, −0.444] | +0.143 [+0.113, +0.176] |
| n_heavy_atoms | +0.246 [+0.216, +0.277] | **−0.651** [−0.754, −0.525] | +0.296 [+0.265, +0.327] |

C1 方向相同，数值略小（`fraction_wl2_unseen` positive +0.655，`max_train_tanimoto`
positive −0.564，`n_heavy_atoms` positive −0.631）。完整数值见
`analysis/molhiv_analysis.json::correlations`。

解释：

* 在**正例内部**，未见比例越高、相似度越低、分子越小，rank loss 越高；
  这与“低支持/低相似度/小分子更难”一致。
* 在**负例内部**符号相反：高未见比例的负例往往被模型打得更高（更接近正例），
  因而全局 rank loss 更低。全体相关是 96.8% 负例主导的混合，不能当作
  “低支持降低错误”的证据。
* 这些相关都只是描述性，且与类别比例、分子大小、化学空间互相混杂。

**协议指定的极端组对照不可估计。** `PROTOCOL.md` §2.6 指定在同一重原子大小
分层内比较 `fraction_wl2_unseen=0` 与 `(0.50,1]`。实际样本为：

| size | support=0 n/pos/neg | support=(0.50,1] n/pos/neg | 可估计？ |
|---|---|---|---|
| ≤15 | 20 / 1 / 19 | 223 / 2 / 221 | 否（两侧均 LOW_SUPPORT） |
| 16–25 | 180 / 1 / 179 | 168 / 3 / 165 | 否（两侧均 LOW_SUPPORT） |
| >25 | 327 / 41 / 286 | 55 / 2 / 53 | 否（高支持侧 LOW_SUPPORT） |

`(0.50,1]` 在三个大小层都只有 2–3 个正例，按预定义不能报告独立 AUC；
因此**协议指定的 0 vs 极端高支持对照在任何大小层都不可估计**，不构造替代阈值。

作为**明确标注的探索性折叠**，下面另报 `0 vs >0`（任意未见根结构）的
同大小层描述性对照；它有多于 10 个正例的只有 >25 层，仍不是协议主对照：

| size | support=0 n/pos/neg | A0/C1 AUC | support>0 n/pos/neg | A0/C1 AUC | ΔAUC C1−A0 (0 / >0) |
|---|---|---|---|---|---|
| ≤15 | 20 / 1 / 19 | LOW_SUPPORT | 601 / 8 / 593 | LOW_SUPPORT | — |
| 16–25 | 180 / 1 / 179 | LOW_SUPPORT | 1797 / 36 / 1761 | 0.6633 / 0.6511 | — / −0.0123 |
| >25 | 327 / 41 / 286 | **0.9194 / 0.9149** | 1188 / 43 / 1145 | **0.7360 / 0.7155** | **−0.0044 / −0.0205** |

在 >25 内，support=0 与 >0 的绝对 AUC 差约 0.18，C1 的额外退化在 >0 组更大
（−0.0205 vs −0.0044）；但正例比例 12.5% vs 3.6%，仍然无法把“支持度”与
“类别比例/化学难度”分开。此折叠是事后描述，不作为协议确证。

### 2.7 对 §3.8 五个问题的直接回答

1. **排序错误主要集中在哪些预定义结构组？** 绝对 AUC 最低的是
   `fraction_wl2_unseen ∈ (0.25,0.50]`（0.681）、低相似度 `[0,0.30)`（0.630）
   和 16–25 重原子（0.661）；但这些组正例比例也最低。C1 的**额外**退化
   在三 seed 同向且 CI 排除 0 的只有 `(0.25,0.50]`（低相似度 `[0,0.30)`
   也三 seed 同向，但只有 12 个正例、CI 跨 0）。
2. **是否在 A0/C1、三个 seed 中稳定？** 绝对低 AUC 在 A0/C1 都出现；
   C1 的额外退化不是全组稳定；三 seed 同向且 CI 排除 0 的只有
   `(0.25,0.50]`（`[0,0.30)` 也同向但 CI 跨 0）。
3. **低支持度是否提供超出大小/类别比例的可见信息？** 协议指定的 0 vs
   `(0.50,1]` 对照因高支持侧仅 2–3 个正例而不可估计；探索性的 0 vs >0
   折叠在 >25 内有可见差异，但类别比例同时差 3.5 倍。**不能**说已经超出
   大小与类别比例。记为 **UNIDENTIFIED / 混杂**。
4. **C1:s2 的退化是否局限于某些组？** 不是；seed 2 的额外退化在大多数组为负，
   是较广的 seed 效应。`(0.25,0.50]` 的三 seed 一致退化不是 seed 2 独有。
5. **是否有足够证据值得下一轮检验 S2 的重训后结构泛化价值？**
   有可检验的迹象，但证据不足以直接购买严格对照；见 `DECISION.md` Q4
   `NEEDS_BETTER_EVIDENCE`。

---

## 3. Local-edit：模型编辑响应 vs 零变化

### 3.1 数据核验

只读 `lcrc_local_edit_error_diagnosis_v1/analysis/probe_audit_scores.json`：

* 6 个 units（A0/C1 × s0/s1/s2），每个 **74 对 / 26 components**；
* `pair_id` 唯一；`component`、`class_id`、`same_var_heavy` 六个 unit 完全一致；
* `Δy = d_pred − e_delta` 在六个 unit 间的最大重建差 **4.44e-16**
  （容限 `1e-5`）；
* 未读取 Audit 原始标签；未重新配对或映射。

### 3.2 主指标（component-equal）

| unit | MAE_MODEL | MAE_ZERO | Edit_Gain | Spearman | Pearson | direction agreement |
|---|---:|---:|---:|---:|---:|---:|
| A0:s0 | 0.4221 | 0.6823 | **+0.2601** | 0.641 | 0.783 | 0.701 (67/74) |
| A0:s1 | 0.3944 | 0.6823 | **+0.2879** | 0.643 | 0.792 | 0.731 (67/74) |
| A0:s2 | 0.3916 | 0.6823 | **+0.2907** | 0.665 | 0.768 | 0.716 (67/74) |
| C1:s0 | 0.4070 | 0.6823 | **+0.2752** | 0.629 | 0.779 | 0.672 (67/74) |
| C1:s1 | 0.4178 | 0.6823 | **+0.2644** | 0.602 | 0.783 | 0.701 (67/74) |
| C1:s2 | 0.4149 | 0.6823 | **+0.2674** | 0.603 | 0.759 | 0.731 (67/74) |
| A0 seed mean | 0.4027 | 0.6823 | **+0.2796** (sd 0.0169) | 0.650 | — | 0.716 |
| C1 seed mean | 0.4133 | 0.6823 | **+0.2690** (sd 0.0056) | 0.611 | — | 0.701 |

* `Edit_Gain = MAE_ZERO − MAE_MODEL`；正值表示优于“性质不变”。
* 6 个 unit 的 gain 全部为正；A0/C1 seed 均值分别为 +0.2796 / +0.2690。
* 方向一致率只在 `|Δy| ≥ 0.10` 的 **67/74** 对上计算（7 对接近 0 的
  真实效应被排除，不强行定义正负）。
* 辅助 `RMSE_MODEL` 0.617–0.657，`RMSE_ZERO` 1.008（pair-level，未按 component 加权）。

### 3.3 Component bootstrap

* 重抽单位：26 个 component；A0/C1 三个 seed 共用同一 component 重抽序列；
* 2000 次，seed `20261103`；条件于六个冻结 checkpoint。

| unit | Edit_Gain point | 95% component CI | frac bootstrap >0 |
|---|---:|---|---:|
| A0 seed mean | +0.2796 | **[+0.1153, +0.4544]** | 0.9995 |
| C1 seed mean | +0.2690 | **[+0.0968, +0.4492]** | 0.9995 |
| A0:s0 | +0.2601 | [+0.0958, +0.4360] | 0.9995 |
| A0:s1 | +0.2879 | [+0.1227, +0.4688] | 1.0000 |
| A0:s2 | +0.2907 | [+0.1286, +0.4671] | 1.0000 |
| C1:s0 | +0.2752 | [+0.1013, +0.4494] | 0.9990 |
| C1:s1 | +0.2644 | [+0.0896, +0.4552] | 0.9995 |
| C1:s2 | +0.2674 | [+0.1053, +0.4436] | 1.0000 |

CI 很宽（component 只有 26 个），但下界仍明显大于 0。区间条件于当前模型；
训练 seed 的 gain 波动（A0 sd 0.0169，C1 sd 0.0056）远小于效应量。

### 3.4 分层

| subset | pairs | components | A0 gain range | C1 gain range | status |
|---|---:|---:|---|---|---|
| same_var_heavy | 25 | 16 | +0.247…+0.271 | +0.252…+0.274 | OK |
| diff_var_heavy | 49 | 19 | +0.209…+0.250 | +0.208…+0.233 | OK |
| mapping_unique | 12 | 5 | +0.242…+0.294 | +0.230…+0.267 | **LOW_SUPPORT** |
| mapping_not_unique | 62 | 21 | +0.265…+0.296 | +0.265…+0.277 | OK |

`same_var_heavy` 与 `diff_var_heavy` 都有正 gain；`mapping_unique` 只有 5 个
component，只报告描述性均值，不推断泛化。未拟合任何新回归/先验/残差预测器。

### 3.5 与旧 TRACE 并列，但不混淆

**旧结论**：TRACE 不能稳定预测 `e_delta`；`mean_gain_primary = +0.00147`，
pooled CI [−0.01080, +0.01135]，判定 `NO_INCREMENT_THIS_PROBE`。旧 `NATIVE`
baseline 是“令 `e_delta=0`”，其 MAE 是**预测 `e_delta` 的误差**。

**新结果**：`MODEL` 的 `Δpred` 比 `ZERO_CHANGE`（`Δpred=0`）明确更接近 `Δy`
（component bootstrap 下界 >0）；`Edit_Gain` +0.27～+0.28。

两者独立：

* 旧 `NATIVE` **不是** `ZERO_CHANGE`；旧 MAE ~0.42 与新 MAE_MODEL ~0.40
  数值接近纯属巧合，目标量不同，不能相减或互相解释。
* 模型能预测真实编辑效应，**不**证明 TRACE 的分支归因能解释它的误差；
  TRACE 失败也**不**否定模型编辑响应的任务价值。

### 3.6 限制

* Audit 已曝光（`POST_HOC_EXPOSED_EVALUATION`），不是预注册成功；
* 74 对不是独立样本，只有 26 个 component；component bootstrap 已反映这一点，
  但仍只有一个数据集、一个编辑家族；
* 方向一致率 0.70 左右，仍有约 30% 的 `|Δy|≥0.10` 对方向错误；
* 不能用本结果声称通用分子编辑预测能力；它是一个值得前瞻性检验的信号。

---

## 4. ESOL 与 NCI1 辅助诊断

### 4.1 ESOL

只使用既有 6 个 A0/C1 Test 预测；固定重原子数组。A0/C1 为三 seed 均值：

| group | n | A0 RMSE | C1 RMSE | ΔRMSE (C1−A0) | status |
|---|---:|---:|---:|---:|---|
| ≤15 | 76 | 0.4473 | 0.4583 | **+0.0110** | OK |
| 16–25 | 34 | 0.7330 | 0.6946 | **−0.0384** | OK |
| >25 | 3 | 0.8419 | 0.7826 | −0.0593 | **LOW_SUPPORT** |

* 总体 C1 略好（−0.0111 RMSE），但分组后方向相反：≤15 上 C1 更差 +0.0110，
  16–25 上 C1 更好 −0.0384；>25 只有 3 个分子，不能推断。
* 配对差的 seed 波动：≤15 ΔRMSE 为 +0.0217/−0.0047/+0.0160；
  16–25 为 −0.0540/−0.0216/−0.0397；>25 为 −0.1267/+0.0024/−0.0536。
  16–25 三 seed 同向；≤15 不是。
* 可选相似度分组（同 Morgan 工具）：最低相似度 `[0,0.30)` 只有 6 个分子
  （LOW_SUPPORT），其余组 ΔRMSE 在 −0.0134～−0.0041；不展开推断。
* 误差集中度：每个模型/seed 的绝对误差最高 10% 分子贡献了
  **47%–60% 的平方误差**、27%–32% 的绝对误差；A0 与 C1 的高误差 top-10%
  重叠 **9/12**（Jaccard 0.60）。固定 A0 top-10% 高误差子集（n=12）上
  A0 MAE 1.159 / C1 1.053，其余 101 个分子 A0 0.345 / C1 0.352。
  这只是描述性集中，不删除样本、不重算主成绩。

### 4.2 NCI1

固定 5 folds × 2 seeds × A0/C1。用冻结 5-fold（fold_seed 20261102，
outer train 内 10% 分层 Valid）只读重建，Train/Valid/Test 索引 SHA-256
与 `split_manifest.json` **全部一致**。每个 fold 用其 **Train 2959 图**的
节点数 33.3%/66.7% 分位阈值（线性插值；各 fold q33=23，q67=31–32），
把同 fold outer Test 分为 small/medium/large。

| group | n per unit range | A0 acc pooled | C1 acc pooled | Δ acc | A0对/C1错 | A0错/C1对 | C1 更低的 unit 数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 270–324 | 0.8309 | 0.8275 | **−0.0034** | 154 | 144 | 7/10 |
| medium | 241–286 | 0.7956 | 0.7922 | **−0.0035** | 155 | 146 | 4/10（5 高，1 平） |
| large | 244–309 | 0.8617 | 0.8518 | **−0.0099** | 119 | 93 | 7/10 |

* 每折/seed 的完整差异见 `analysis/nci1_paired_errors.csv` 与
  `analysis/nci1_subgroups.csv`。
* large 的负差最大（−0.0099），但 small 也有 −0.0034，medium 近似持平；
  large 上 C1 更低的 unit 是 7/10，不是 10/10。**没有“只在大图上稳定退化”的证据**；
  large 的负差主要由 fold 2 两个 seed（−0.0323/−0.0363）和 fold 3 seed0
  （−0.0287）驱动，fold 0/1/4 有正有负。
* 这些组不是独立 benchmark；同一 outer Test 图在两个 seed 下重复出现，
  不能把 10 个 unit 当作 10 倍独立样本。

---

## 5. 统计和实现审计

### 5.1 数据来源与哈希

* 所有输入、代码、输出的 SHA-256 在 `analysis/source_manifest.json`；
  共 **128** 个预测/评测记录输入（36 分子 × 3 文件 + 20 NCI1 预测）、
  **19** 个冻结报告/结构输入。
* 预测文件从上一轮 `runs/` 复制到 gitignored `_local/predictions/`；
  **没有复制 checkpoint**。MolHIV 6 个 checkpoint 的远程 SHA-256 只读核验，
  与 `checkpoint_manifest.json` 全部一致。
* 基准 commit = `6cf9ca4e…`，本轮开始时 HEAD 与该 commit 相同；
  旧 `lcrc_predictive_trace_benchmark_v1/REPORT.md` 只加了勘误指向和最小文本修正。

### 5.2 实际读取的标签来源

| 来源 | 内容 | 是否新 Test 访问 |
|---|---|---|
| 36 个分子 `test_predictions.npz` | 已保存 `y_true` | 否（上一轮已曝光） |
| 20 个 NCI1 `test_predictions.npz` | 已保存 `y_true` | 否（上一轮已曝光） |
| `probe_audit_scores.json` | `d_pred/e_delta` 派生量 | 否；未读 Audit 原始标签 |
| OGB `mapping/mol.csv.gz` | 仅 `smiles` 列 | 否；`graph-label.csv.gz` 未打开 |
| ESOL `ESOL.csv.gz` | 仅 `smiles` 列 | 否 |
| NCI1 `NCI1_graph_labels.txt` | 图标签 | 否；仅用于精确重建冻结 fold |
| 旧 `statistics.json` / `molecular_results.csv` / `tu_results.csv` | 聚合指标 | 否 |

**没有调用** `ptb_score.py`、`edit_probe.py --stage audit`、`--all-frozen`
或任何训练/推理入口；**没有 CUDA**（`CUDA_VISIBLE_DEVICES=""`）。

### 5.3 正确性门禁与测试

`analysis/correctness_gates.json`：7/7 PASS。

* 基准 commit；无 CUDA；无训练/评分入口；无新 Test 标签源；
* 36 个分子单元行/标签/预测有限性/离线指标与保存值一致（最大差 0.0）；
* 20 个 NCI1 单元 gids 对齐、accuracy 与 `tu_results.csv` 一致；
* MolHIV 官方 test row-id 逐位一致；远程 checkpoint SHA 全部一致。

`tests/`：20 passed（行对齐、rank-loss 恒等式、component 权重、
bootstrap 同步/复现、WL 支持度按分子计数、fallback 解析）。

### 5.4 CPU 时间与缺失

`analysis/runtime.json`：完整 pipeline **261.15 s**（4.35 min）CPU wall time；
**GPU 0**。分项：integrity 4.58 s、P0 104.43 s、MolHIV 149.22 s、
local-edit 0.76 s、aux 1.38 s、manifest 0.78 s。

* `MISSING_ARTIFACT`：**无**。所有需要的预测/派生文件都存在。
* 未完成项：无科学定义上的未完成项；`(0.50,1]` 支持度组和 `≤15` 重原子组
  因样本不足按预定义记 `LOW_SUPPORT`，不是缺失。
* 所有新结果只适用于当前已训练 checkpoint 与已曝光 Test/Audit；
  不能外推为独立验证或通用能力。
