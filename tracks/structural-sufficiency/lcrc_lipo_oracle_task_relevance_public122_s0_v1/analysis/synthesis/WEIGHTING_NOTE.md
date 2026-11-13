# WEIGHTING_NOTE — 统计口径说明（加权、pooled、分母、证据层次）

本文件只做一件事：把本轮（以及被引用的旧轮次）里**混在一起的几种权重和分母**写清楚，
并把新汇总表 `analysis/weighted_response.csv` 的读法固定下来。

**旧文件一个都没改。** `partB_*_metrics.json`、`partB_*_pairs.csv`、`summary_tables.json`
的字节与 revision `b1688975` 一致；新增的只有 `weighted_response.{csv,json}` 与本目录下的文档。

生成命令（纯 CPU，读已提交的 pair 预测，不加载模型）：

```bash
uv run python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/weighted_response.py
# 或
uv run python tracks/structural-sufficiency/lcrc_lipo_oracle_task_relevance_public122_s0_v1/code/run.py --stages weighted
```

---

## 1. 两种权重，回答两个不同问题

### 1.1 层级权重（主量）

`molecule → instance(center,target,control) → control/donor`：

1. 每个 pair 上先把 T 与 K0 两个视图平均：$a_{\text{pair}}=\tfrac12(|Δp_T|+|Δp_{K_0}|)$；
2. 同一 instance 内平均；
3. 同一分子内平均；
4. 最后对**分子**取无权平均。

实现就是被冻结的 `analysis.molecule_means`（源轮 `lcrc_lipo_multi_motif_mechanism_public122_v1/code/analysis.py:280`），
`delta_mae` / `gain` / `baseline_zero_change` / 全部 bootstrap 区间都用它。
这样每个分子对点估计的贡献相同，不会因为某分子被切出更多镜像或更多焦点对而被放大。

### 1.2 平铺（pair/view）权重

`abs_delta_pred.{mean,median,p95,max}` 是把所有被选 pair 的 T 与 K0 视图**摊平成一个数组**
再算的分位数描述：每个 (pair, view) 算一个观测。
它回答「模型在这些视图上的典型移动量是多少」，**不是**族层面的效应量，
也不等于 `delta_mae`。

### 1.3 二者不同，而且差值不小

以 `molwt` / Test / amide 为例：`delta_mae` = 1.3951（层级），同名的平铺 `mean` = 1.4205。
差 1.8%，来自「每个分子的 pair 数不同」。族内差异最大的是 sulfonamide
（层级 1.9214 vs 平铺 1.8490，符号都为正，但排序不同）。

**因此**：正文若写「平均响应」，必须声明用的是哪一种。本文件与 `weighted_response.csv`
统一用**分子权重**，并把平铺值并列保留、显式标为 pair/view 加权描述。

---

## 2. `weighted_response.csv` 的读法

每个 cell = (task, split, family)。另有每个 task/split 的 `POOLED` 行。

| 列 | 含义 |
|---|---|
| `abs_delta_pred_mean_mol` / `_median` / `_p95` / `_max` | 分子权重描述：先按 §1.1 聚合到分子，再在**分子层**取均值/中位数/95 分位/最大 |
| `abs_delta_pred_mean_mol_over_target_std` | 除以该任务 Train 目标 std（WC_LOGP 1.3238；MolWt 108.294；logD 见 checkpoint） |
| `abs_delta_pred_mean_mol_ci95_{low,high}` | 用**同一套**冻结 scaffold bootstrap（seed 20261114，5000 次）重算的区间；不是新设计 |
| `baseline_mean_mol` | 分子权重的 `mean|Δref|`（MolWt 恒为 0） |
| `delta_mae_mol` | 分子权重的 `mean|Δp−Δref|`；由 `molecule_means` 重算，与 `partB_*_metrics.json::family_stats.*.delta_mae` **逐位一致（最大差 0.0）** |
| `abs_delta_pred_mean_pairview` 等 | 平铺口径，与 `partB_*_metrics.json::abs_delta_pred` 逐位一致（最大差 0.0），用于对照 |
| `abs_delta_pred_mean_{T,K}_pairview` | 只取 T（或只取 K0）视图的平铺均值。**旧 REPORT 的 pooled 数就是 T 视图这一个** |
| `check_delta_mae_vs_committed` / `check_molwt_identity` | 两个算术核对，见 §3 |
| `worst_view` / `worst_pair_id` / `worst_row_id` / `worst_abs_dpred` | 该 cell 内最大的单视图移动，**保留在统计内**，不剔除 |
| `notes` | 该 cell 的人群/去重/单位/特例说明 |

Test 的 12 个 cell（split = test，main CF，`FULL`）：

| task | family | 对 / 分子 | mean_mol | median_mol | p95_mol | max_mol | mean_pairview | baseline_mol |
|---|---|---|---|---|---|---|---|---|
| wc_logp | amide | 533 / 164 | 0.1630 | 0.1585 | 0.2901 | 0.3932 | 0.1725 | 0.158 |
| wc_logp | sulfonamide | 112 / 30 | 0.1874 | 0.1781 | 0.3484 | 0.4523 | 0.2023 | 0.178 |
| wc_logp | aryl_ether | 112 / 58 | 0.2275 | 0.2374 | 0.3755 | 0.4120 | 0.2180 | 0.185 |
| wc_logp | arylamine | 41 / 17 | 0.3542 | 0.3418 | 0.4635 | 0.4724 | 0.3752 | 0.378 |
| wc_logp | ester | 46 / 8 | 0.1486 | 0.1195 | 0.2390 | 0.2607 | 0.1545 | 0.136 |
| wc_logp | POOLED | 844 / 219 | 0.1846 | 0.1683 | 0.3620 | 0.4724 | 0.1914 | 0.172 |
| molwt | amide | 533 / 164 | 1.3951 | 1.3394 | 2.4724 | 3.3195 | 1.4205 | 0.000 |
| molwt | sulfonamide | 112 / 30 | 1.9214 | 1.6331 | 2.9116 | 8.6163 | 1.8490 | 0.000 |
| molwt | aryl_ether | 112 / 58 | 0.9153 | 0.8603 | 2.1282 | 3.1539 | 0.9065 | 0.000 |
| molwt | arylamine | 41 / 17 | 1.0042 | 0.9386 | 1.8361 | 2.2671 | 0.9994 | 0.000 |
| molwt | ester | 46 / 8 | 0.9045 | 0.8532 | 1.3120 | 1.3862 | 0.9747 | 0.000 |
| molwt | POOLED | 844 / 219 | 1.3293 | 1.2464 | 2.4025 | 8.6163 | 1.3644 | 0.000 |

（molwt 单位 g/mol，σ_train = **108.294**；wc_logp 单位 logP，σ_train = 1.3238。）

---

## 3. 两个算术核对（这是本文件的核心，不是装饰）

### 3.1 MolWt：`Δref ≡ 0` ⇒ `delta-MAE` 必须等于 `mean|Δp|`

因为 `Δref = 0`，`|Δp − Δref| = |Δp|`，所以两种口径下
`delta_mae` 与 `mean|Δp|` 是**同一个算术量**。

结果：`weighted_response.json::checks.molwt_identity_all_cells = true`，
最大差 **0.0**（24 个 cell 全过）。`wc_logp` 上两者不等（`wc_logp_identity_holds = false`），
这正是应有的样子——如果它也相等，说明 `Δref` 被算成了 0。

这条核对同时验证了三件事：`Δref` 确实全为 0、`delta_mae` 确实用的是层级权重、
新表没有在聚合时悄悄换权重。

### 3.2 与已提交数字逐位对齐

- `delta_mae_max_abs_gap_vs_committed = 0.0`（24 个 cell）
- `delta_mae_ci95_reproduced_max_abs_gap = 0.0`（说明复用的 bootstrap 与 `evaluate.py` 是同一套）
- `flat_block_reproduced_max_abs_gap = 0.0`（平铺 mean/median/p95/max 全部复现）

因此本表可以安全地**替换正文里的平均响应表述**，而不会与已提交的 JSON 冲突。

---

## 4. 分母与证据层次

| 数字 | 它是什么 | 它不是什么 |
|---|---|---|
| **1810** | Valid + Test 的 **main CF 对数**（Valid 966 + Test 844） | **不是** 1810 个 Test 分子 |
| **844** | Test 主队列对数（amide 533 + sulfonamide 112 + aryl_ether 112 + arylamine 41 + ester 46） | 分子**跨家族重叠**，各家族分子数不能相加当总数 |
| **4200 / 3360-420-420** | 分子总数与官方 split | 不是 3360 个训练对 |
| **539 / 364** | C 链接数 / C 唯一视图数（全家族） | 不是 539 个分子 |
| **62 / 164、110 / 533** | amide/Test 校准后 分子 / 主队列 | 覆盖 ≈ **38% 分子、21% 对**，两者不能互换 |
| **2516 / 3617（69.6%）** | WC_LOGP 参照值在改线上确实变化的比例 | 不是「模型正确率」 |

`POOLED` 行 = 该 split 全部五个家族的 main CF，**每个分子只算一次**（其多个家族实例先合并），
所以它**不是**家族均值的平均，也不能与家族均值直接比大小。正文优先逐家族表。

---

## 5. 正文引用规则（照抄即可）

1. **平均响应**：写「分子权重平均 |Δpred|」并给 `weighted_response.csv` 的列名；
   若要引用平铺值，必须写成「pair/view 加权描述」。
2. **MolWt 平均响应**：用分子权重，且同时给 `σ_train = 108.294`。
   已提交的 `mean |Δp|` 逐家族 0.9045–1.9214 g/mol = **0.0084–0.0177 σ**；
   Test POOLED = 1.3293 g/mol = **0.01228 σ**。
   旧 REPORT 的 1.81 g/mol 是 **T 视图 + pair 加权**，不要与新值混写。
3. **中位/95 分位/最大值**：若保留平铺描述，必须写明「pair/view 加权」。
   分子权重下 Test POOLED 的 max = 8.6163 g/mol（分子层），单视图 max = 14.2353 g/mol。
4. **WC_LOGP pooled**：旧文 0.246 / 0.241 / 0.072 是 T 视图 + pair 加权；
   T+K 平均为 0.1914 / 0.1845 / 0.0560；分子权重为 0.1846 / 0.1723 / 0.0557
   （三者顺序：mean|dp| / mean|dref| / mean|dp-dref|）。
   三者不可互换，正文只用最后一套并注明。
5. **方向一致率**：只在 `|Δref| > 1e-8` 上计算（零变化视图另计 `n_views_zero_ref`）；
   `q = |Δ_T|−|Δ_K0|` 的一致率**另列**，且不要求任何家族 T>K0。
6. **Bonferroni**：6 项（5 家族 × 1 个 gain），磺酸酯 NA，乘子不缩减。
7. **MolWt 不写「不变」**：只写「小但非零」，并保留尾部/最坏个例。
   不从零均值或不显著推等价。
8. **WC_LOGP 不是新测量**：它是计算估计函数值；计算任务的正确性不能追认实验 logD。
9. **实验 logD 强主张**只来自校准酰胺子群（62/164 分子、110/533 对）。
