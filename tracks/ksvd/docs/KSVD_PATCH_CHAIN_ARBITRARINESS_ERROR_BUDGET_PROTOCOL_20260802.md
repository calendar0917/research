# Patch-chain arbitrariness 与 reconstruction error budget 协议

> 日期：2026-08-02  
> 状态：正式结果不可见前冻结  
> 范围：无 graph labels；只研究 continuous cover、compression、stitching 与 graph completion。

## 1. 研究问题

当前 target-edge single chain 已稳定优于 local frontier，但 `patch_size=10`、`overlap=5`、budget multiplier `1.5`、`K=24,T=3` 仍是工程选择。本轮回答：

1. 当前 cover 配置是否位于 coverage–cost Pareto frontier？
2. 当前 KSVD 配置是否位于 compression–error Pareto frontier？
3. target-edge heuristic 与每一步枚举最大 marginal coverage 相差多少？
4. full reconstruction error 中有多少来自 compression、uniform stitching 和 unseen-pair zero fill？

不根据单一 test 指标事后挑冠军；报告完整 frontier 和 error decomposition。

## 2. A：cover Pareto grid

复用 72 张图：

```text
graph_bank_seed = 810001
families = regular / small_world / block
n = 50
degrees = 15 / 20 / 25
8 graphs per cell
cover_seed = 870101
```

冻结 27 个 cell：

```text
patch_size 8:  overlap 2 / 4 / 6
patch_size 10: overlap 3 / 5 / 7
patch_size 12: overlap 4 / 6 / 8
budget multiplier: 1.0 / 1.5 / 2.0
```

每个 cell 报告：

- sampler feasible graph fraction；
- mean patch count 和 raw pair slots；
- node/edge/pair coverage；
- bridge length、new pairs per patch、redundancy；
- RAW full adjacency RMSE/recall；
- graph-balanced family breakdown。

Pareto axes：minimize patch count/raw pair slots/full RMSE，maximize edge/pair coverage。连续性、target hit 和 patch connectivity 是 hard constraints，不作为可交换收益。

当前 `10/5/1.5` 只有在不被另一个 feasible cell 同时以更低或相等成本、更高或相等 coverage、且至少一项严格改善时，才算 nondominated。

## 3. B：KSVD compression Pareto

冻结当前 cover `10/5/1.5` 和同一三折。主网格：

```text
K = 16 / 24 / 32
T = 2 / 3 / 4
updates = 25
T_min = 1
PCA rank = T
```

额外 convergence probe：

```text
K=24,T=3, updates=10 / 50
```

报告：patch error、observed/full RMSE、F1/recall、disagreement、nondead atoms、maximum activation share、mean code nonzeros、dictionary scalar count、每图 code scalar count。

Compression Pareto axes：minimize observed RMSE、dictionary size 和 code nonzeros。不得仅以更大 `K/T` 的最低误差替换当前配置。

## 4. C：small-graph marginal oracle

在独立小图上比较当前 target-edge bridge 与 one-step exhaustive marginal cover：

```text
n=18
families = regular / small_world / block
degrees = 4 / 6
4 graphs per cell = 24 graphs
patch_size=6, overlap=3, multiplier=1.5
```

Exhaustive marginal branch 在每一步枚举所有满足以下条件的 next patch：

- 与 previous patch 恰好重叠 3 nodes；
- patch connected；
- size=6。

按 lexicographic objective 选择：

```text
maximize new true edges
then maximize new observed pairs
then maximize new nodes
then maximize total induced edges
```

相同 budget 下报告 target heuristic 相对 exhaustive marginal 的 edge/pair coverage gap。该 oracle 只验证局部 heuristic gap，不声称得到全链全局最优。

## 5. D：stitching 与 unseen completion

复用 `K=24,T=3,updates=25` 三折 FINAL predictions。

### Uniform stitch

当前同一 global pair 的 occurrence 直接平均。

### Train-only slot reliability stitch

在 train graphs 上估计 45 个 local pair slots 的 reconstruction MSE：

```text
weight_r = 1 / (slot_mse_r + 0.1 * global_mse)
```

test graphs 只使用冻结权重；不读 test truth。Gate：observed RMSE 至少降低 `1%`，F1/recall不坏。

### Unseen-pair completion

比较：

1. `ZERO`：当前 unseen pairs = 0；
2. `TRAIN_DENSITY`：用 train unseen-pair edge prevalence 填充；
3. `STRUCTURAL_RIDGE`：train-only ridge，`alpha=1e-2`。

Ridge feature 只由 cover 已观察到的正边构造：

```text
min/max observed degree
degree sum / absolute difference / product
observed common neighbors
observed-neighbor Jaccard
```

targets 只来自 train graphs 的 unseen pairs；test graph truth 不参与拟合。额外在每张 test graph 内 shuffle structural predictions，检查 feature binding。

分别报告：

- RAW-observed + completion；
- KSVD-uniform + completion；
- KSVD-weighted + completion。

主 gate：STRUCTURAL_RIDGE 相对 TRAIN_DENSITY 的 full adjacency RMSE 至少降低 `1%`，且优于 graph-internal shuffled predictions。

## 6. 误差分解

最终必须明确报告：

```text
coverage error     = RAW full RMSE - RAW observed RMSE
compression error  = KSVD uniform full RMSE - RAW full RMSE
stitching gain     = weighted KSVD full RMSE - uniform KSVD full RMSE
completion gain    = completed full RMSE - zero-fill full RMSE
```

其中差值只作描述，不能当作严格可加的正交 variance decomposition。

## 7. 结论标签

- `CURRENT_CONFIGURATION_PARETO_SUPPORTED`：当前 cover nondominated，KSVD 也在 compression frontier；
- `REVISE_COVER_CONFIGURATION`：存在稳定支配当前 cover 的配置；
- `REVISE_KSVD_CAPACITY`：存在更优 compression Pareto 点；
- `HEURISTIC_HAS_MATERIAL_ORACLE_GAP`：small-graph edge coverage gap >= `0.03`；
- `TRAIN_ONLY_POSTPROCESSING_REDUCES_ERROR`：registered stitch 或 completion gate 通过；
- `CURRENT_PIPELINE_ROBUST_BUT_NOT_OPTIMAL`：没有单一最终冠军，但多处存在明确 tradeoff。

这些标签可并存；本轮不强行压成单一 PASS/FAIL。
