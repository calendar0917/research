# ZINC mechanism screen：统计绑定与 patch 关系

日期：2026-08-30  
协议：`luyin16-zinc-mechanism-screen-v1`  
状态：screen-only；未加载 test

## 问题与协议

在官方 PyG ZINC-12K 的 train/valid 内，固定 radius-2、全原子中心 patch、
`max_nodes=12` 和固定 XGBoost，快速检查：

1. typed marginal readout 是否稳定补充 global statistics；
2. topology--attribute joint statistics 是否含真实绑定信号；
3. 粗粒度 patch-relation side channel 是否稳定增加预测信息。

使用两个互不重叠的连续切片复验：

- slice A：train `[0:5000]`，valid `[0:500]`；
- slice B：train `[5000:10000]`，valid `[500:1000]`；
- 每个视图固定 XGBoost，model seeds `0/1/2`；
- joint control 在每个图内同步打乱 attribute columns，保留 topology、
  attribute marginals 和 patch 数量；
- 不做 Optuna，不根据 test 选择视图。

两个 valid 半区的目标分布尺度不同，slice A 还包含更极端的负目标值；因此只比较
同一切片内的 MAE 差值，不横向比较两个切片的绝对 MAE。

## 结果

MAE 越低越好。

| view / delta | slice A | slice B |
|---|---:|---:|
| global | 0.69399 | 0.58892 |
| global + typed raw | 0.67419 | 0.55329 |
| global + joint v1 | 0.69203 | 0.55738 |
| global + relation | 0.66805 | 0.55286 |
| global + typed raw + relation | 0.64894 | 0.54751 |
| typed raw 对 global 的增益 | +0.01979 | +0.03564 |
| raw+relation 对 raw 的增益 | +0.02525 | +0.00578 |
| joint 相对 3 次 shuffle 的平均间隔 | +0.02195 | +0.01914 |
| joint 相对 shuffle 的最小间隔 | +0.01369 | +0.01361 |

`+` 表示 MAE 降低。关键视图的 model-seed 标准差约为 `0.002--0.005`，
小于 joint-shuffle 间隔。

原始结果：

- `ZINC_MECHANISM_SCREEN_RADIUS2_5000_20260830.json`
- `ZINC_MECHANISM_SCREEN_RADIUS2_OFFSET5000_20260830.json`

早期 JSON 中的 `gain_relation_over_global_plus_raw` 字段曾误用
`global+relation`；上表直接由各 view score 重新计算。runner 已修正为
`gain_raw_plus_relation_over_raw`，历史原始结果不回写。

## 判定

### 1. typed marginal：保留

`global+typed raw` 在两个切片都优于 global，说明全中心 radius-2 的局部 typed
分布不是单次 prefix 偶然。它仍是当前低成本主 baseline。

### 2. structure--attribute binding：存在，但 joint v1 不晋级 full

joint v1 在两个切片均稳定优于所有 shuffle，说明 patch 内的结构--属性对应关系
确实携带信息。然而 joint v1 只用 patch 边数的 9 个 bin 表示 topology；它在两个
切片都没有超过 typed raw。因此当前证据支持“绑定信息存在”，不支持“现有 joint
readout 已是更好的最终特征”。

### 3. patch relation：当前版本不晋级

预注册门槛是两个独立切片均令 `global+raw+relation` 相对 `global+raw` 改善至少
`0.01 MAE`。slice B 只有 `0.00578`，因此失败。粗 relation 可以作为诊断量，
但不值得直接做 10K/full 或超参数搜索。

### 4. K-SVD：本轮不进入

本轮先验证统计对象和绑定控制。由于当前 joint/relation 读出尚未稳定超过 raw，
此时加入 K-SVD 只会混入压缩与优化变量，不能澄清增益来源。既有 ZINC factorial
也已表明 FINAL 不稳定优于 INIT。

## 下一步：只做一个分层快筛

候选为 `conditional joint v2`：用 permutation-invariant patch 结构签名替代单一
边数 bin，例如 `n_nodes`、`n_edges`、cycle rank、中心度、radius-1/2 shell 大小，
再统计这些结构条件下的 atom/bond 分布。

执行门槛：

1. 先在两个 `2000/200` 非重叠切片运行固定模型；
2. 只有两片的 joint-shuffle 平均间隔均 `>=0.01`，且至少一片超过 typed raw，
   才扩大到两个 `5000/500` 切片；
3. 只有两个大切片都超过 typed raw `>=0.01 MAE`，才进入 10K/1K official-valid；
4. test 继续冻结；不同时扫描 K/T、Optuna 或融合模型。

这一路径把问题收窄为“结构条件下统计什么属性”，并以最少训练判断统计对象是否
值得进入后续 K-SVD/字典表示。

## conditional joint v2 后续结果

v2 已按上述分层门槛完成。两个 `5000/500` 切片的 true conditional 均稳定优于
shuffle，确认结构--属性 binding；但相对 typed raw 的增益为
`-0.02189/+0.00797`，未通过两片均 `>=0.01` 的预测门槛。因此 v2 不进入
10K/1K、Optuna 或 K-SVD。完整结果见
[`ZINC_CONDITIONAL_JOINT_V2_20260830.md`](ZINC_CONDITIONAL_JOINT_V2_20260830.md)。
