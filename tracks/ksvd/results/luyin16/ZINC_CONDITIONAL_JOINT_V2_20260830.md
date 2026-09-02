# ZINC conditional joint v2 快筛

日期：2026-08-30  
协议：`luyin16-zinc-conditional-joint-v2-screen-v1`  
状态：no-go for full；未加载 test

## 表示与控制

每个 radius-2、全原子中心 patch 被映射为 permutation-invariant rooted signature：

```text
(n_nodes, n_edges, cycle_rank, center_degree/shell1, shell2_size)
```

signature vocabulary 只由当前训练切片的 patch 拟合，最多保留 64 类；valid 未见类型
进入 unknown。条件读出统计 signature mass、signature×atom distribution 和
signature×bond distribution。shuffle 在图内同步打乱 atom+bond columns，只破坏
signature--attribute 对应关系，保留两侧 marginals 和 patch 数量。

固定 radius、特征生成和 XGBoost；model seeds 为 `0/1/2`，不做 Optuna。先运行
两个非重叠 `2000/200` 切片，通过最低门槛后，再运行两个非重叠 `5000/500`
终止性复核。official test 始终未加载。

## 结果

MAE 越低越好；`conditional−raw gain` 为正时表示 conditional 更好。

| 阶段 | slice | global | global+raw | global+signature | global+conditional | conditional−raw gain | mean shuffle gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 2000/200 | A | 0.74307 | 0.73813 | 0.75068 | 0.76811 | -0.02998 | +0.02147 |
| 2000/200 | B | 0.60253 | 0.56485 | 0.58288 | 0.54223 | +0.02261 | +0.05307 |
| 5000/500 | A | 0.69399 | 0.67419 | 0.69087 | 0.69608 | -0.02189 | +0.01559 |
| 5000/500 | B | 0.58892 | 0.55329 | 0.57165 | 0.54532 | +0.00797 | +0.03609 |

两个大切片的三次 shuffle 间隔分别为：

- A：`0.01617 / 0.01807 / 0.01253`；
- B：`0.03777 / 0.03852 / 0.03196`。

大切片 train vocabulary 覆盖均为 `100%`，valid unknown patch rate 分别为
`0` 和 `0.000086`；条件特征维度分别为 `2015/2048`。因此失败不能归因于
valid 大量出现训练未见结构签名。

## 判定

### 机制判断：PASS

在两个 `5000/500` 切片上，true conditional 均稳定优于所有 shuffle，且间隔
大于 model-seed 波动。结构--属性对应关系确实存在可检测信息。

### 预测表示：NO-GO

预注册的大切片晋级门槛要求两片均超过 typed raw 至少 `0.01 MAE`。slice A
反而下降 `0.02189`，slice B 仅提高 `0.00797`，因此失败。不运行 10K/1K、
Optuna 或 K-SVD。

结果说明问题不再是“有没有 binding”，而是“如何提取 binding 中能跨样本切片
泛化的低维增量”。当前 2K 维条件直方图稀疏、冗余，并把稳定 marginal 信号与
条件偏差一起交给 XGBoost，容易产生切片敏感性。

## 后续唯一低成本候选

若继续该问题，只允许测试 centered conditional residual：

```text
P(signature, attribute) - P(signature)P(attribute)
```

先用 train-only truncated SVD 压到固定 16 维，再拼到 `global+typed raw`；同时
对 shuffled residual 使用相同协议。仍先做两个非重叠 `2000/200` 切片。只有两片
都相对 raw 改善 `>=0.01 MAE` 且 true 优于 shuffle，才允许扩大。若失败，则停止
ZINC 上的结构--属性统计展开，转回采样覆盖和跨 patch 长程关系问题。

该 centered residual SVD16 已完成。两个 `2000/200` 切片的 true-shuffle 平均
间隔为 `0.04408/0.02089`，但相对 typed raw 的增益为
`+0.02567/-0.00155`，严格 gate 失败，未扩大。完整结果见
[`ZINC_CONDITIONAL_RESIDUAL_SVD16_20260830.md`](ZINC_CONDITIONAL_RESIDUAL_SVD16_20260830.md)。

原始结果：

- `ZINC_CONDITIONAL_JOINT_V2_SLICE_A_20260830.json`
- `ZINC_CONDITIONAL_JOINT_V2_SLICE_B_20260830.json`
- `ZINC_CONDITIONAL_JOINT_V2_5000_SLICE_A_20260830.json`
- `ZINC_CONDITIONAL_JOINT_V2_5000_SLICE_B_20260830.json`
