# MolHIV 中心级结构--属性交互快筛

日期：2026-09-01  
协议：`luyin16-molhiv-cross-center-interaction-screen-v1`

## 问题

在固定的 invariant all-centre radius-2 rooted-WL patch 上，检验交互是否必须在
图级 pooling 之前形成。实验不使用 GINE、attention、K-SVD 或 official-valid/test。

比较两个交互轴：

1. `cross_cov`：同一分子不同中心之间，拓扑行与属性行的中心化协方差；
2. `binding`：每个 patch 内 role--attribute centered binding 的跨中心 mean+std。

高维交互只用每折 train 部分拟合 PCA-8。`cross_cov` 的 null 在同一图内打乱
attribute centre rows；`binding` 的 null 在每个 patch 内打乱 attribute entities。
置乱分支使用 true 分支训练得到的同一 PCA 坐标。

## 审计

- 8 图随机重标号审计通过；全部 true/null 图级特征最大漂移 `1.79e-7`；
- 三折各用 `1200/600` official-train scaffold 子样本；
- 固定 XGBoost，model seeds `0/1/2`；
- official validation/test 均未编码或评估。

## 结果

| view | mean ROC-AUC | 相对 `S+marginal` | fold wins |
|---|---:|---:|---:|
| `S` | 0.6298 | -0.0429 | 0/3 |
| `S+marginal` | 0.6726 | — | — |
| `S+cross_cov` | 0.6805 | **+0.00785** | **3/3** |
| `S+binding` | 0.6744 | +0.00175 | 3/3 |
| `S+cross_cov+binding` | **0.6844** | **+0.01181** | **3/3** |
| fixed 50/50 marginal/binding late fusion | 0.6747 | +0.00211 | 3/3 |

机制对照：

- `cross_cov − centre-shuffle = +0.00491`，2/3 folds；
- `binding − patch-shuffle = -0.00195`，1/3 folds；
- `cross_cov+binding` 的增益主要由 cross-centre interaction 驱动，不能归因于
  patch 内精确 role--attribute binding。

## 阶段判断

本轮给出了一个值得确认的新信号：

> 对当前 radius-2 局部对象，有用的结构--属性交互更可能是一个分子内部
> **不同中心环境如何共同变化**，而不是单个 patch 内属性落在精确结构角色上的关系。

这与此前结果一致：跨中心 `std` 是强 readout，而 patch 内 binding/cross-attention
跨 scaffold 不稳定。现在可以把“环境异质性”从各维独立的 std 扩展为
结构--属性之间的联合异质性。

但当前只属于 **feasibility pass**：

- `cross_cov` 对 shuffle 的三个 fold 中有一个为负；
- 固定参数绝对 AUC 较低，不能与 full official-valid 数字横比；
- `both` 尚无 matched joint-null，因此只能证明预测增量，不能把全部增量归因于
  两种真实对应同时成立；
- 尚未确认 PCA rank、XGBoost seed 和样本切片稳定性。

## 下一步边界

允许一次确认实验：

1. 保持同一表示和 PCA-8，不扫描 rank/role bins；
2. 在更大的 official-train scaffold folds 或第二组非重叠子样本上复核
   `marginal / cross_cov / both / centre-shuffle`；
3. 增加 `both` 的 matched double-shuffle control；
4. 只有 `cross_cov` 同时稳定超过 marginal 和 shuffle，才进入完整 train-fold
   等预算 XGBoost 搜索；
5. patch 内 binding 不单独晋级，保留为 both 分支中的辅助块；
6. 不引入 GINE、attention、预训练或 K-SVD update。

原始结果：

- [`cross_center_interaction_screen/summary.md`](cross_center_interaction_screen/summary.md)
- [`cross_center_interaction_screen/summary.json`](cross_center_interaction_screen/summary.json)
- runner：`tracks/ksvd/experiments/luyin16/cross_center_interaction_screen.py`
- config：`tracks/ksvd/configs/luyin16/cross_center_interaction_screen.yaml`
