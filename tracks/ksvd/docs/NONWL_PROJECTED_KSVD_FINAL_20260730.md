# 不使用 WL 的最终 KSVD 复核（2026-07-30）

> 比较 WL histogram、exact canonical adjacency、root-preserving exact canonical adjacency。三者使用相同 patches、outer folds、训练预算和下游评估；后两者都不以 WL 作为最终结构特征。

## 表示含义

- `WL`：度/度对统计与 3 轮 1-WL histogram；置换不变，但理论上有损。
- `canonical`：nauty 规范标号后的完整邻接上三角与 node mask；对无根 induced patch 在同构意义下无损。
- `rooted`：额外约束 sampler center 必须在同构下被保留，区分相同 induced graph 中不同中心角色。
- 多维启发式排序没有作为主方案：tie 最终若由 node id 打破便不置换不变；exact canonical labeling 才是严格的无 WL 对照。

所有数值均为 `stats + feature` 的 3 seeds × 5 folds balanced accuracy。

## IMDB-BINARY/cleaned/B0/max_nodes=12

图数：493。

| 方法 | WL | canonical | rooted |
|---|---:|---:|---:|
| raw | 0.7668 | 0.7440 | 0.7502 |
| PCA | 0.7549 | 0.7552 | 0.7495 |
| real-patch | 0.7544 | 0.7579 | 0.7403 |
| KSVD | 0.7479 | 0.7478 | 0.7375 |
| final-projected | 0.7536 | 0.7517 | 0.7431 |
| iterative-projected | 0.7480 | 0.7415 | 0.7265 |
| relation-only | 0.7466 | 0.7466 | 0.7466 |
| final-projected+graph | 0.7663 | 0.7545 | 0.7461 |

Projected 方法的跨表示 paired 变化：

| 比较 | 方法 | mean Δ | fold W/T/L | seed means |
|---|---|---:|---:|---|
| canonical_minus_wl | final-projected | -0.0019 | 7/0/8 | [+0.0049, -0.0122, +0.0016] |
| canonical_minus_wl | iterative-projected | -0.0065 | 6/1/8 | [+0.0145, -0.0149, -0.0189] |
| rooted_canonical_minus_wl | final-projected | -0.0105 | 5/0/10 | [-0.0086, -0.0123, -0.0105] |
| rooted_canonical_minus_wl | iterative-projected | -0.0214 | 5/0/10 | [-0.0063, -0.0293, -0.0287] |
| rooted_minus_canonical | final-projected | -0.0086 | 5/0/10 | [-0.0135, -0.0000, -0.0121] |
| rooted_minus_canonical | iterative-projected | -0.0150 | 4/0/11 | [-0.0208, -0.0144, -0.0098] |

## IMDB-MULTI/cleaned/B0/max_nodes=12

图数：321。

| 方法 | WL | canonical | rooted |
|---|---:|---:|---:|
| raw | 0.5463 | 0.4978 | 0.5128 |
| PCA | 0.5172 | 0.4768 | 0.5119 |
| real-patch | 0.5538 | 0.4894 | 0.4853 |
| KSVD | 0.5400 | 0.5091 | 0.5044 |
| final-projected | 0.5410 | 0.5485 | 0.5095 |
| iterative-projected | 0.5673 | 0.4918 | 0.5106 |
| relation-only | 0.5055 | 0.5055 | 0.5055 |
| final-projected+graph | 0.5397 | 0.5379 | 0.5097 |

Projected 方法的跨表示 paired 变化：

| 比较 | 方法 | mean Δ | fold W/T/L | seed means |
|---|---|---:|---:|---|
| canonical_minus_wl | final-projected | +0.0075 | 8/0/7 | [-0.0125, +0.0180, +0.0171] |
| canonical_minus_wl | iterative-projected | -0.0755 | 2/0/13 | [-0.0938, -0.0246, -0.1082] |
| rooted_canonical_minus_wl | final-projected | -0.0314 | 4/0/11 | [-0.0446, -0.0178, -0.0319] |
| rooted_canonical_minus_wl | iterative-projected | -0.0567 | 3/1/11 | [-0.1023, -0.0161, -0.0517] |
| rooted_minus_canonical | final-projected | -0.0390 | 4/0/11 | [-0.0320, -0.0358, -0.0490] |
| rooted_minus_canonical | iterative-projected | +0.0189 | 8/0/7 | [-0.0084, +0.0086, +0.0565] |

## REDDIT-BINARY/raw/R2/max_nodes=24

图数：2000。

| 方法 | WL | canonical | rooted |
|---|---:|---:|---:|
| raw | 0.8202 | 0.8160 | 0.8115 |
| PCA | 0.8325 | 0.8333 | 0.8362 |
| real-patch | 0.8252 | 0.8375 | 0.8362 |
| KSVD | 0.8328 | 0.8378 | 0.8402 |
| final-projected | 0.8318 | 0.8418 | 0.8327 |
| iterative-projected | 0.8333 | 0.8335 | 0.8417 |
| relation-only | 0.8550 | 0.8550 | 0.8550 |
| final-projected+graph | 0.8360 | 0.8488 | 0.8458 |

Projected 方法的跨表示 paired 变化：

| 比较 | 方法 | mean Δ | fold W/T/L | seed means |
|---|---|---:|---:|---|
| canonical_minus_wl | final-projected | +0.0100 | 9/0/6 | [+0.0040, +0.0105, +0.0155] |
| canonical_minus_wl | iterative-projected | +0.0002 | 9/1/5 | [-0.0015, +0.0030, -0.0010] |
| rooted_canonical_minus_wl | final-projected | +0.0008 | 7/0/8 | [+0.0050, -0.0045, +0.0020] |
| rooted_canonical_minus_wl | iterative-projected | +0.0083 | 11/1/3 | [+0.0130, +0.0095, +0.0025] |
| rooted_minus_canonical | final-projected | -0.0092 | 1/2/12 | [+0.0010, -0.0150, -0.0135] |
| rooted_minus_canonical | iterative-projected | +0.0082 | 9/1/5 | [+0.0145, +0.0065, +0.0035] |

## 预注册式最终判定

- `wl/final_projected` 对自身 raw、PCA、real-patch 均 3/3 seed 正向的数据集：无。
- `wl/iterative_projected` 对自身 raw、PCA、real-patch 均 3/3 seed 正向的数据集：无。
- `canonical/final_projected` 对自身 raw、PCA、real-patch 均 3/3 seed 正向的数据集：['IMDB-MULTI/cleaned/B0/max_nodes=12']。
- `canonical/iterative_projected` 对自身 raw、PCA、real-patch 均 3/3 seed 正向的数据集：无。
- `rooted_canonical/final_projected` 对自身 raw、PCA、real-patch 均 3/3 seed 正向的数据集：无。
- `rooted_canonical/iterative_projected` 对自身 raw、PCA、real-patch 均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

门槛要求同一个 projected 方法至少通过两个结构差异明显的数据集；单一数据集提升、仅优于 WL 对应实现、或不超过 relation-only 均不足以支持 KSVD 主线。
