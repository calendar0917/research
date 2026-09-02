# Exact canonical adjacency vs WL：Projected-KSVD 配对复核（2026-07-30）

> 唯一改变是 patch 向量化：从 WL histogram 改为 nauty canonical labeling 后的完整邻接上三角 + node mask。采样、outer split、字典规模、稀疏度、训练迭代和分类协议保持一致。

## 判读问题

1. 若 canonical 显著改善同一种 KSVD 方法，说明 WL 的表示几何可能限制了字典学习，即使观察到的真实 patches 没有 WL type collision。
2. 若 canonical 对 raw、PCA 和 KSVD 都不改善，或 KSVD 没有相对 canonical 自身 baseline 获得稳定优势，则此前 no-go 不能归因于 WL 信息损失。
3. `relation graph only` 不依赖 patch 向量，理论上应保持不变；它用于检查两次实验的 split 与评估是否对齐。

## IMDB-BINARY/cleaned/B0/max_nodes=12

图数：493。

| 方法 | WL | canonical | canonical − WL | fold W/T/L | seed means |
|---|---:|---:|---:|---:|---|
| raw mean | 0.7668 | 0.7440 | -0.0228 | 4/1/10 | [-0.0134, -0.0366, -0.0185] |
| PCA | 0.7549 | 0.7552 | +0.0003 | 8/0/7 | [+0.0071, -0.0139, +0.0077] |
| clustered real-patch | 0.7544 | 0.7579 | +0.0034 | 6/0/9 | [-0.0090, -0.0152, +0.0345] |
| unconstrained KSVD | 0.7479 | 0.7478 | -0.0000 | 9/0/6 | [+0.0199, -0.0091, -0.0110] |
| final-projected KSVD | 0.7536 | 0.7517 | -0.0019 | 7/0/8 | [+0.0049, -0.0122, +0.0016] |
| iterative-projected KSVD | 0.7480 | 0.7415 | -0.0065 | 6/1/8 | [+0.0145, -0.0149, -0.0189] |
| relation graph only (unchanged control) | 0.7466 | 0.7466 | +0.0000 | 0/15/0 | [+0.0000, +0.0000, +0.0000] |
| real-patch + relation graph | 0.7590 | 0.7400 | -0.0190 | 5/0/10 | [-0.0209, -0.0333, -0.0028] |
| KSVD + relation graph | 0.7600 | 0.7423 | -0.0177 | 4/0/11 | [-0.0152, -0.0154, -0.0226] |
| final-projected + relation graph | 0.7663 | 0.7545 | -0.0118 | 4/0/11 | [+0.0032, -0.0196, -0.0190] |
| iterative-projected + relation graph | 0.7509 | 0.7433 | -0.0076 | 7/1/7 | [+0.0128, -0.0129, -0.0227] |

## IMDB-MULTI/cleaned/B0/max_nodes=12

图数：321。

| 方法 | WL | canonical | canonical − WL | fold W/T/L | seed means |
|---|---:|---:|---:|---:|---|
| raw mean | 0.5463 | 0.4978 | -0.0485 | 4/0/11 | [-0.0476, -0.0601, -0.0380] |
| PCA | 0.5172 | 0.4768 | -0.0404 | 2/0/13 | [-0.0347, -0.0275, -0.0589] |
| clustered real-patch | 0.5538 | 0.4894 | -0.0644 | 3/0/12 | [-0.0590, -0.0708, -0.0633] |
| unconstrained KSVD | 0.5400 | 0.5091 | -0.0309 | 4/0/11 | [-0.0838, -0.0132, +0.0042] |
| final-projected KSVD | 0.5410 | 0.5485 | +0.0075 | 8/0/7 | [-0.0125, +0.0180, +0.0171] |
| iterative-projected KSVD | 0.5673 | 0.4918 | -0.0755 | 2/0/13 | [-0.0938, -0.0246, -0.1082] |
| relation graph only (unchanged control) | 0.5055 | 0.5055 | +0.0000 | 0/15/0 | [+0.0000, +0.0000, +0.0000] |
| real-patch + relation graph | 0.5518 | 0.4886 | -0.0631 | 3/0/12 | [-0.0618, -0.0800, -0.0476] |
| KSVD + relation graph | 0.5570 | 0.5264 | -0.0306 | 4/0/11 | [-0.0597, -0.0387, +0.0065] |
| final-projected + relation graph | 0.5397 | 0.5379 | -0.0018 | 8/0/7 | [-0.0413, +0.0252, +0.0108] |
| iterative-projected + relation graph | 0.5685 | 0.5154 | -0.0531 | 4/0/11 | [-0.0693, -0.0075, -0.0825] |

## REDDIT-BINARY/raw/R2/max_nodes=24

图数：2000。

| 方法 | WL | canonical | canonical − WL | fold W/T/L | seed means |
|---|---:|---:|---:|---:|---|
| raw mean | 0.8202 | 0.8160 | -0.0042 | 5/1/9 | [+0.0055, -0.0075, -0.0105] |
| PCA | 0.8325 | 0.8333 | +0.0008 | 8/0/7 | [+0.0130, -0.0075, -0.0030] |
| clustered real-patch | 0.8252 | 0.8375 | +0.0123 | 11/1/3 | [+0.0075, +0.0200, +0.0095] |
| unconstrained KSVD | 0.8328 | 0.8378 | +0.0050 | 9/1/5 | [-0.0040, +0.0185, +0.0005] |
| final-projected KSVD | 0.8318 | 0.8418 | +0.0100 | 9/0/6 | [+0.0040, +0.0105, +0.0155] |
| iterative-projected KSVD | 0.8333 | 0.8335 | +0.0002 | 9/1/5 | [-0.0015, +0.0030, -0.0010] |
| relation graph only (unchanged control) | 0.8550 | 0.8550 | +0.0000 | 0/15/0 | [+0.0000, +0.0000, +0.0000] |
| real-patch + relation graph | 0.8372 | 0.8482 | +0.0110 | 11/1/3 | [+0.0110, +0.0165, +0.0055] |
| KSVD + relation graph | 0.8480 | 0.8462 | -0.0018 | 5/1/9 | [-0.0055, +0.0050, -0.0050] |
| final-projected + relation graph | 0.8360 | 0.8488 | +0.0128 | 12/1/2 | [+0.0015, +0.0115, +0.0255] |
| iterative-projected + relation graph | 0.8403 | 0.8478 | +0.0075 | 10/2/3 | [+0.0105, +0.0090, +0.0030] |

## 跨表示结论

- REDDIT-BINARY/raw/R2/max_nodes=24 的 final-projected KSVD 在 3/3 seeds 优于 WL 对应方法，平均变化 +0.0100。

最终是否值得继续 rooted canonical，除跨表示变化外，还必须结合 canonical 报告中相对 raw/PCA/real-patch baseline 的结果判断。
