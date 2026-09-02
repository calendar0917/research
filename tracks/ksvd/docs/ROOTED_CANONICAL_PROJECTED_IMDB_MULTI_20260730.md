# Projected KSVD 有界 Go/No-Go 实验（2026-07-30）

> 问题：把自由 KSVD atom 约束为训练折中的 distinct real patches 后，能否获得稳定、跨数据集、且不只是重构更好的任务增益？

## 协议

- outer CV：5-fold × split seeds [0, 1, 2]；inner CV：3-fold 选择 logistic C。
- atoms=16，T=2，iterations=10，每折最多 4000 个训练 patches。
- 所有 PCA/字典只在 outer-train 拟合；所有图均 structure-only。
- `final-projected`：普通 KSVD 完成后做一次 Hungarian joint projection。
- `iterative-projected`：每次 dictionary update 后投影，并重新 OMP。
- projection 使用 absolute cosine，但输出 atom 本身始终是 normalized real training patch；同时要求 vector signature distinct。
- 晋级标准：projectability=1，且至少两个结构差异明显的数据集上，跨 3 split seeds 稳定优于 clustered-real-patch、PCA、raw，并在控制 stats 后保留增益。

## IMDB-MULTI/cleaned/B0/max_nodes=12/rooted_canonical

图数：321；类别：`{'0': 144, '1': 85, '2': 92}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.5128 | +0.0119 | 9/0/6 | 2/3 |
| PCA | 0.5119 | +0.0109 | 7/0/8 | 2/3 |
| clustered real-patch | 0.4853 | -0.0156 | 6/0/9 | 1/3 |
| unconstrained KSVD | 0.5044 | +0.0035 | 7/0/8 | 1/3 |
| final-projected KSVD | 0.5095 | +0.0086 | 8/0/7 | 3/3 |
| iterative-projected KSVD | 0.5106 | +0.0097 | 9/0/6 | 2/3 |
| relation graph only | 0.5055 | +0.0045 | 9/0/6 | 2/3 |
| real-patch + graph | 0.4946 | -0.0063 | 8/0/7 | 1/3 |
| KSVD + graph | 0.5137 | +0.0127 | 8/0/7 | 2/3 |
| final-projected + graph | 0.5097 | +0.0088 | 8/0/7 | 3/3 |
| iterative-projected + graph | 0.4960 | -0.0050 | 8/0/7 | 1/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：+0.0242，fold W/T/L=9/0/6，seed means=[-0.0033, 0.0314, 0.0446]。
- `iterative_projected_minus_real`：+0.0253，fold W/T/L=12/0/3，seed means=[-0.0258, 0.0412, 0.0605]。
- `final_projected_minus_raw`：-0.0033，fold W/T/L=7/1/7，seed means=[-0.0226, 0.0087, 0.004]。
- `iterative_projected_minus_raw`：-0.0022，fold W/T/L=7/0/8，seed means=[-0.0451, 0.0185, 0.02]。
- `final_projected_minus_pca`：-0.0023，fold W/T/L=8/1/6，seed means=[-0.0343, 0.0093, 0.018]。
- `iterative_projected_minus_pca`：-0.0013，fold W/T/L=9/0/6，seed means=[-0.0568, 0.019, 0.034]。
- `final_projected_minus_unconstrained`：+0.0051，fold W/T/L=8/0/7，seed means=[0.0072, -0.0161, 0.0242]。
- `iterative_projected_minus_unconstrained`：+0.0062，fold W/T/L=11/0/4，seed means=[-0.0153, -0.0063, 0.0401]。
- `final_projected_graph_minus_real_graph`：+0.0151，fold W/T/L=9/0/6，seed means=[-0.0029, 0.0205, 0.0276]。
- `iterative_projected_graph_minus_real_graph`：+0.0013，fold W/T/L=8/0/7，seed means=[-0.0179, 0.0118, 0.0101]。
- `final_projected_graph_minus_relation_graph`：+0.0042，fold W/T/L=7/0/8，seed means=[0.0068, 0.0017, 0.0042]。
- `iterative_projected_graph_minus_relation_graph`：-0.0095，fold W/T/L=6/0/9，seed means=[-0.0082, -0.007, -0.0133]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.1217 |
| final-projected | 1.000 | 0.1515 |
| iterative-projected | 1.000 | 0.1907 |

## 预注册门槛判定

- `final_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。
- `iterative_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。
