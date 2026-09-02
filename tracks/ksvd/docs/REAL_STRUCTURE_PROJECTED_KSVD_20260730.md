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

## IMDB-BINARY/cleaned/B0/max_nodes=12

图数：493；类别：`{'0': 261, '1': 232}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.7668 | +0.0135 | 8/0/7 | 3/3 |
| PCA | 0.7549 | +0.0016 | 6/1/8 | 2/3 |
| clustered real-patch | 0.7544 | +0.0011 | 8/0/7 | 1/3 |
| unconstrained KSVD | 0.7479 | -0.0055 | 7/0/8 | 1/3 |
| final-projected KSVD | 0.7536 | +0.0002 | 10/0/5 | 2/3 |
| iterative-projected KSVD | 0.7480 | -0.0054 | 7/0/8 | 1/3 |
| relation graph only | 0.7466 | -0.0067 | 5/0/10 | 1/3 |
| real-patch + graph | 0.7590 | +0.0057 | 9/1/5 | 2/3 |
| KSVD + graph | 0.7600 | +0.0067 | 9/0/6 | 3/3 |
| final-projected + graph | 0.7663 | +0.0130 | 11/0/4 | 2/3 |
| iterative-projected + graph | 0.7509 | -0.0025 | 5/0/10 | 1/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：-0.0009，fold W/T/L=7/1/7，seed means=[-0.0088, -0.0024, 0.0086]。
- `iterative_projected_minus_real`：-0.0064，fold W/T/L=7/2/6，seed means=[-0.015, -0.0144, 0.0101]。
- `final_projected_minus_raw`：-0.0133，fold W/T/L=4/1/10，seed means=[-0.0179, -0.0096, -0.0123]。
- `iterative_projected_minus_raw`：-0.0189，fold W/T/L=5/1/9，seed means=[-0.0241, -0.0217, -0.0108]。
- `final_projected_minus_pca`：-0.0013，fold W/T/L=9/1/5，seed means=[0.0054, -0.01, 0.0006]。
- `iterative_projected_minus_pca`：-0.0069，fold W/T/L=4/2/9，seed means=[-0.0008, -0.0221, 0.0021]。
- `final_projected_minus_unconstrained`：+0.0057，fold W/T/L=7/3/5，seed means=[0.0162, 0.0064, -0.0056]。
- `iterative_projected_minus_unconstrained`：+0.0001，fold W/T/L=6/1/8，seed means=[0.0101, -0.0057, -0.0041]。
- `final_projected_graph_minus_real_graph`：+0.0073，fold W/T/L=7/3/5，seed means=[-0.0005, 0.0032, 0.0191]。
- `iterative_projected_graph_minus_real_graph`：-0.0081，fold W/T/L=5/1/9，seed means=[-0.0037, -0.0211, 0.0005]。
- `final_projected_graph_minus_relation_graph`：+0.0197，fold W/T/L=12/0/3，seed means=[0.0276, 0.0044, 0.027]。
- `iterative_projected_graph_minus_relation_graph`：+0.0043，fold W/T/L=8/1/6，seed means=[0.0244, -0.02, 0.0083]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.2939 |
| final-projected | 1.000 | 0.3074 |
| iterative-projected | 1.000 | 0.3139 |

## IMDB-MULTI/cleaned/B0/max_nodes=12

图数：321；类别：`{'0': 144, '1': 85, '2': 92}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.5463 | +0.0454 | 10/0/5 | 3/3 |
| PCA | 0.5172 | +0.0163 | 10/0/5 | 3/3 |
| clustered real-patch | 0.5538 | +0.0529 | 10/0/5 | 3/3 |
| unconstrained KSVD | 0.5400 | +0.0391 | 10/0/5 | 3/3 |
| final-projected KSVD | 0.5410 | +0.0400 | 11/0/4 | 3/3 |
| iterative-projected KSVD | 0.5673 | +0.0664 | 11/0/4 | 3/3 |
| relation graph only | 0.5055 | +0.0045 | 9/0/6 | 2/3 |
| real-patch + graph | 0.5518 | +0.0508 | 10/0/5 | 3/3 |
| KSVD + graph | 0.5570 | +0.0561 | 12/0/3 | 3/3 |
| final-projected + graph | 0.5397 | +0.0388 | 11/0/4 | 3/3 |
| iterative-projected + graph | 0.5685 | +0.0676 | 11/0/4 | 3/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：-0.0128，fold W/T/L=5/0/10，seed means=[-0.0304, -0.021, 0.0129]。
- `iterative_projected_minus_real`：+0.0135，fold W/T/L=8/0/7，seed means=[0.0047, -0.0129, 0.0487]。
- `final_projected_minus_raw`：-0.0054，fold W/T/L=6/0/9，seed means=[-0.005, -0.0303, 0.0193]。
- `iterative_projected_minus_raw`：+0.0210，fold W/T/L=11/0/4，seed means=[0.0302, -0.0223, 0.055]。
- `final_projected_minus_pca`：+0.0238，fold W/T/L=9/0/6，seed means=[0.0368, 0.0069, 0.0276]。
- `iterative_projected_minus_pca`：+0.0501，fold W/T/L=11/0/4，seed means=[0.072, 0.0149, 0.0634]。
- `final_projected_minus_unconstrained`：+0.0009，fold W/T/L=7/0/8，seed means=[0.0035, -0.0163, 0.0155]。
- `iterative_projected_minus_unconstrained`：+0.0272，fold W/T/L=11/1/3，seed means=[0.0387, -0.0082, 0.0513]。
- `final_projected_graph_minus_real_graph`：-0.0120，fold W/T/L=5/0/10，seed means=[0.01, -0.0508, 0.0047]。
- `iterative_projected_graph_minus_real_graph`：+0.0167，fold W/T/L=9/0/6，seed means=[0.0347, -0.0307, 0.0462]。
- `final_projected_graph_minus_relation_graph`：+0.0342，fold W/T/L=10/0/5，seed means=[0.0577, 0.0208, 0.0243]。
- `iterative_projected_graph_minus_relation_graph`：+0.0630，fold W/T/L=9/0/6，seed means=[0.0824, 0.041, 0.0658]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.3119 |
| final-projected | 1.000 | 0.3280 |
| iterative-projected | 1.000 | 0.3327 |

## REDDIT-BINARY/raw/R2/max_nodes=24

图数：2000；类别：`{'0': 1000, '1': 1000}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.8202 | -0.0230 | 0/0/15 | 0/3 |
| PCA | 0.8325 | -0.0107 | 2/0/13 | 0/3 |
| clustered real-patch | 0.8252 | -0.0180 | 2/0/13 | 0/3 |
| unconstrained KSVD | 0.8328 | -0.0103 | 4/1/10 | 0/3 |
| final-projected KSVD | 0.8318 | -0.0113 | 4/0/11 | 0/3 |
| iterative-projected KSVD | 0.8333 | -0.0098 | 3/0/12 | 1/3 |
| relation graph only | 0.8550 | +0.0118 | 12/0/3 | 3/3 |
| real-patch + graph | 0.8372 | -0.0060 | 3/1/11 | 0/3 |
| KSVD + graph | 0.8480 | +0.0048 | 10/1/4 | 2/3 |
| final-projected + graph | 0.8360 | -0.0072 | 5/0/10 | 0/3 |
| iterative-projected + graph | 0.8403 | -0.0028 | 7/0/8 | 1/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：+0.0067，fold W/T/L=11/1/3，seed means=[0.0005, 0.019, 0.0005]。
- `iterative_projected_minus_real`：+0.0082，fold W/T/L=9/0/6，seed means=[-0.0015, 0.0125, 0.0135]。
- `final_projected_minus_raw`：+0.0117，fold W/T/L=11/3/1，seed means=[0.013, 0.016, 0.006]。
- `iterative_projected_minus_raw`：+0.0132，fold W/T/L=12/0/3，seed means=[0.011, 0.0095, 0.019]。
- `final_projected_minus_pca`：-0.0007，fold W/T/L=8/1/6，seed means=[0.006, 0.002, -0.01]。
- `iterative_projected_minus_pca`：+0.0008，fold W/T/L=5/3/7，seed means=[0.004, -0.0045, 0.003]。
- `final_projected_minus_unconstrained`：-0.0010，fold W/T/L=8/0/7，seed means=[-0.008, 0.011, -0.006]。
- `iterative_projected_minus_unconstrained`：+0.0005，fold W/T/L=7/1/7，seed means=[-0.01, 0.0045, 0.007]。
- `final_projected_graph_minus_real_graph`：-0.0012，fold W/T/L=6/1/8，seed means=[-0.004, 0.0035, -0.003]。
- `iterative_projected_graph_minus_real_graph`：+0.0032，fold W/T/L=10/0/5，seed means=[-0.006, 0.008, 0.0075]。
- `final_projected_graph_minus_relation_graph`：-0.0190，fold W/T/L=1/0/14，seed means=[-0.0185, -0.0235, -0.015]。
- `iterative_projected_graph_minus_relation_graph`：-0.0147，fold W/T/L=2/0/13，seed means=[-0.0205, -0.019, -0.0045]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.6215 |
| final-projected | 1.000 | 0.6883 |
| iterative-projected | 1.000 | 0.6998 |

## 预注册门槛判定

- `final_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。
- `iterative_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。
