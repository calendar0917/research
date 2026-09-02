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

## REDDIT-BINARY/raw/R2/max_nodes=24/rooted_canonical

图数：2000；类别：`{'0': 1000, '1': 1000}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.8115 | -0.0317 | 1/1/13 | 0/3 |
| PCA | 0.8362 | -0.0070 | 5/2/8 | 0/3 |
| clustered real-patch | 0.8362 | -0.0070 | 3/0/12 | 0/3 |
| unconstrained KSVD | 0.8402 | -0.0030 | 5/1/9 | 0/3 |
| final-projected KSVD | 0.8327 | -0.0105 | 2/1/12 | 0/3 |
| iterative-projected KSVD | 0.8417 | -0.0015 | 6/0/9 | 1/3 |
| relation graph only | 0.8550 | +0.0118 | 12/0/3 | 3/3 |
| real-patch + graph | 0.8508 | +0.0077 | 11/1/3 | 2/3 |
| KSVD + graph | 0.8462 | +0.0030 | 9/1/5 | 2/3 |
| final-projected + graph | 0.8458 | +0.0027 | 9/0/6 | 2/3 |
| iterative-projected + graph | 0.8528 | +0.0097 | 10/0/5 | 3/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：-0.0035，fold W/T/L=5/1/9，seed means=[-0.0055, 0.0005, -0.0055]。
- `iterative_projected_minus_real`：+0.0055，fold W/T/L=9/1/5，seed means=[0.0005, 0.008, 0.008]。
- `final_projected_minus_raw`：+0.0212，fold W/T/L=14/0/1，seed means=[0.02, 0.024, 0.0195]。
- `iterative_projected_minus_raw`：+0.0302，fold W/T/L=15/0/0，seed means=[0.026, 0.0315, 0.033]。
- `final_projected_minus_pca`：-0.0035，fold W/T/L=7/0/8，seed means=[0.003, -0.01, -0.0035]。
- `iterative_projected_minus_pca`：+0.0055，fold W/T/L=9/0/6，seed means=[0.009, -0.0025, 0.01]。
- `final_projected_minus_unconstrained`：-0.0075，fold W/T/L=4/1/10，seed means=[-0.0025, -0.0115, -0.0085]。
- `iterative_projected_minus_unconstrained`：+0.0015，fold W/T/L=7/1/7，seed means=[0.0035, -0.004, 0.005]。
- `final_projected_graph_minus_real_graph`：-0.0050，fold W/T/L=6/0/9，seed means=[-0.0065, -0.0025, -0.006]。
- `iterative_projected_graph_minus_real_graph`：+0.0020，fold W/T/L=8/0/7，seed means=[-0.0065, 0.01, 0.0025]。
- `final_projected_graph_minus_relation_graph`：-0.0092，fold W/T/L=4/0/11，seed means=[-0.0045, -0.018, -0.005]。
- `iterative_projected_graph_minus_relation_graph`：-0.0022，fold W/T/L=7/1/7，seed means=[-0.0045, -0.0055, 0.0035]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.3457 |
| final-projected | 1.000 | 0.3834 |
| iterative-projected | 1.000 | 0.4185 |

## 预注册门槛判定

- `final_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。
- `iterative_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。
