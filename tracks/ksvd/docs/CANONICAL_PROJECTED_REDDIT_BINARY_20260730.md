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

## REDDIT-BINARY/raw/R2/max_nodes=24/canonical

图数：2000；类别：`{'0': 1000, '1': 1000}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.8160 | -0.0272 | 1/0/14 | 0/3 |
| PCA | 0.8333 | -0.0098 | 3/1/11 | 0/3 |
| clustered real-patch | 0.8375 | -0.0057 | 5/1/9 | 0/3 |
| unconstrained KSVD | 0.8378 | -0.0053 | 5/0/10 | 0/3 |
| final-projected KSVD | 0.8418 | -0.0013 | 8/0/7 | 2/3 |
| iterative-projected KSVD | 0.8335 | -0.0097 | 5/1/9 | 1/3 |
| relation graph only | 0.8550 | +0.0118 | 12/0/3 | 3/3 |
| real-patch + graph | 0.8482 | +0.0050 | 7/2/6 | 3/3 |
| KSVD + graph | 0.8462 | +0.0030 | 9/0/6 | 3/3 |
| final-projected + graph | 0.8488 | +0.0057 | 9/0/6 | 2/3 |
| iterative-projected + graph | 0.8478 | +0.0047 | 9/0/6 | 3/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：+0.0043，fold W/T/L=7/1/7，seed means=[-0.003, 0.0095, 0.0065]。
- `iterative_projected_minus_real`：-0.0040，fold W/T/L=4/1/10，seed means=[-0.0105, -0.0045, 0.003]。
- `final_projected_minus_raw`：+0.0258，fold W/T/L=14/0/1，seed means=[0.0115, 0.034, 0.032]。
- `iterative_projected_minus_raw`：+0.0175，fold W/T/L=12/1/2，seed means=[0.004, 0.02, 0.0285]。
- `final_projected_minus_pca`：+0.0085，fold W/T/L=9/1/5，seed means=[-0.003, 0.02, 0.0085]。
- `iterative_projected_minus_pca`：+0.0002，fold W/T/L=6/2/7，seed means=[-0.0105, 0.006, 0.005]。
- `final_projected_minus_unconstrained`：+0.0040，fold W/T/L=10/0/5，seed means=[-0.0, 0.003, 0.009]。
- `iterative_projected_minus_unconstrained`：-0.0043，fold W/T/L=6/1/8，seed means=[-0.0075, -0.011, 0.0055]。
- `final_projected_graph_minus_real_graph`：+0.0007，fold W/T/L=7/1/7，seed means=[-0.0135, -0.0015, 0.017]。
- `iterative_projected_graph_minus_real_graph`：-0.0003，fold W/T/L=8/0/7，seed means=[-0.0065, 0.0005, 0.005]。
- `final_projected_graph_minus_relation_graph`：-0.0062，fold W/T/L=4/0/11，seed means=[-0.017, -0.012, 0.0105]。
- `iterative_projected_graph_minus_relation_graph`：-0.0072，fold W/T/L=3/0/12，seed means=[-0.01, -0.01, -0.0015]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.3997 |
| final-projected | 1.000 | 0.4484 |
| iterative-projected | 1.000 | 0.5314 |

## 预注册门槛判定

- `final_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。
- `iterative_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。
