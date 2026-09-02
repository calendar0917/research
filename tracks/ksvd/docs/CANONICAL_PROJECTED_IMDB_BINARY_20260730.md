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

## IMDB-BINARY/cleaned/B0/max_nodes=12/canonical

图数：493；类别：`{'0': 261, '1': 232}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.7440 | -0.0093 | 7/0/8 | 0/3 |
| PCA | 0.7552 | +0.0019 | 8/2/5 | 2/3 |
| clustered real-patch | 0.7579 | +0.0045 | 9/0/6 | 1/3 |
| unconstrained KSVD | 0.7478 | -0.0055 | 7/0/8 | 0/3 |
| final-projected KSVD | 0.7517 | -0.0017 | 8/0/7 | 1/3 |
| iterative-projected KSVD | 0.7415 | -0.0118 | 7/0/8 | 0/3 |
| relation graph only | 0.7466 | -0.0067 | 5/0/10 | 1/3 |
| real-patch + graph | 0.7400 | -0.0133 | 4/0/11 | 1/3 |
| KSVD + graph | 0.7423 | -0.0110 | 6/0/9 | 0/3 |
| final-projected + graph | 0.7545 | +0.0012 | 8/0/7 | 2/3 |
| iterative-projected + graph | 0.7433 | -0.0100 | 7/0/8 | 1/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：-0.0062，fold W/T/L=5/0/10，seed means=[0.0051, 0.0006, -0.0243]。
- `iterative_projected_minus_real`：-0.0163，fold W/T/L=4/0/11，seed means=[0.0085, -0.0142, -0.0433]。
- `final_projected_minus_raw`：+0.0077，fold W/T/L=6/0/9，seed means=[0.0004, 0.0148, 0.0078]。
- `iterative_projected_minus_raw`：-0.0025，fold W/T/L=6/1/8，seed means=[0.0038, 0.0, -0.0112]。
- `final_projected_minus_pca`：-0.0035，fold W/T/L=9/0/6，seed means=[0.0032, -0.0083, -0.0055]。
- `iterative_projected_minus_pca`：-0.0137，fold W/T/L=5/0/10，seed means=[0.0066, -0.0231, -0.0245]。
- `final_projected_minus_unconstrained`：+0.0038，fold W/T/L=8/0/7，seed means=[0.0013, 0.0033, 0.007]。
- `iterative_projected_minus_unconstrained`：-0.0063，fold W/T/L=8/0/7，seed means=[0.0046, -0.0115, -0.012]。
- `final_projected_graph_minus_real_graph`：+0.0145，fold W/T/L=7/0/8，seed means=[0.0236, 0.0169, 0.003]。
- `iterative_projected_graph_minus_real_graph`：+0.0033，fold W/T/L=6/0/9，seed means=[0.03, -0.0007, -0.0194]。
- `final_projected_graph_minus_relation_graph`：+0.0079，fold W/T/L=11/0/4，seed means=[0.0309, -0.0152, 0.0081]。
- `iterative_projected_graph_minus_relation_graph`：-0.0033，fold W/T/L=7/0/8，seed means=[0.0372, -0.0328, -0.0143]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.1157 |
| final-projected | 1.000 | 0.1502 |
| iterative-projected | 1.000 | 0.2313 |

## 预注册门槛判定

- `final_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。
- `iterative_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。
