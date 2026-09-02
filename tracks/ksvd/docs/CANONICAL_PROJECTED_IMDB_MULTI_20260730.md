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

## IMDB-MULTI/cleaned/B0/max_nodes=12/canonical

图数：321；类别：`{'0': 144, '1': 85, '2': 92}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.4978 | -0.0031 | 9/0/6 | 1/3 |
| PCA | 0.4768 | -0.0241 | 3/1/11 | 0/3 |
| clustered real-patch | 0.4894 | -0.0115 | 6/0/9 | 1/3 |
| unconstrained KSVD | 0.5091 | +0.0082 | 9/0/6 | 2/3 |
| final-projected KSVD | 0.5485 | +0.0476 | 11/0/4 | 3/3 |
| iterative-projected KSVD | 0.4918 | -0.0092 | 7/0/8 | 1/3 |
| relation graph only | 0.5055 | +0.0045 | 9/0/6 | 2/3 |
| real-patch + graph | 0.4886 | -0.0123 | 7/0/8 | 0/3 |
| KSVD + graph | 0.5264 | +0.0255 | 10/0/5 | 2/3 |
| final-projected + graph | 0.5379 | +0.0370 | 9/0/6 | 3/3 |
| iterative-projected + graph | 0.5154 | +0.0144 | 7/0/8 | 3/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：+0.0591，fold W/T/L=10/1/4，seed means=[0.016, 0.0679, 0.0934]。
- `iterative_projected_minus_real`：+0.0023，fold W/T/L=8/0/7，seed means=[-0.0301, 0.0333, 0.0038]。
- `final_projected_minus_raw`：+0.0507，fold W/T/L=12/0/3，seed means=[0.03, 0.0477, 0.0744]。
- `iterative_projected_minus_raw`：-0.0060，fold W/T/L=8/0/7，seed means=[-0.0161, 0.0131, -0.0152]。
- `final_projected_minus_pca`：+0.0717，fold W/T/L=12/0/3，seed means=[0.059, 0.0523, 0.1037]。
- `iterative_projected_minus_pca`：+0.0149，fold W/T/L=10/0/5，seed means=[0.0129, 0.0177, 0.0141]。
- `final_projected_minus_unconstrained`：+0.0394，fold W/T/L=11/0/4，seed means=[0.0747, 0.015, 0.0284]。
- `iterative_projected_minus_unconstrained`：-0.0174，fold W/T/L=7/0/8，seed means=[0.0287, -0.0196, -0.0611]。
- `final_projected_graph_minus_real_graph`：+0.0493，fold W/T/L=11/0/4，seed means=[0.0306, 0.0544, 0.063]。
- `iterative_projected_graph_minus_real_graph`：+0.0268，fold W/T/L=10/0/5，seed means=[0.0273, 0.0419, 0.0112]。
- `final_projected_graph_minus_relation_graph`：+0.0325，fold W/T/L=9/0/6，seed means=[0.0164, 0.046, 0.035]。
- `iterative_projected_graph_minus_relation_graph`：+0.0099，fold W/T/L=8/0/7，seed means=[0.0131, 0.0335, -0.0168]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.1227 |
| final-projected | 1.000 | 0.1688 |
| iterative-projected | 1.000 | 0.2093 |

## 预注册门槛判定

- `final_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：['IMDB-MULTI/cleaned/B0/max_nodes=12/canonical']。
- `iterative_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。
