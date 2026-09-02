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

## IMDB-BINARY/cleaned/B0/max_nodes=12/rooted_canonical

图数：493；类别：`{'0': 261, '1': 232}`。

| 表征 | stats+feature balanced acc | Δ vs stats | fold W/T/L | seed wins |
|---|---:|---:|---:|---:|
| raw mean | 0.7502 | -0.0031 | 6/2/7 | 2/3 |
| PCA | 0.7495 | -0.0039 | 8/0/7 | 2/3 |
| clustered real-patch | 0.7403 | -0.0131 | 6/0/9 | 0/3 |
| unconstrained KSVD | 0.7375 | -0.0158 | 6/0/9 | 0/3 |
| final-projected KSVD | 0.7431 | -0.0102 | 4/1/10 | 0/3 |
| iterative-projected KSVD | 0.7265 | -0.0268 | 5/0/10 | 0/3 |
| relation graph only | 0.7466 | -0.0067 | 5/0/10 | 1/3 |
| real-patch + graph | 0.7435 | -0.0098 | 7/0/8 | 0/3 |
| KSVD + graph | 0.7394 | -0.0139 | 6/0/9 | 0/3 |
| final-projected + graph | 0.7461 | -0.0072 | 5/2/8 | 1/3 |
| iterative-projected + graph | 0.7269 | -0.0265 | 2/2/11 | 0/3 |

关键 paired comparisons（均已包含 stats）：

- `final_projected_minus_real`：+0.0028，fold W/T/L=9/1/5，seed means=[-0.0102, 0.0107, 0.008]。
- `iterative_projected_minus_real`：-0.0137，fold W/T/L=6/1/8，seed means=[-0.0141, -0.0185, -0.0086]。
- `final_projected_minus_raw`：-0.0071，fold W/T/L=6/1/8，seed means=[0.0016, -0.0063, -0.0166]。
- `iterative_projected_minus_raw`：-0.0237，fold W/T/L=5/1/9，seed means=[-0.0023, -0.0355, -0.0333]。
- `final_projected_minus_pca`：-0.0064，fold W/T/L=7/1/7，seed means=[0.012, -0.0158, -0.0154]。
- `iterative_projected_minus_pca`：-0.0229，fold W/T/L=5/0/10，seed means=[0.0081, -0.0449, -0.032]。
- `final_projected_minus_unconstrained`：+0.0056，fold W/T/L=10/0/5，seed means=[0.012, -0.0033, 0.008]。
- `iterative_projected_minus_unconstrained`：-0.0110，fold W/T/L=7/1/7，seed means=[0.0081, -0.0325, -0.0086]。
- `final_projected_graph_minus_real_graph`：+0.0026，fold W/T/L=9/1/5，seed means=[-0.0161, 0.0129, 0.0111]。
- `iterative_projected_graph_minus_real_graph`：-0.0166，fold W/T/L=6/0/9，seed means=[-0.0222, -0.022, -0.0057]。
- `final_projected_graph_minus_relation_graph`：-0.0005，fold W/T/L=8/0/7，seed means=[0.0134, -0.018, 0.0032]。
- `iterative_projected_graph_minus_relation_graph`：-0.0197，fold W/T/L=8/0/7，seed means=[0.0073, -0.0529, -0.0136]。

字典训练折诊断（所有 outer folds 的 mean）：

| 字典 | projectability | reconstruction |
|---|---:|---:|
| unconstrained | nan | 0.1138 |
| final-projected | 1.000 | 0.1474 |
| iterative-projected | 1.000 | 0.2173 |

## 预注册门槛判定

- `final_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。
- `iterative_projected` 在全部三个 baseline 上均 3/3 seed 正向的数据集：无。

**最终判定：NO-GO。**

注意：重构改善不能覆盖分类门槛失败；relation graph 的结果也不能归因于 KSVD vocabulary。
