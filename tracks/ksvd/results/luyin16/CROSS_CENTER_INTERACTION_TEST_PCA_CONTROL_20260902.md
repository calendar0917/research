# MolHIV 中心级交互：固定 train-only PCA 的 test 控制

协议：`luyin16-molhiv-cross-center-interaction-test-pca-scope-control-v1`

这是对正式 frozen test 的敏感性控制：PCA 只在 official-train 拟合并固定，分类器则用 official-train+valid 标签重训。它不改变任何视图或 XGBoost 参数，也不使用 test 结果做选择。

| view | fixed-train-PCA + train+valid mean | seed ensemble | relative to PCA-refit mean |
|---|---:|---:|---:|
| `s` | 0.753994 | 0.756810 | +0.000000 |
| `s_marginal` | 0.785279 | 0.788797 | +0.000000 |
| `s_cross_cov` | 0.802397 | 0.805668 | +0.002573 |
| `s_binding` | 0.788674 | 0.790792 | +0.000636 |
| `s_both` | 0.803644 | 0.807049 | +0.008002 |

解释：若该控制高于 PCA-refit 结果，差异来自交互坐标系改变；若仍低，才更可能是加入 valid 后的模型拟合/分布问题。这里的结果只用于诊断，不允许据此回头调 test。

原始结果：[test_pca_control.json](cross_center_interaction_terminal/test_pca_control.json)
