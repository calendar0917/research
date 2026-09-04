# MolHIV 中心级结构–属性交互：official-test 冻结评估

协议：`luyin16-molhiv-cross-center-interaction-controlled-test-pca16-v1`

这是在 valid 搜索和视图判断完成后进行的一次 controlled terminal test；test 结果没有用于选择视图、超参数或融合权重。仓库中更早路线已经查看过MolHIV test，因此不宣称 untouched-test。

数据：train/valid/test = 32901/4113/4113，test positive = 130。

`train_only` 是严格只用 official-train 拟合 PCA 和模型；`train_valid_refit` 是冻结后常规的 train+valid 重训，PCA 也只在该拟合范围内学习。

## Strict train-only fit

| view | valid tuned mean (reference) | test five-seed mean | test seed ensemble | test − valid |
|---|---:|---:|---:|---:|
| `s` | 0.791586 | 0.743173 | 0.746054 | -0.048414 |
| `s_marginal` | 0.841285 | 0.788872 | 0.791753 | -0.052414 |
| `s_cross_cov` | 0.837160 | 0.803619 | 0.806223 | -0.033541 |
| `s_binding` | 0.832134 | 0.799801 | 0.803913 | -0.032333 |
| `s_both` | 0.832970 | 0.808644 | 0.811317 | -0.024325 |

## Train+valid refit

| view | valid tuned mean (reference) | test five-seed mean | test seed ensemble | test − valid |
|---|---:|---:|---:|---:|
| `s` | 0.791586 | 0.753994 | 0.756810 | -0.037592 |
| `s_marginal` | 0.841285 | 0.785279 | 0.788797 | -0.056006 |
| `s_cross_cov` | 0.837160 | 0.797590 | 0.799998 | -0.039570 |
| `s_binding` | 0.832134 | 0.793064 | 0.795979 | -0.039070 |
| `s_both` | 0.832970 | 0.801946 | 0.804531 | -0.031024 |

## Per-seed test AUC

| view | train-only seeds 0…4 | train+valid seeds 0…4 |
|---|---|---|
| `s` | 0.736473/0.754295/0.742978/0.742200/0.739919 | 0.761466/0.755948/0.760195/0.743955/0.748407 |
| `s_marginal` | 0.793275/0.781898/0.785591/0.792364/0.791232 | 0.773829/0.786045/0.788603/0.788542/0.789378 |
| `s_cross_cov` | 0.803785/0.799900/0.800715/0.807804/0.805892 | 0.795324/0.795587/0.796873/0.798604/0.801562 |
| `s_binding` | 0.794247/0.805554/0.795454/0.804610/0.799143 | 0.778646/0.789111/0.798749/0.800369/0.798445 |
| `s_both` | 0.808953/0.809662/0.808241/0.809656/0.806709 | 0.800025/0.798215/0.808301/0.802277/0.800910 |

## Interpretation boundary

- 这里的主问题是 valid 上的增益能否迁移到 test；不能用 test 排名反向改写路线。
- 五个视图都按各自 valid 搜索得到的参数评估；`S+both` 的 train-CV 选择身份被保留，但 test 只作冻结后的泛化检查。
- 若不同视图在 test 上排序反转，应视为 scaffold shift / 小阳性集不确定性的证据，而不是继续用 test 调参。

原始结果：[test_summary.json](cross_center_interaction_terminal/test_summary.json)
