# MolHIV 中心级结构–属性交互：official-test 冻结评估

协议：`luyin16-molhiv-cross-center-interaction-controlled-test-v1`

这是在 valid 搜索和视图判断完成后进行的一次 controlled terminal test；test 结果没有用于选择视图、超参数或融合权重。仓库中更早路线已经查看过 MolHIV test，因此不宣称 untouched-test。

数据：train/valid/test = 32901/4113/4113，test positive = 130。

`train_only` 是严格只用 official-train 拟合 PCA 和模型；`train_valid_refit` 是冻结后常规的 train+valid 重训，PCA 也只在该拟合范围内学习。

## Strict train-only fit

| view | valid tuned mean (reference) | test five-seed mean | test seed ensemble | test − valid |
|---|---:|---:|---:|---:|
| `s` | 0.791586 | 0.743173 | 0.746054 | -0.048414 |
| `s_marginal` | 0.841285 | 0.788872 | 0.791753 | -0.052414 |
| `s_cross_cov` | 0.839257 | 0.800937 | 0.805587 | -0.038320 |
| `s_binding` | 0.838859 | 0.789697 | 0.792804 | -0.049162 |
| `s_both` | 0.838222 | 0.808749 | 0.813525 | -0.029473 |

## Train+valid refit

| view | valid tuned mean (reference) | test five-seed mean | test seed ensemble | test − valid |
|---|---:|---:|---:|---:|
| `s` | 0.791586 | 0.753994 | 0.756810 | -0.037592 |
| `s_marginal` | 0.841285 | 0.785279 | 0.788797 | -0.056006 |
| `s_cross_cov` | 0.839257 | 0.799824 | 0.803652 | -0.039433 |
| `s_binding` | 0.838859 | 0.788038 | 0.790477 | -0.050821 |
| `s_both` | 0.838222 | 0.795642 | 0.800228 | -0.042580 |

## PCA-scope control

为区分“加入 valid 监督样本”和“重新拟合低秩交互坐标”，另做了一个冻结坐标控制：PCA 只在 official-train 拟合，分类器用 train+valid 标签重训。

| view | fixed-train-PCA mean | seed ensemble | vs PCA-refit mean |
|---|---:|---:|---:|
| `s` | 0.753994 | 0.756810 | +0.000000 |
| `s_marginal` | 0.785279 | 0.788797 | +0.000000 |
| `s_cross_cov` | 0.802397 | 0.805668 | +0.002573 |
| `s_binding` | 0.788674 | 0.790792 | +0.000636 |
| `s_both` | **0.803644** | **0.807049** | **+0.008002** |

这个控制说明 `S+both` 的 train+valid 重训下降主要来自 PCA 坐标系改变；固定坐标后交互增益仍保留。

## Per-seed test AUC

| view | train-only seeds 0…4 | train+valid seeds 0…4 |
|---|---|---|
| `s` | 0.736473/0.754295/0.742978/0.742200/0.739919 | 0.761466/0.755948/0.760195/0.743955/0.748407 |
| `s_marginal` | 0.793275/0.781898/0.785591/0.792364/0.791232 | 0.773829/0.786045/0.788603/0.788542/0.789378 |
| `s_cross_cov` | 0.803250/0.797252/0.797071/0.799607/0.807506 | 0.796624/0.808604/0.796576/0.800967/0.796352 |
| `s_binding` | 0.784536/0.791168/0.792393/0.790037/0.790351 | 0.785004/0.791126/0.781048/0.788725/0.794288 |
| `s_both` | 0.812648/0.804726/0.811248/0.807824/0.807302 | 0.798926/0.797518/0.790359/0.804943/0.786464 |

## Interpretation boundary

- 这里的主问题是 valid 上的增益能否迁移到 test；不能用 test 排名反向改写路线。
- 五个视图都按各自 valid 搜索得到的参数评估；`S+both` 的 train-CV 选择身份被保留，但 test 只作冻结后的泛化检查。
- 若不同视图在 test 上排序反转，应视为 scaffold shift / 小阳性集不确定性的证据，而不是继续用 test 调参。
- `cross_cov` 在严格 train-only、固定 train-only PCA 的 train+valid 控制和 PCA-refit
  三种读法中都保持约 `0.800`；`binding` 单独接近 marginal，而 `both` 的额外收益
  随 PCA scope 变化。因此更稳妥的机制表述是“cross-centre covariance 为核心，
  binding 为条件辅助块”。
- official-test 只有 130 个 positive；以 AUC 的常用近似标准误约 `0.023` 看，
  0.01--0.02 量级的视图差异不能单独当作精确显著性结论。

原始结果：[test_summary.json](cross_center_interaction_terminal/test_summary.json)；
PCA 控制原始结果：[test_pca_control.json](cross_center_interaction_terminal/test_pca_control.json)
