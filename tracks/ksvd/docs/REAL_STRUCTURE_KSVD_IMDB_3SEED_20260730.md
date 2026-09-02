# 真实纯结构数据上的 KSVD 前置审计（2026-07-30）

> 目标不是追 TUData leaderboard，而是在无节点/边属性的真实图上，严格区分简单统计、原始 patch 表示、普通降维、真实 patch 字典与 KSVD。所有数据依赖变换只在训练折拟合。

## 协议

- 数据集：`IMDB-BINARY, IMDB-MULTI`；版本：`raw, cleaned`。
- patch：`B0, R2`；每图最多 16 个中心，每个 patch 最多 12 个节点。
- 字典：16 atoms，稀疏度 T=2，KSVD iterations=3。
- 外层评估：5-fold × split seeds [0, 1, 2]。
- `raw` 与 `cleaned` 同时报告，用于检查同构重复图造成的结果膨胀。
- `real_patch_dictionary` 的每个 atom 是真实训练 patch；KSVD atom 仍是 WL 特征空间中的欧氏向量，不自动等价于合法图。

## IMDB-BINARY/raw/B0

图数：1000；类别：`{'0': 500, '1': 500}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.7083 ± 0.0244 | 0.7083 ± 0.0244 |
| degree_hist | 0.7240 ± 0.0314 | 0.7240 ± 0.0314 |
| stats_degree | 0.7327 ± 0.0220 | 0.7327 ± 0.0220 |
| relation_graph_only | 0.6470 ± 0.0207 | 0.6470 ± 0.0207 |
| raw_patch_mean | 0.6690 ± 0.0234 | 0.6690 ± 0.0234 |
| pca_content | 0.6733 ± 0.0251 | 0.6733 ± 0.0251 |
| random_dictionary_content | 0.6687 ± 0.0275 | 0.6687 ± 0.0275 |
| real_patch_dictionary_content | 0.6837 ± 0.0283 | 0.6837 ± 0.0283 |
| ksvd_content | 0.6737 ± 0.0275 | 0.6737 ± 0.0275 |
| ksvd_content_graph | 0.6953 ± 0.0238 | 0.6953 ± 0.0238 |
| ksvd_content_true_relation | 0.6897 ± 0.0251 | 0.6897 ± 0.0251 |
| ksvd_content_shuffled_relation | 0.6807 ± 0.0300 | 0.6807 ± 0.0300 |

重构误差（test graph mean）：

- KSVD：0.1021 ± 0.0041
- PCA：0.0898 ± 0.0027
- clustered real patch：0.0854 ± 0.0035

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.8029736869724087, 'std': 0.04564616656860617, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.8281161299677376, 'std': 0.044493087981454946, 'n_pairs': 105}`

## IMDB-BINARY/raw/R2

图数：1000；类别：`{'0': 500, '1': 500}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.7083 ± 0.0244 | 0.7083 ± 0.0244 |
| degree_hist | 0.7240 ± 0.0314 | 0.7240 ± 0.0314 |
| stats_degree | 0.7327 ± 0.0220 | 0.7327 ± 0.0220 |
| relation_graph_only | 0.5753 ± 0.0336 | 0.5753 ± 0.0336 |
| raw_patch_mean | 0.6687 ± 0.0216 | 0.6687 ± 0.0216 |
| pca_content | 0.6673 ± 0.0276 | 0.6673 ± 0.0276 |
| random_dictionary_content | 0.6523 ± 0.0331 | 0.6523 ± 0.0331 |
| real_patch_dictionary_content | 0.6510 ± 0.0211 | 0.6510 ± 0.0211 |
| ksvd_content | 0.6557 ± 0.0283 | 0.6557 ± 0.0283 |
| ksvd_content_graph | 0.6820 ± 0.0366 | 0.6820 ± 0.0366 |
| ksvd_content_true_relation | 0.6847 ± 0.0277 | 0.6847 ± 0.0277 |
| ksvd_content_shuffled_relation | 0.6900 ± 0.0394 | 0.6900 ± 0.0394 |

重构误差（test graph mean）：

- KSVD：0.2894 ± 0.0184
- PCA：0.2403 ± 0.0092
- clustered real patch：0.2387 ± 0.0180

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.7710598629195197, 'std': 0.044039955987254145, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.8818462831971514, 'std': 0.04161975510399817, 'n_pairs': 105}`

## IMDB-BINARY/cleaned/B0

图数：493；类别：`{'0': 261, '1': 232}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.7346 ± 0.0384 | 0.7377 ± 0.0383 |
| degree_hist | 0.7465 ± 0.0328 | 0.7472 ± 0.0328 |
| stats_degree | 0.7506 ± 0.0325 | 0.7519 ± 0.0325 |
| relation_graph_only | 0.6377 ± 0.0459 | 0.6403 ± 0.0451 |
| raw_patch_mean | 0.6626 ± 0.0482 | 0.6626 ± 0.0475 |
| pca_content | 0.6962 ± 0.0400 | 0.6963 ± 0.0403 |
| random_dictionary_content | 0.6903 ± 0.0369 | 0.6903 ± 0.0373 |
| real_patch_dictionary_content | 0.6976 ± 0.0498 | 0.6977 ± 0.0491 |
| ksvd_content | 0.7004 ± 0.0440 | 0.7011 ± 0.0442 |
| ksvd_content_graph | 0.7306 ± 0.0446 | 0.7302 ± 0.0447 |
| ksvd_content_true_relation | 0.6702 ± 0.0295 | 0.6700 ± 0.0284 |
| ksvd_content_shuffled_relation | 0.6869 ± 0.0502 | 0.6862 ± 0.0504 |

重构误差（test graph mean）：

- KSVD：0.1481 ± 0.0106
- PCA：0.1322 ± 0.0087
- clustered real patch：0.1266 ± 0.0124

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.7963975902746637, 'std': 0.041566138216793, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.8257008500356613, 'std': 0.0441722044162254, 'n_pairs': 105}`

## IMDB-BINARY/cleaned/R2

图数：493；类别：`{'0': 261, '1': 232}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.7346 ± 0.0384 | 0.7377 ± 0.0383 |
| degree_hist | 0.7465 ± 0.0328 | 0.7472 ± 0.0328 |
| stats_degree | 0.7506 ± 0.0325 | 0.7519 ± 0.0325 |
| relation_graph_only | 0.6420 ± 0.0571 | 0.6444 ± 0.0569 |
| raw_patch_mean | 0.6228 ± 0.0436 | 0.6220 ± 0.0431 |
| pca_content | 0.6702 ± 0.0470 | 0.6700 ± 0.0468 |
| random_dictionary_content | 0.6468 ± 0.0360 | 0.6470 ± 0.0365 |
| real_patch_dictionary_content | 0.6697 ± 0.0516 | 0.6694 ± 0.0524 |
| ksvd_content | 0.6694 ± 0.0351 | 0.6701 ± 0.0357 |
| ksvd_content_graph | 0.6751 ± 0.0294 | 0.6754 ± 0.0295 |
| ksvd_content_true_relation | 0.6510 ± 0.0548 | 0.6512 ± 0.0550 |
| ksvd_content_shuffled_relation | 0.6458 ± 0.0434 | 0.6450 ± 0.0430 |

重构误差（test graph mean）：

- KSVD：0.4129 ± 0.0205
- PCA：0.3666 ± 0.0126
- clustered real patch：0.3962 ± 0.0191

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.7554084046184437, 'std': 0.047543949602980584, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.8245752120428572, 'std': 0.04500411579855153, 'n_pairs': 105}`

## IMDB-MULTI/raw/B0

图数：1500；类别：`{'0': 500, '1': 500, '2': 500}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.4629 ± 0.0218 | 0.4629 ± 0.0218 |
| degree_hist | 0.4822 ± 0.0225 | 0.4822 ± 0.0225 |
| stats_degree | 0.4918 ± 0.0240 | 0.4918 ± 0.0240 |
| relation_graph_only | 0.4571 ± 0.0226 | 0.4571 ± 0.0226 |
| raw_patch_mean | 0.4769 ± 0.0235 | 0.4769 ± 0.0235 |
| pca_content | 0.4678 ± 0.0259 | 0.4678 ± 0.0259 |
| random_dictionary_content | 0.4616 ± 0.0298 | 0.4616 ± 0.0298 |
| real_patch_dictionary_content | 0.4716 ± 0.0291 | 0.4716 ± 0.0291 |
| ksvd_content | 0.4678 ± 0.0208 | 0.4678 ± 0.0208 |
| ksvd_content_graph | 0.4744 ± 0.0278 | 0.4744 ± 0.0278 |
| ksvd_content_true_relation | 0.4627 ± 0.0169 | 0.4627 ± 0.0169 |
| ksvd_content_shuffled_relation | 0.4613 ± 0.0228 | 0.4613 ± 0.0228 |

重构误差（test graph mean）：

- KSVD：0.0676 ± 0.0082
- PCA：0.0606 ± 0.0063
- clustered real patch：0.0566 ± 0.0078

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.7749334930719501, 'std': 0.04733006304479584, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.7665275962110143, 'std': 0.04377156859280489, 'n_pairs': 105}`

## IMDB-MULTI/raw/R2

图数：1500；类别：`{'0': 500, '1': 500, '2': 500}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.4629 ± 0.0218 | 0.4629 ± 0.0218 |
| degree_hist | 0.4822 ± 0.0225 | 0.4822 ± 0.0225 |
| stats_degree | 0.4918 ± 0.0240 | 0.4918 ± 0.0240 |
| relation_graph_only | 0.4611 ± 0.0187 | 0.4611 ± 0.0187 |
| raw_patch_mean | 0.4567 ± 0.0223 | 0.4567 ± 0.0223 |
| pca_content | 0.4427 ± 0.0193 | 0.4427 ± 0.0193 |
| random_dictionary_content | 0.4449 ± 0.0209 | 0.4449 ± 0.0209 |
| real_patch_dictionary_content | 0.4447 ± 0.0197 | 0.4447 ± 0.0197 |
| ksvd_content | 0.4462 ± 0.0229 | 0.4462 ± 0.0229 |
| ksvd_content_graph | 0.4573 ± 0.0214 | 0.4573 ± 0.0214 |
| ksvd_content_true_relation | 0.4593 ± 0.0186 | 0.4593 ± 0.0186 |
| ksvd_content_shuffled_relation | 0.4562 ± 0.0228 | 0.4562 ± 0.0228 |

重构误差（test graph mean）：

- KSVD：0.2664 ± 0.0214
- PCA：0.2255 ± 0.0132
- clustered real patch：0.2082 ± 0.0165

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.752152377479918, 'std': 0.042382862043329556, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.8367829053876044, 'std': 0.04756912844613146, 'n_pairs': 105}`

## IMDB-MULTI/cleaned/B0

图数：321；类别：`{'0': 144, '1': 85, '2': 92}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.4211 ± 0.0372 | 0.4734 ± 0.0428 |
| degree_hist | 0.5346 ± 0.0409 | 0.5483 ± 0.0422 |
| stats_degree | 0.4901 ± 0.0524 | 0.5026 ± 0.0502 |
| relation_graph_only | 0.3972 ± 0.0506 | 0.4672 ± 0.0558 |
| raw_patch_mean | 0.4951 ± 0.0474 | 0.5089 ± 0.0453 |
| pca_content | 0.5127 ± 0.0403 | 0.5233 ± 0.0415 |
| random_dictionary_content | 0.5177 ± 0.0488 | 0.5347 ± 0.0432 |
| real_patch_dictionary_content | 0.5287 ± 0.0400 | 0.5441 ± 0.0384 |
| ksvd_content | 0.5154 ± 0.0517 | 0.5347 ± 0.0532 |
| ksvd_content_graph | 0.5077 ± 0.0457 | 0.5243 ± 0.0498 |
| ksvd_content_true_relation | 0.4794 ± 0.0551 | 0.4932 ± 0.0558 |
| ksvd_content_shuffled_relation | 0.4558 ± 0.0509 | 0.4662 ± 0.0483 |

重构误差（test graph mean）：

- KSVD：0.1688 ± 0.0142
- PCA：0.1537 ± 0.0131
- clustered real patch：0.1490 ± 0.0130

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.8006996834584208, 'std': 0.04195392472725876, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.8006147754345689, 'std': 0.0400144490988718, 'n_pairs': 105}`

## IMDB-MULTI/cleaned/R2

图数：321；类别：`{'0': 144, '1': 85, '2': 92}`。

| 表征 | Balanced Acc | Accuracy |
|---|---:|---:|
| size_degree_stats | 0.4211 ± 0.0372 | 0.4734 ± 0.0428 |
| degree_hist | 0.5346 ± 0.0409 | 0.5483 ± 0.0422 |
| stats_degree | 0.4901 ± 0.0524 | 0.5026 ± 0.0502 |
| relation_graph_only | 0.3713 ± 0.0301 | 0.4589 ± 0.0387 |
| raw_patch_mean | 0.4272 ± 0.0474 | 0.4435 ± 0.0416 |
| pca_content | 0.4414 ± 0.0400 | 0.4475 ± 0.0458 |
| random_dictionary_content | 0.4140 ± 0.0514 | 0.4258 ± 0.0548 |
| real_patch_dictionary_content | 0.4389 ± 0.0552 | 0.4497 ± 0.0510 |
| ksvd_content | 0.4305 ± 0.0538 | 0.4434 ± 0.0535 |
| ksvd_content_graph | 0.3991 ± 0.0635 | 0.4143 ± 0.0605 |
| ksvd_content_true_relation | 0.4392 ± 0.0462 | 0.4579 ± 0.0424 |
| ksvd_content_shuffled_relation | 0.4491 ± 0.0620 | 0.4609 ± 0.0599 |

重构误差（test graph mean）：

- KSVD：0.4584 ± 0.0303
- PCA：0.4195 ± 0.0286
- clustered real patch：0.4578 ± 0.0358

字典稳定性（跨 folds/seeds，Hungarian matched absolute cosine）：

- KSVD：`{'mean_matched_absolute_cosine': 0.741599294533727, 'std': 0.04400932203453529, 'n_pairs': 105}`
- clustered real patch：`{'mean_matched_absolute_cosine': 0.7717552468344022, 'std': 0.03964721318644604, 'n_pairs': 105}`

## 解释纪律

1. KSVD 只有稳定优于 `raw_patch_mean`、PCA、random dictionary 和 clustered real-patch dictionary，才支持“字典学习本身提供额外信息”。
2. `true_relation` 必须稳定优于 `shuffled_relation`，才能支持当前 sparse atom identity 与 patch 位置的正确绑定有用。
3. raw 好、cleaned 明显下降时，应首先解释为同构重复/数据集偏差，而不是模型表达力。
4. 即便 KSVD 分类更好，当前 WL 向量 atom 仍不自动满足合法图、精确可解码或 exact occurrence；这些需要单独验证。
