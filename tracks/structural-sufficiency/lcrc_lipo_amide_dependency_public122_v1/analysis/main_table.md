# 主表 - MATCHED 人群 (FULL 为主估计, G_FIXED 为定位诊断)

loss 为原始 logD 尺度上的绝对误差; B = mean(LT - L0), D = mean(LT - LK)。
CI 为 scaffold cluster bootstrap 95% percentile 区间 (1000 次, seed 20261114)。

| split | scope | view | MAE | RMSE | B [95% CI] | D [95% CI] |
|---|---|---|---|---|---|---|
| valid | FULL | O | 0.3586 | 0.4813 | 7.3545 [6.8255, 7.8513] | 0.4192 [0.1917, 0.6276] |
| valid | FULL | T | 7.7131 | 7.8754 | | |
| valid | FULL | K | 7.2940 | 7.4456 | | |
| valid | G_FIXED | O | 0.3586 | 0.4813 | 0.2977 [0.1742, 0.4149] | 0.1542 [0.0167, 0.2759] |
| valid | G_FIXED | T | 0.6563 | 0.7869 | | |
| valid | G_FIXED | K | 0.5021 | 0.6109 | | |
| test | FULL | O | 0.3841 | 0.5364 | 7.5644 [6.9791, 8.1151] | 0.4705 [0.0894, 0.8531] |
| test | FULL | T | 7.9485 | 8.1624 | | |
| test | FULL | K | 7.4780 | 7.6764 | | |
| test | G_FIXED | O | 0.3841 | 0.5364 | 0.1988 [0.0678, 0.3300] | 0.1449 [-0.0165, 0.3049] |
| test | G_FIXED | T | 0.5828 | 0.7790 | | |
| test | G_FIXED | K | 0.4379 | 0.5676 | | |
