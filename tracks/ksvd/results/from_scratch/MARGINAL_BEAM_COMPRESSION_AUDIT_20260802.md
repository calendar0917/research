# Marginal beam cover：matched KSVD compression 审计

> 日期：2026-08-02  
> same s=10, K24/T3/u25；3-fold graph isolation。

## 1. 判定

**ADOPT_MARGINAL_BEAM_COVER_V2**

## 2. Fold-balanced means

| branch | stage | patch err | observed RMSE | full RMSE | observed F1 | full recall/F1 | disagreement |
|---|---|---:|---:|---:|---:|---:|---:|
| target_o5 | raw | 0.0000 | 0.0000 | 0.3538 | 1.0000 | 0.6851/0.8125 | 0.0000 |
| target_o5 | init | 0.5162 | 0.3838 | 0.4442 | 0.8237 | 0.5816/0.6737 | 0.1442 |
| target_o5 | final | 0.4928 | 0.3653 | 0.4367 | 0.8387 | 0.5909/0.6854 | 0.1409 |
| target_o5 | pca3 | 0.5622 | 0.4199 | 0.4601 | 0.7653 | 0.5469/0.6266 | 0.1230 |
| target_o3 | raw | 0.0000 | 0.0000 | 0.3073 | 1.0000 | 0.7616/0.8642 | 0.0000 |
| target_o3 | init | 0.5238 | 0.3943 | 0.4255 | 0.8073 | 0.6356/0.7015 | 0.1431 |
| target_o3 | final | 0.5021 | 0.3760 | 0.4162 | 0.8211 | 0.6442/0.7127 | 0.1470 |
| target_o3 | pca3 | 0.5701 | 0.4276 | 0.4432 | 0.7458 | 0.5875/0.6472 | 0.1466 |
| beam_o3 | raw | 0.0000 | 0.0000 | 0.2096 | 1.0000 | 0.8852/0.9386 | 0.0000 |
| beam_o3 | init | 0.4460 | 0.3647 | 0.3375 | 0.8717 | 0.8237/0.8224 | 0.1273 |
| beam_o3 | final | 0.4208 | 0.3419 | 0.3246 | 0.8844 | 0.8289/0.8337 | 0.1245 |
| beam_o3 | pca3 | 0.4789 | 0.3927 | 0.3536 | 0.8348 | 0.8078/0.7886 | 0.1071 |

## 3. Registered gate

- full RMSE reduction：`0.2201`；
- observed RMSE reduction：`0.0909`；
- checks：`{'full_rmse_reduction_at_least_005': True, 'full_recall_improves': True, 'full_f1_improves': True, 'observed_rmse_not_worse_by_005': True, 'raw_gates': True, 'patch_final_better_all_folds': True, 'beats_own_pca3_full_rmse': True}`；
- gate：`True`。

## 4. 边界

- beam 偏向 edge-dense patches；必须同时阅读 observed RMSE、pair coverage 和 full graph metrics。
- 本轮不含 completion；unseen pair 仍 zero fill。
