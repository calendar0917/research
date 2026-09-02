# Revised cover configurations：matched KSVD 审计

> 日期：2026-08-02  
> RAW Pareto follow-up；K24/T3/u25；3-fold isolation。

## 1. 判定

**RAW_COVER_GAIN_LOST_AFTER_COMPRESSION**

## 2. Costs and FINAL metrics

| branch | s/o/m | patches | dict scalars | code scalars/graph | raw pair slots | patch err | observed RMSE | full RMSE | full recall/F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 10/5/1.5 | 17.6528 | 1080 | 52.9583 | 794.3750 | 0.4894 | 0.3648 | 0.4352 | 0.5928/0.6870 |
| lower_overlap | 10/3/1.5 | 17.6528 | 1080 | 52.9583 | 794.3750 | 0.5009 | 0.3749 | 0.4149 | 0.6446/0.7137 |
| larger_patch | 12/4/1.5 | 12.0000 | 1584 | 36.0000 | 792.0000 | 0.5425 | 0.4033 | 0.4356 | 0.6051/0.6725 |

## 3. Registered lower-overlap gate

- full RMSE reduction：`0.0467`；
- observed RMSE reduction：`-0.0278`；
- checks：`{'patch_count_not_higher': True, 'dictionary_not_larger': True, 'code_not_larger': True, 'full_rmse_reduction_at_least_002': True, 'observed_rmse_not_worse_by_001': False, 'full_recall_not_worse': True, 'full_f1_not_worse': True, 'raw_gates': True, 'patch_final_better_all_folds': True}`；
- gate：`False`。

## 4. Larger-patch tradeoff

`{'full_rmse_reduction_vs_current': -0.0007628514660245828, 'code_scalar_reduction_vs_current': 0.32022029897718335, 'dictionary_scalar_increase_vs_current': 0.4666666666666667}`

## 5. 边界

- lower_overlap 与 current 的 patch dimension、dictionary cost 和 mean patch count匹配，因此可以直接比较。
- larger_patch 减少 code length，但增加 dictionary dimension；保留为独立 Pareto 点。
