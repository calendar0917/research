# Marginal candidate beam continuous-cover 审计

> 日期：2026-08-02  
> 72 graphs；single chain；exact overlap；RAW cover comparison。

## 1. 判定

**BEAM_IMPROVES_COVERAGE_BELOW_GATE**

## 2. Mean results

| branch | edge cover | pair cover | RAW full RMSE | new pairs/patch | bridge length |
|---|---:|---:|---:|---:|---:|
| target_s10_o5 | 0.6855 | 0.4935 | 0.3537 | 34.3655 | 1.0404 |
| beam_s10_o5 | 0.8172 | 0.4718 | 0.2667 | 32.9346 | 0.0000 |
| target_s10_o3 | 0.7654 | 0.5600 | 0.3053 | 39.0466 | 1.0497 |
| beam_s10_o3 | 0.8844 | 0.5313 | 0.2105 | 37.1272 | 0.0000 |

## 3. Registered comparisons

`{'o5': {'edge_coverage_gain': 0.13171451149588664, 'pair_coverage_gain': -0.02165532879818599, 'raw_full_rmse_reduction': 0.24584324063945334, 'checks': {'edge_gain_at_least_003': True, 'pair_not_worse_by_001': False, 'full_rmse_reduction_at_least_002': True}, 'passed': False}, 'o3': {'edge_coverage_gain': 0.11895922661596459, 'pair_coverage_gain': -0.028741496598639338, 'raw_full_rmse_reduction': 0.31057320588644083, 'checks': {'edge_gain_at_least_003': True, 'pair_not_worse_by_001': False, 'full_rmse_reduction_at_least_002': True}, 'passed': False}}`

## 4. 边界

- beam 近似每一步 marginal objective，不是全链 global optimum。
- 第一 patch 与 target branch 共用同类 seed；差异来自后续 candidate selection。
- beam 不显式追逐单一 target edge，因此 bridge length 字段为 0；连续性由 exact overlap 和 connected patch 保证。
