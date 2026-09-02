# MolHIV Beam8 cover seed-diversity audit

> 日期：2026-08-16  
> 范围：OGB official-train only；不读取 official-valid/test 分子。

## 结论

**BEAM_SEED_DIVERSITY_MEANINGFUL**

该审计只比较真实 `slot_nodes` patch 集合；token shuffle、chain shuffle 和 mapping shuffle 均不计为 cover 多样性。

## 汇总（仅统计至少一个连通分量大于 patch size 的分子）

| cover | changed graphs | unique covers | exact pair | patch Jaccard | node occurrence L1 | edge occurrence L1 | variable nodes | variable edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Beam8 | 0.9639 | 6.267 | 0.1246 | 0.1985 | 0.1153 | 0.1140 | 0.7565 | 0.7551 |
| random BFS | 1.0000 | 7.754 | 0.0116 | 0.1833 | 0.2157 | 0.2164 | 0.9637 | 0.9776 |

## 协议边界

- 抽样 official-train 分子：`1000`。
- cover seeds：`[20260813, 20260814, 20260815, 20260816, 20260817, 20260818, 20260819, 20260820]`。
- patch size / overlap：`8 / 2`。
- 标签未参与 cover 构造、选择或审计。
- `official_valid_evaluations = 0`，`official_test_evaluations = 0`。
