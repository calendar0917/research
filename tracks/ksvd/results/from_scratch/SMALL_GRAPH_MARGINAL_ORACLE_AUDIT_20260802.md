# Small-graph exhaustive marginal coverage 审计

> 日期：2026-08-02  
> exact one-step candidate enumeration；不声称全链 global optimum。

## 1. 判定

**HEURISTIC_HAS_MATERIAL_ORACLE_GAP**

## 2. Mean results

| branch | edge cover | pair cover | node cover | new pairs/patch | edge multiplicity CV |
|---|---:|---:|---:|---:|---:|
| target_edge | 0.7358 | 0.4420 | 0.9630 | 12.3097 | 0.4265 |
| exhaustive_marginal | 0.8737 | 0.4458 | 0.9815 | 12.4125 | 0.3633 |

## 3. Gaps

- edge coverage gap：`0.1379`；
- pair coverage gap：`0.0038`；
- oracle edge better graph fraction：`1.0000`；
- oracle edge not-worse graph fraction：`1.0000`。

## 4. 边界

- exhaustive branch 只保证当前一步的 lexicographic marginal objective 最优，后续 chain 仍是 greedy。
- 小图结果用于量化 heuristic gap；不能直接把枚举算法扩展到 50 nodes。
- 若 gap 明显，下一步应做 candidate beam/search，而不是继续修改 target deficit 权重。
