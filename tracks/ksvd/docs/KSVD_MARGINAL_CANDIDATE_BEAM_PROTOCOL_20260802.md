# Continuous cover：marginal candidate beam follow-up protocol

> 日期：2026-08-02  
> 状态：small-graph oracle 结果可见后注册；beam 正式结果不可见前冻结。

## 1. 触发原因

18-node exhaustive one-step marginal audit 显示：相同 budget/continuity 下，exhaustive marginal edge coverage 比 target-edge heuristic 高 `13.79` 个百分点，24/24 graphs 更好；pair coverage几乎相同。

本轮测试一个能扩展到 50 nodes 的近似 candidate search，避免继续手调 target deficit 权重。

## 2. Candidate beam

每一步：

1. 从 previous patch 枚举所有 connected retained subsets；
2. 按 retained nodes 通向未覆盖边的 boundary potential 排序，保留 top `32`；
3. 每个 retained subset 做 `2` 次 greedy frontier fill；
4. fill 时只加入 previous patch 外、与当前 partial patch 相邻的节点；
5. 对完整 candidate 按以下 lexicographic objective 选择：

```text
maximize new true edges
then maximize new observed pairs
then maximize new nodes
then maximize total induced edges
```

第一张 patch 完全复用 target-edge seed patch。所有后续 patch 保持 exact overlap、connected 和单链连续。

## 3. Frozen branches

72 graphs、`m=1.5`，比较：

```text
target_s10_o5
beam_s10_o5
target_s10_o3
beam_s10_o3
```

cover seed `890101`。本轮先评 RAW cover，不训练 KSVD；若 beam 形成明确 Pareto gain，再单独注册 compression follow-up。

## 4. Gate

beam 相对同 `s/o` target 必须：

1. 72/72 feasible；
2. exact overlap、patch connected、single-chain continuity 全部为 1；
3. mean edge coverage 至少提高 `0.03`；
4. pair coverage不降低超过 `0.01`；
5. RAW full adjacency RMSE 至少降低 `0.02` relative。

## 5. 判定

- `PASS_MARGINAL_CANDIDATE_BEAM`：至少一个 overlap branch 通过；
- `BEAM_IMPROVES_COVERAGE_BELOW_GATE`：方向为正但效果量不足；
- `REJECT_SCALABLE_MARGINAL_BEAM`：没有稳定收益；
- `FAIL_MARGINAL_BEAM_INVARIANTS`：feasibility/continuity 失败。
