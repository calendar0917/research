# Marginal candidate beam：cost–coverage sensitivity protocol

> 日期：2026-08-02  
> 状态：beam32/restarts2 compression 结果可见后注册；sensitivity 结果不可见前冻结。

## 1. 研究问题

`beam=32,restarts=2` 已显著改善 RAW 与 KSVD full reconstruction，但它仍是工程参数。本轮固定 `s10/o3/m1.5`，比较：

```text
TARGET
B8_R1
B16_R1
B32_R1
B32_R2
```

同一 72-graph bank，cover seed `900101`。每个 graph/branch 记录 wall-clock sampling seconds。

## 2. Metrics

- edge/pair coverage；
- RAW full adjacency RMSE；
- mean seconds/graph；
- exact overlap、connected、single-chain invariants。

## 3. Pareto

minimize seconds/graph 和 RAW full RMSE，maximize edge/pair coverage。不得只因 beam 更大就选择它。

推荐 knee 的冻结规则：

1. invariants 全过；
2. 相对 TARGET edge coverage gain >= `0.08`；
3. RAW full RMSE reduction >= `0.20`；
4. 在满足 1–3 的 cells 中，选择最小 mean seconds/graph；
5. 若多个 cells 时间差 <=5%，选择 edge coverage 更高者。

## 4. 判定

- `SMALL_BEAM_SUFFICIENT`：B8 或 B16 被选为 knee；
- `BEAM32_REQUIRED`：只有 B32 满足效果 gate；
- `NO_STABLE_BEAM_KNEE`：没有 cell 满足 gate；
- `FAIL_BEAM_SENSITIVITY_INVARIANTS`：任一正式 cell 的 cover contract 失败。
