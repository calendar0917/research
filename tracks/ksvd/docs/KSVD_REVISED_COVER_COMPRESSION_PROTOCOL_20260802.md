# Revised cover configurations：matched KSVD follow-up protocol

> 日期：2026-08-02  
> 状态：cover Pareto 结果可见后注册；本 follow-up 的 KSVD 结果不可见前冻结。

## 1. 触发原因

预注册 27-cell RAW Pareto audit 发现当前 `s10/o5/m1.5` 被以下两个 cells 支配：

- `s10/o3/m1.5`：相同 patch count 和 raw pair slots，更高 edge/pair coverage、更低 RAW full RMSE；
- `s12/o4/m1.5`：更少 patches、近似 raw pair slots，更高 coverage、更低 RAW full RMSE。

本轮只验证这些 RAW dominators 在相同 KSVD capacity 下是否仍成立，不再搜索新 cover 参数。

## 2. Frozen branches

```text
current: s10/o5/m1.5
lower_overlap: s10/o3/m1.5
larger_patch: s12/o4/m1.5
cover_seed = 870101（精确复现 Pareto audit）
graph bank = 72 graphs, seed 810001
K=24, T=3, T_min=1, updates=25
3-fold graph isolation
PCA rank=3
```

## 3. 成本

分别报告：

- mean patch count；
- dictionary scalars `C(s,2)*K`；
- mean code scalars/graph `patch_count*T`；
- raw pair slots。

不同 patch dimension 不以单一任意权重合并成本。

## 4. Registered comparison

`lower_overlap` 替换 current 必须满足：

1. mean patch count、dictionary scalars 和 code scalars不增加；
2. FINAL full adjacency RMSE 至少降低 `0.02` relative；
3. FINAL observed RMSE不恶化超过 `0.01` relative；
4. FINAL full edge recall 和 F1 均不低于 current；
5. RAW invariants 和 3/3 fold patch FINAL<INIT 通过。

`larger_patch` 作为独立 Pareto 点报告；由于 dictionary dimension 增加，不要求它无条件替换 current。

## 5. 判定

- `ADOPT_LOWER_OVERLAP_COVER_V2`：lower-overlap gate 通过；
- `RAW_COVER_GAIN_LOST_AFTER_COMPRESSION`：RAW 支配，但 FINAL gate 不通过；
- `LARGER_PATCH_ALTERNATIVE_PARETO_POINT`：s12/o4 在 full RMSE/code cost 上形成有意义替代；
- `FAIL_REVISED_COVER_INVARIANTS`：RAW 或 fold isolation 失败。
