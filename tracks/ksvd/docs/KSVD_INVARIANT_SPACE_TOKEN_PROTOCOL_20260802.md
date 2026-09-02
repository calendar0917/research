# Invariant-space KSVD patch token：优化协议

> 日期：2026-08-02  
> 状态：ID/slot decomposition 之后、优化结果之前冻结。

## 1. 动机

Matched decomposition 显示：

- global ID mapped replay 完全一致；
- 同一 patch node set 只做 local-slot permutation，KSVD graph-code cosine 约 `0.636`；
- numeric-ID sort 也只有约 `0.645`；
- patch-code support Jaccard 约 `0.13`。

因此不直接修改 Beam8 sampler，而是把 KSVD 从 arbitrary 45D adjacency-slot space 移到 permutation-invariant local-structure space。

## 2. 输入与 KSVD

每个 patch 先变成22D invariant descriptor：sorted degrees、sorted adjacency eigenvalues、density、triangle density。

在 train graphs 的 descriptors 上学习 overcomplete KSVD：

```text
feature dimension 22
K = 24
T = 3
updates = 25
```

dictionary、feature mean 和 sparse encoding 均严格 train-only。

## 3. 分支

- `INV_KSVD_BAG`；
- `INV_KSVD_TRUE_RELATION`；
- `INV_KSVD_SHUFFLED_RELATION`。

masked target、overlap/distance relation、ridge、fold split 和 relabel-resampling stability 沿用 invariant raw-token follow-up。

## 4. Gate

1. TRUE vs BAG RMSE 改善至少 `2%`；
2. TRUE vs SHUFFLED 改善至少 `2%`；
3. 3/3 folds TRUE 同时更好；
4. TRUE graph-embedding relabel cosine `>=0.90`；
5. TRUE overall RMSE 不得比 raw invariant TRUE `0.1247372` 恶化超过 `5%`；
6. dictionary/row/isolation invariants 通过。

全部通过：`ADOPT_INVARIANT_SPACE_KSVD_TOKEN`。

relation/stability 通过但信息损失超过5%：`INVARIANT_KSVD_STABLE_BUT_TOO_LOSSY`。

否则：`REJECT_INVARIANT_SPACE_KSVD_TOKEN`。

## 5. 边界

- 字典原子现在表示 degree/spectrum/density pattern，不再直接是 adjacency motif image；
- 该优化目标是稳定 patch token，不是恢复带编号局部邻接矩阵；
- 若通过，ordered-adjacency KSVD 仍可作为 auxiliary reconstruction channel，但不再作为主 graph token。
