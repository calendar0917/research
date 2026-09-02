# ID-free invariant patch token + relation：跟进协议

> 日期：2026-08-02  
> 状态：首轮 relation audit 之后、跟进结果之前冻结。

## 1. 动机

首轮 masked audit 中，KSVD_TRUE_RELATION 相对 KSVD_BAG 和 SHUFFLED 均有强增益，但 node-relabel + resampling graph-embedding cosine 只有约 `0.72`。这说明 relation binding 有用，当前 ordered-adjacency KSVD token 不稳定。

本轮只替换 local token，不改变 Beam8 covers、relation features、masked target、fold split 或 ridge capacity。

## 2. ID-free token

每个 context patch 使用与 masked target 同构的 22D permutation-invariant descriptor：

```text
10 sorted normalized degrees
10 sorted normalized adjacency eigenvalues
edge density
triangle density
```

该 token 不读取 global node IDs 或 construction slots。IDs 只在数据层用于计算 overlap 与 center distance。

## 3. 三个分支

- `INVARIANT_BAG`：其他 invariant tokens 的 mean/std/max；
- `INVARIANT_TRUE_RELATION`：加入真实 overlap/distance weighted token context；
- `INVARIANT_SHUFFLED_RELATION`：同容量 graph-local binding shuffle。

设置沿用首轮：72 graphs、Beam8/R1、3-fold graph isolation、ridge `alpha=0.01`。

## 4. Gate

1. TRUE 相对 BAG masked overall RMSE 改善至少 `2%`；
2. TRUE 相对 SHUFFLED 改善至少 `2%`；
3. 3/3 folds TRUE 同时更好；
4. node-relabel + resampling graph-embedding cosine `>=0.90`；
5. invariants 全部通过。

全部通过：`ID_FREE_PATCH_RELATION_SUBSTRATE_SUPPORTED`。

若 relation gate 通过但 stability 失败：`INVARIANT_TOKEN_RELATION_USEFUL_BUT_SAMPLER_UNSTABLE`。

否则：`REJECT_INVARIANT_PATCH_RELATION_SUBSTRATE`。

## 5. 边界

- 该 descriptor 是手工 invariant control，不是最终 learned encoder；
- 通过意味着可以把 numeric node IDs 从模型输入中移除，并保留 patch-relation signal；
- 它不证明 KSVD 应被删除，KSVD 可在后续作为辅助 token 或通过 consistency training 修复；
- 无 graph labels，不单独证明分类增益。
