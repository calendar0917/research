# Structural canonical slot KSVD 审计协议

> 日期：2026-08-02  
> 状态：结果前冻结。

## 1. 问题

本轮不把45D patch adjacency 换成 invariant summary。冻结 Beam8 产生的 patch node sets，只改变 induced adjacency 的 local-slot order，直接检验：不用 numeric node ID 排序，能否同时改善 KSVD 重编号稳定性与 stitched graph reconstruction。

## 2. Matched branches

1. `CONSTRUCTION`：Beam8 原始 construction order；
2. `ID_SORT`：按 numeric node ID；
3. `SIGNATURE`：center、patch distance、degree、triangle participation、3-round WL color；最终 tie 仍按 ID，作为 heuristic control；
4. `CANONICAL`：完整 unrooted exact canonical adjacency；
5. `ROOTED_CANONICAL`：把 patch center 作为 singleton color 的 exact canonical adjacency；
6. `OVERLAP_CANONICAL`：第一 patch rooted canonical；后续把上一 patch 中的 shared canonical slots 作为有序 singleton colors，再 canonicalize 其余节点。

所有 branches 保留完整45D upper-triangle adjacency、相同 patch node sets、patch count、coverage、folds、`K24/T3/updates25` 和 stitching。

## 3. Exact canonicalization

使用 ordered color refinement + individualization search。canonical adjacency code 在所有允许排列中取 lexicographic minimum。node ID 只允许：

- 作为返回某个等价 automorphism order 的内部记账；
- 存入 slot-to-global-node map 供 stitching 使用。

node ID 不进入 canonical adjacency code。若多个 node orders 产生相同最优 code，记录 automorphism ambiguity；不把 node-map 唯一性与 adjacency-vector 唯一性混为一谈。

## 4. 两层稳定性

### A. Frozen patch-set mapped relabel

同一 patch node sets 随全图 permutation 搬运，然后各 branch 独立重新排序。报告：

- patch-vector exact match；
- matched KSVD-code cosine/support Jaccard；
- graph pooled-code cosine；
- transition slot-map exact match。

### B. Relabel + Beam8 resampling

重编号后用相同 RNG seed 重新采样，再独立排序。报告 pooled-code cosine。该层同时包含 sampler patch-selection 差异。

## 5. Reconstruction

在 outer-train graphs 学习 branch-specific dictionary，并在 held-out graphs 报告：

- patch relative reconstruction error；
- observed-pair RMSE/F1；
- full-adjacency RMSE、recall、F1；
- repeated-pair disagreement；
- dictionary nondead/effective atoms。

## 6. 判定

优先支持某个 structural-slot branch 需要：

1. frozen-set vector match >=0.999；
2. matched code cosine >=0.99；
3. pooled cosine >=0.99；
4. 3/3 folds invariants；
5. observed RMSE 不比 construction 恶化超过2%；
6. full-adjacency RMSE 不比 construction 恶化超过2%。

若 canonical vector 稳定但 reconstruction 明显恶化，结论为 canonical slots 修复 invariance、但破坏 KSVD coordinate geometry。若 overlap-rooted 同时通过稳定性与 reconstruction gates，则保留完整-adjacency canonical KSVD 路线。

## 7. 边界

- frozen-set 成功不保证 Beam8 sampler 本身严格 relabel-equivariant；
- canonical adjacency 可恢复 patch topology；恢复带全局编号的图仍需要 slot-to-node/transition sidecar；
- automorphism-equivalent nodes 不存在纯结构意义下的唯一绝对身份。
