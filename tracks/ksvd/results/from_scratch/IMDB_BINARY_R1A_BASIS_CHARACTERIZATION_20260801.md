# IMDB-BINARY R1-A label-free basis characterization

> 日期：2026-08-01  
> 输入：R0-D registered dictionaries；不重新训练、不使用 labels 作 gate  
> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R1A_BASIS_CHARACTERIZATION_PROTOCOL_20260801.md`

## 1. Frozen design

- raw IMDB-BINARY；stratified/grouped 各 5 folds。
- 复用 R0-D：s=7,d=21,K=12,T=2,updates=25，单 deterministic INIT，restart=0。
- top-activating patches：每 atom 在 held-out test 中取 absolute coefficient top-5。
- 主 gate：usage、cross-fold stability、nearest-real-patch proximity、top-patch diversity。
- graph labels 仅保存在 top-patch descriptive metadata，不进入任何 gate。

## 2. View summary

| view | nondead | INIT cross-fold | FINAL cross-fold | INIT nearest | FINAL nearest | Gaussian nearest | FINAL-Gaussian | top5 unique WALK | edge-regime dispersion | INIT->FINAL | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stratified | 12/12 | 0.5357 | 0.7529 | 1.0000 | 0.8865 | 0.6361 | 0.2504 | 2.5000 | 1.8609 | 0.5653 | FAIL |
| exact_isomorphism_grouped | 12/12 | 0.5614 | 0.7458 | 1.0000 | 0.8819 | 0.6375 | 0.2444 | 2.4500 | 2.1223 | 0.5739 | FAIL |

## 3. Fold details

### stratified

| fold | nondead | INIT nearest | FINAL nearest | Gaussian nearest | top5 unique WALK | edge-regime dispersion | INIT->FINAL cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12/12 | 1.0000 | 0.8823 | 0.6402 | 2.2500 | 2.3027 | 0.5281 |
| 1 | 12/12 | 1.0000 | 0.8823 | 0.6415 | 2.4167 | 2.0677 | 0.5928 |
| 2 | 12/12 | 1.0000 | 0.8892 | 0.6359 | 1.6667 | 1.6173 | 0.5576 |
| 3 | 12/12 | 1.0000 | 0.8998 | 0.6282 | 3.0000 | 1.5769 | 0.6030 |
| 4 | 12/12 | 1.0000 | 0.8788 | 0.6348 | 3.1667 | 1.7397 | 0.5450 |

### exact_isomorphism_grouped

| fold | nondead | INIT nearest | FINAL nearest | Gaussian nearest | top5 unique WALK | edge-regime dispersion | INIT->FINAL cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12/12 | 1.0000 | 0.8582 | 0.6391 | 2.3333 | 2.5902 | 0.5424 |
| 1 | 12/12 | 1.0000 | 0.8745 | 0.6355 | 2.0833 | 2.6121 | 0.6099 |
| 2 | 12/12 | 1.0000 | 0.8981 | 0.6428 | 2.2500 | 2.1777 | 0.5801 |
| 3 | 12/12 | 1.0000 | 0.8910 | 0.6379 | 2.8333 | 2.2455 | 0.5913 |
| 4 | 12/12 | 1.0000 | 0.8879 | 0.6323 | 2.7500 | 0.9860 | 0.5459 |

## 4. Registered conditions

### stratified

- [x] `final_nondead_all_folds`
- [x] `final_cross_fold_matched_cosine_mean_at_least_0_70`
- [x] `final_cross_fold_more_stable_than_init`
- [x] `final_nearest_real_patch_beats_gaussian_by_0_10`
- [ ] `final_nearest_real_patch_not_below_init`
- [x] `top5_unique_walk_mean_at_least_2`
- [x] `top5_edge_count_dispersion_at_least_1`

### exact_isomorphism_grouped

- [x] `final_nondead_all_folds`
- [x] `final_cross_fold_matched_cosine_mean_at_least_0_70`
- [x] `final_cross_fold_more_stable_than_init`
- [x] `final_nearest_real_patch_beats_gaussian_by_0_10`
- [ ] `final_nearest_real_patch_not_below_init`
- [x] `top5_unique_walk_mean_at_least_2`
- [x] `top5_edge_count_dispersion_at_least_1`

## 5. Decision

> **FAIL_R1A_BASIS_CHARACTERIZATION**

- next step：Do not add restarts or reopen classification. Either accept reconstruction-valid but weak structural characterization, or design a new multi-view compressor protocol.

## 6. Interpretation boundary

本轮不重新打开 classification claim。通过只说明 learned basis 具有 label-free compressor/basis-discovery 意义；不要求 atom 都可命名，也不表示它们是 task-optimal motifs。

## 7. Post-hoc protocol audit（不改变 classification）

Formal FAIL 只由 `final_nearest_real_patch_not_below_init` 触发。这个 condition 在执行后被确认存在结构性失配：

- deterministic maximin INIT 的每列就是一条 outer-train centered real patch 的归一化；
- 因而 INIT nearest-real absolute cosine 按构造恒为 `1.0`；
- FINAL 是 KSVD 更新后的 continuous direction，要求它不低于 `1.0` 等价于要求它仍精确等于某条训练 patch。

所以不能事后删除该 gate 并把 R1-A 改判为 PASS；正式结果保持
`FAIL_R1A_BASIS_CHARACTERIZATION`。但也不能把它解释成 learned basis 缺乏全部结构证据，因为其余 `6/7` 条件在两个 views 都通过：FINAL cross-fold stability 为 `0.753/0.746`，明显高于 INIT 的 `0.536/0.561`；FINAL nearest-real cosine 为 `0.886/0.882`，比 Gaussian 高 `0.250/0.244`；top activators 覆盖多种 WALK vectors 和 edge-density regimes。

R1-A 的 overall characterization 因此标记为 **protocol-misspecified / inconclusive**。下一步只做无 gate 的 grouped-fold consensus basis atlas，直接展示 matched atoms 与 top-activating real patches，不再用同一结果重设 confirmatory threshold。
