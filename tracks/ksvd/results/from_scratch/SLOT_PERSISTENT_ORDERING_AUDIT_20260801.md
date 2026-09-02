# Slot-persistent continuous cover：KSVD matched audit

> 日期：2026-08-01  
> node sets、cover budget 和 train/test folds 完全匹配；只改变 local ordering。

## 1. 判定

**REJECT_SLOT_PERSISTENT_ORDERING**

## 2. Ordering invariants

`{'node_sets_match': True, 'node_coverage_match': True, 'edge_coverage_match': True, 'pair_coverage_match': True, 'persistent_slot_rate_one': True, 'mapped_patch_rate_one': True, 'mapped_transition_rate_one': True}`

## 3. Fold-balanced stage means

| branch | stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |
|---|---|---:|---:|---:|---:|---:|
| construction | raw | 0.0000 | 0.0000 | 1.0000 | 0.6845 | 0.0000 |
| construction | init | 0.5163 | 0.3858 | 0.8214 | 0.5783 | 0.1427 |
| construction | final | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| construction | pca3 | 0.5572 | 0.4175 | 0.7695 | 0.5521 | 0.1214 |
| persistent | raw | 0.0000 | 0.0000 | 1.0000 | 0.6845 | 0.0000 |
| persistent | init | 0.5526 | 0.4153 | 0.7878 | 0.5633 | 0.1252 |
| persistent | final | 0.5411 | 0.4053 | 0.7956 | 0.5757 | 0.1243 |
| persistent | pca3 | 0.6028 | 0.4561 | 0.7204 | 0.5467 | 0.0753 |

## 4. Registered comparison

- persistent FINAL vs construction FINAL observed RMSE relative reduction：`-0.1173`；
- overlap disagreement relative reduction：`0.1091`；
- comparisons：`{'rmse_relative_reduction_at_least_002': False, 'disagreement_relative_reduction_at_least_002': True, 'observed_f1_not_worse': False, 'full_edge_recall_not_worse': False, 'beats_persistent_pca3': True}`；
- persistent mean patch INIT→FINAL reduction：`0.0208`；
- persistent optimization gate：`False`。

## 5. 解释边界

- persistent slots 只运输同一条 traversal 中的局部坐标，不产生跨图绝对位置。
- node sets 和 edge coverage 完全不变，因此差异可以归因于 ordering/coordinate continuity，而不是 sampler。
- 若 ordering 无法通过 comparison，下一步必须让 learner 显式使用 transition map，而不是继续手工排序。
