# Laplacian graph-global ordering：matched KSVD 审计

> 日期：2026-08-01  
> normalized-Laplacian Fiedler rank；node sets、coverage、folds 和容量完全匹配。

## 1. 判定

**FAIL_LAPLACIAN_ORDERING_INVARIANTS**

## 2. Spectral coordinate audit

| graph group | n | canonical pair-order agreement | sign-invariant agreement | stable fraction | relative eigengap mean/median |
|---|---:|---:|---:|---:|---:|
| all | 72 | 0.8593 | 0.9599 | 0.8611 | 1.3134/0.0908 |
| regular | 24 | 0.8316 | 0.9340 | 0.8333 | 0.0356/0.0296 |
| small_world | 24 | 0.8871 | 0.9676 | 0.8750 | 0.0978/0.0845 |
| block | 24 | 0.8592 | 0.9782 | 0.8750 | 3.8068/3.6423 |

Coordinate stability checks：`{'mapped_relabel_rank_exact': False, 'mean_canonical_pair_order_agreement_at_least_080': True, 'stable_graph_fraction_at_least_two_thirds': True}`。

### Post-result tie diagnostic（不参与 gate）

| graph group | mean tied-node fraction | graphs with ties |
|---|---:|---:|
| all | 0.0061 | 0.0972 |
| regular | 0.0000 | 0.0000 |
| small_world | 0.0000 | 0.0000 |
| block | 0.0183 | 0.2917 |

该 diagnostic 是在正式 invariant failure 后用于定位原因，不改变预注册判定。tied nodes 的 Fiedler 值在 `1e-10` 内不可区分，其内部 rank 会依赖 tie-breaking。

## 3. Fold-balanced reconstruction means

| branch | stage | patch rel err | observed RMSE | observed F1 | full edge recall | overlap disagreement |
|---|---|---:|---:|---:|---:|---:|
| construction | raw | 0.0000 | 0.0000 | 1.0000 | 0.6845 | 0.0000 |
| construction | init | 0.5163 | 0.3858 | 0.8214 | 0.5783 | 0.1427 |
| construction | final | 0.4884 | 0.3627 | 0.8411 | 0.5917 | 0.1395 |
| construction | pca3 | 0.5572 | 0.4175 | 0.7695 | 0.5521 | 0.1214 |
| fiedler | raw | 0.0000 | 0.0000 | 1.0000 | 0.6845 | 0.0000 |
| fiedler | init | 0.5203 | 0.3880 | 0.8223 | 0.5904 | 0.1294 |
| fiedler | final | 0.4924 | 0.3651 | 0.8380 | 0.6012 | 0.1181 |
| fiedler | pca3 | 0.5489 | 0.4126 | 0.7780 | 0.5712 | 0.0745 |

## 4. Registered gates

- representation invariants：`{'node_sets_match': True, 'node_coverage_match': True, 'edge_coverage_match': True, 'pair_coverage_match': True, 'mapped_rank_match_one': False, 'mapped_patch_rate_one': True, 'mapped_transition_rate_one': False, 'construction_raw_gate': True, 'fiedler_raw_gate': True}`。
- coordinate stability gate：`False`。
- Fiedler FINAL vs construction FINAL RMSE relative reduction：`-0.0064`。
- comparative checks：`{'rmse_relative_reduction_at_least_002': False, 'disagreement_not_worse': True, 'observed_f1_not_worse': False, 'full_edge_recall_preserved': True, 'beats_fiedler_pca3': True, 'final_patch_better_all_folds': True}`。
- comparative gate：`False`。
- Fiedler optimization gate：`False`；mean patch INIT→FINAL reduction：`0.0536`。

## 5. 如何理解“位置”

- Fiedler rank 是每张完整图内部的 global spectral coordinate，不是跨图共享的物理坐标。
- node relabeling 不应改变它；一次边交换稳定性则检查它是否对轻微结构变化过度敏感。
- 即使 rank 稳定，也只有 matched reconstruction 改善后，才能说它对当前 patch/KSVD 表示有实际价值。
- 本轮不使用 labels，也不回答 LapPE 输入 GNN/Transformer 后的分类效果。
