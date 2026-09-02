# Compact local random-walk relation ablation（2026-07-28）

## 1. Protocol

- 8,000-graph subset；只使用 6,400 个 official-train graphs 的三个 outer scaffold folds。
- 直接复用上一轮保存的 occurrence logits；base 不重训、不解冻，因此 baseline 完全相同。
- official valid/test 编码与评估均为 **0**。
- 每个 walk operator 固定压缩为 `70` 维：`diag(C)`、`row_sum(C)`、trace/off-diagonal mass、prototype semantic similarity、support fraction 与 active mass。
- 使用 reversible stationary flow `Q_t = diag(pi)P^t`；每个 ablation 独立训练 zero-init bounded linear residual，real/shuffled 协议完全相同。
- 这是 exact distance 1–2 之后的 operator-value 实验；沿用 `mean gain >= +0.005`、赢 base 至少 2/3、且 real mean > shuffled mean 的严格 promotion gate。

## 2. Single-vocabulary random-walk means

| Vocabulary | Walk operator | Dim | Base | Real | Shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Farthest | rw2_full | 70 | 0.743729 | 0.745693 | 0.744712 | +0.001964 | +0.000981 | 2/3 | 2/3 |
| Farthest | rw2_nonreturn | 70 | 0.743729 | 0.745946 | 0.745989 | +0.002217 | -0.000042 | 2/3 | 2/3 |
| Farthest | rw2_exact_distance2 | 70 | 0.743729 | 0.745966 | 0.745965 | +0.002237 | +0.000001 | 2/3 | 1/3 |
| Farthest | rw1_rw2_full | 140 | 0.743729 | 0.746456 | 0.745065 | +0.002727 | +0.001391 | 2/3 | 3/3 |
| Farthest | rw1_rw2_nonreturn | 140 | 0.743729 | 0.746651 | 0.746305 | +0.002922 | +0.000346 | 2/3 | 2/3 |
| Farthest | rw1_rw2_exact_distance2 | 140 | 0.743729 | 0.746633 | 0.746306 | +0.002904 | +0.000327 | 2/3 | 2/3 |
| Farthest | rw1_rw2_nonreturn_mix | 70 | 0.743729 | 0.745999 | 0.745573 | +0.002271 | +0.000426 | 2/3 | 1/3 |
| Scaffold | rw2_full | 70 | 0.743614 | 0.744766 | 0.742866 | +0.001152 | +0.001900 | 2/3 | 3/3 |
| Scaffold | rw2_nonreturn | 70 | 0.743614 | 0.745309 | 0.743022 | +0.001695 | +0.002287 | 2/3 | 3/3 |
| Scaffold | rw2_exact_distance2 | 70 | 0.743614 | 0.745243 | 0.743003 | +0.001629 | +0.002240 | 2/3 | 3/3 |
| Scaffold | rw1_rw2_full | 140 | 0.743614 | 0.745369 | 0.742991 | +0.001755 | +0.002377 | 3/3 | 3/3 |
| Scaffold | rw1_rw2_nonreturn | 140 | 0.743614 | 0.745779 | 0.742216 | +0.002166 | +0.003564 | 3/3 | 3/3 |
| Scaffold | rw1_rw2_exact_distance2 | 140 | 0.743614 | 0.745760 | 0.742207 | +0.002146 | +0.003554 | 3/3 | 3/3 |
| Scaffold | rw1_rw2_nonreturn_mix | 70 | 0.743614 | 0.745388 | 0.743626 | +0.001775 | +0.001762 | 2/3 | 3/3 |

## 3. Farthest + scaffold random-walk probability ensembles

| Walk operator | Dim/family | Fold 0 real | Fold 1 real | Fold 2 real | Base mean | Real mean | Shuffled mean | Real−base | Real−shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rw2_full | 70 | 0.766659 | 0.701705 | 0.802798 | 0.755736 | 0.757054 | 0.755619 | +0.001318 | +0.001435 |
| rw2_nonreturn | 70 | 0.766741 | 0.701422 | 0.804028 | 0.755736 | 0.757397 | 0.756108 | +0.001661 | +0.001289 |
| rw2_exact_distance2 | 70 | 0.766690 | 0.701358 | 0.803926 | 0.755736 | 0.757325 | 0.756097 | +0.001588 | +0.001228 |
| rw1_rw2_full | 140 | 0.766850 | 0.702193 | 0.804041 | 0.755736 | 0.757694 | 0.755944 | +0.001958 | +0.001751 |
| rw1_rw2_nonreturn | 140 | 0.766561 | 0.701556 | 0.805092 | 0.755736 | 0.757736 | 0.755889 | +0.002000 | +0.001848 |
| rw1_rw2_exact_distance2 | 140 | 0.766550 | 0.701535 | 0.805073 | 0.755736 | 0.757719 | 0.755861 | +0.001983 | +0.001858 |
| rw1_rw2_nonreturn_mix | 70 | 0.766700 | 0.701217 | 0.804429 | 0.755736 | 0.757449 | 0.756458 | +0.001713 | +0.000990 |

## 4. Assignment-specific diagnostic ranking

按 `real − shuffled` 优先、再按 `real − base` 排序：

| Rank | Scope | Walk operator | Real | Real−base | Real−shuffled | Wins shuffled |
|---:|---|---|---:|---:|---:|---:|
| 1 | family:scaffold_facility | rw1_rw2_nonreturn | 0.745779 | +0.002166 | +0.003564 | 3/3 |
| 2 | family:scaffold_facility | rw1_rw2_exact_distance2 | 0.745760 | +0.002146 | +0.003554 | 3/3 |
| 3 | family:scaffold_facility | rw1_rw2_full | 0.745369 | +0.001755 | +0.002377 | 3/3 |
| 4 | family:scaffold_facility | rw2_nonreturn | 0.745309 | +0.001695 | +0.002287 | 3/3 |
| 5 | family:scaffold_facility | rw2_exact_distance2 | 0.745243 | +0.001629 | +0.002240 | 3/3 |
| 6 | family:scaffold_facility | rw2_full | 0.744766 | +0.001152 | +0.001900 | 3/3 |

## 5. Decision

- 严格 promotion：**False**。
- Pair 中 assignment-specific 最强 RW：`rw1_rw2_exact_distance2`；real−base `+0.001983`，real−shuffled `+0.001858`。
- 相对对应 exact-distance baseline 最好的 pair RW：`rw1_rw2_nonreturn`，mean delta `-0.000282`，wins `0/3`。

## 6. Direct comparison with exact distance operators

| Scope | RW operator | Exact reference | RW−exact mean | Wins |
|---|---|---|---:|---:|
| Farthest | rw2_exact_distance2 | distance_2 | -0.000341 | 0/3 |
| Farthest | rw1_rw2_exact_distance2 | distance_1_2 | -0.000145 | 2/3 |
| Farthest | rw1_rw2_nonreturn | distance_1_2 | -0.000126 | 2/3 |
| Farthest | rw1_rw2_nonreturn_mix | distance_1_2 | -0.000778 | 0/3 |
| Scaffold | rw2_exact_distance2 | distance_2 | -0.000475 | 0/3 |
| Scaffold | rw1_rw2_exact_distance2 | distance_1_2 | -0.000499 | 0/3 |
| Scaffold | rw1_rw2_nonreturn | distance_1_2 | -0.000480 | 0/3 |
| Scaffold | rw1_rw2_nonreturn_mix | distance_1_2 | -0.000871 | 0/3 |
| Pair | rw2_exact_distance2 | distance_2 | -0.000514 | 0/3 |
| Pair | rw1_rw2_exact_distance2 | distance_1_2 | -0.000298 | 0/3 |
| Pair | rw1_rw2_nonreturn | distance_1_2 | -0.000282 | 0/3 |
| Pair | rw1_rw2_nonreturn_mix | distance_1_2 | -0.000569 | 0/3 |

判定原则：只有 RW 在相同 frozen base、相同 compact statistics 下稳定超过对应 exact-distance operator，才能宣称 random walk 本身提供了额外价值。

## 7. Research conclusion

- 所有映射到 exact baseline 的 pair RW 是否均为 0/3 wins：**True**。
- Exact distance 1+2 pair：`0.758018`；最佳对应 RW pair：`0.757736`，差值 `-0.000282`。
- `rw1_stationary = diag(pi)P` 在无向图上等价于 uniformly normalized directed-edge mask；真正新增的只有二步 path/degree weighting。
- 二步 weighting 没有增加 operator value；return removal 有益于 RW 内部比较，但仍稳定弱于 uniform exact distance 1+2。
- 因而不晋级该 random-walk family，也不触碰 full official valid/test。下一步应把精力从 relation operator 转向 task-matched vocabulary / prototype selection。