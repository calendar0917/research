# Compact prototype-distance bin ablation（2026-07-28）

## 1. Protocol

- 8,000-graph subset；只使用 6,400 个 official-train graphs 的三个 outer scaffold folds。
- 直接复用上一轮保存的 occurrence logits；base 不重训、不解冻，因此 baseline 完全相同。
- official valid/test 编码与评估均为 **0**。
- 每个距离 bin 固定压缩为 `70` 维：`diag(C)`、`row_sum(C)`、trace/off-diagonal mass、prototype semantic similarity、pair fraction 与 active mass。
- 每个 ablation 独立训练 zero-init bounded linear residual；real 与 node-assignment-shuffled 使用完全相同协议。
- 这是尺度定位实验；沿用 `mean gain >= +0.005`、赢 base 至少 2/3、且 real mean > shuffled mean 的严格 promotion gate。

## 2. Single vocabulary means

| Vocabulary | Distance bins | Dim | Base | Real | Shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Farthest | distance_1 | 70 | 0.743729 | 0.746600 | 0.744365 | +0.002871 | +0.002235 | 2/3 | 3/3 |
| Farthest | distance_2 | 70 | 0.743729 | 0.746307 | 0.746130 | +0.002578 | +0.000177 | 2/3 | 3/3 |
| Farthest | distance_1_2 | 140 | 0.743729 | 0.746778 | 0.746420 | +0.003049 | +0.000358 | 2/3 | 2/3 |
| Farthest | distance_3plus | 70 | 0.743729 | 0.746019 | 0.746349 | +0.002291 | -0.000330 | 2/3 | 1/3 |
| Farthest | connected | 210 | 0.743729 | 0.746948 | 0.746936 | +0.003219 | +0.000012 | 2/3 | 1/3 |
| Farthest | disconnected | 70 | 0.743729 | 0.744199 | 0.744045 | +0.000470 | +0.000154 | 2/3 | 2/3 |
| Scaffold | distance_1 | 70 | 0.743614 | 0.745192 | 0.743262 | +0.001579 | +0.001930 | 3/3 | 3/3 |
| Scaffold | distance_2 | 70 | 0.743614 | 0.745718 | 0.743224 | +0.002104 | +0.002494 | 2/3 | 3/3 |
| Scaffold | distance_1_2 | 140 | 0.743614 | 0.746259 | 0.742512 | +0.002646 | +0.003748 | 3/3 | 3/3 |
| Scaffold | distance_3plus | 70 | 0.743614 | 0.744109 | 0.745423 | +0.000496 | -0.001314 | 1/3 | 1/3 |
| Scaffold | connected | 210 | 0.743614 | 0.746107 | 0.743302 | +0.002493 | +0.002805 | 2/3 | 3/3 |
| Scaffold | disconnected | 70 | 0.743614 | 0.744372 | 0.744150 | +0.000759 | +0.000222 | 2/3 | 3/3 |

## 3. Farthest + scaffold probability ensembles

| Distance bins | Dim/family | Fold 0 real | Fold 1 real | Fold 2 real | Base mean | Real mean | Shuffled mean | Real−base | Real−shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_1 | 70 | 0.766917 | 0.701691 | 0.804270 | 0.755736 | 0.757626 | 0.755865 | +0.001890 | +0.001761 |
| distance_2 | 70 | 0.767381 | 0.701839 | 0.804295 | 0.755736 | 0.757838 | 0.756467 | +0.002102 | +0.001372 |
| distance_1_2 | 140 | 0.767154 | 0.701783 | 0.805118 | 0.755736 | 0.758018 | 0.756061 | +0.002282 | +0.001957 |
| distance_3plus | 70 | 0.766891 | 0.700877 | 0.803097 | 0.755736 | 0.756955 | 0.757999 | +0.001219 | -0.001044 |
| connected | 210 | 0.768453 | 0.701563 | 0.805130 | 0.755736 | 0.758382 | 0.757045 | +0.002646 | +0.001338 |
| disconnected | 70 | 0.767809 | 0.699073 | 0.801835 | 0.755736 | 0.756239 | 0.756158 | +0.000503 | +0.000081 |

## 4. Assignment-specific diagnostic ranking

按 `real − shuffled` 优先、再按 `real − base` 排序：

| Rank | Scope | Distance bins | Real | Real−base | Real−shuffled | Wins shuffled |
|---:|---|---|---:|---:|---:|---:|
| 1 | family:scaffold_facility | distance_1_2 | 0.746259 | +0.002646 | +0.003748 | 3/3 |
| 2 | family:scaffold_facility | connected | 0.746107 | +0.002493 | +0.002805 | 3/3 |
| 3 | family:scaffold_facility | distance_2 | 0.745718 | +0.002104 | +0.002494 | 3/3 |
| 4 | family:farthest | distance_1 | 0.746600 | +0.002871 | +0.002235 | 3/3 |
| 5 | pair | distance_1_2 | 0.758018 | +0.002282 | +0.001957 | 3/3 |
| 6 | family:scaffold_facility | distance_1 | 0.745192 | +0.001579 | +0.001930 | 3/3 |

## 5. Decision

- 严格 promotion：**False**。
- Pair 中 assignment-specific 最强尺度：`distance_1_2`；real−base `+0.002282`，real−shuffled `+0.001957`。
- 只有当某个尺度同时稳定超过 occurrence 与 shuffled，才值得将该尺度替换为 diffusion/random-walk operator。

## 6. Comparison with the previous full 32×32×5 relation

- Full relation：`2650` params/family，pair mean `0.758440`。
- Compact connected：`210` params/family，pair mean `0.758382`。
- AUC difference：`-0.000058`；即用约 1/12.6 的 relation 参数基本复现 full relation 的 pair AUC。
- 结合 shuffled control，真正可定位的信号集中在 distance 1–2；distance 3+ 不应进入下一版 operator。
- 下一步应只测试 local 1–2 hop diffusion / random-walk composition，并保持 frozen base、bounded residual 和 matched shuffled control。