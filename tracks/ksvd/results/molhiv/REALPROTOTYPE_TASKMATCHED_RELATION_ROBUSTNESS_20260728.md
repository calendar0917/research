# Task-matched relation sidecar: prototype-seed robustness（2026-07-28）

## 1. Why this check was required

cap `0.3125` 在 prototype seed `20260728` 上通过开发 gate，但该 cap 是 near-gate 后的一步优化。为避免把单个随机 candidate reservoir 当作稳定结论，本轮固定所有 downstream 配置，只把 task-matched candidate-bank/prototype seed 改为 `20260729`。official valid/test 编码与评估仍为 **0**。

## 2. Robustness results

| Prototype seed | Frozen base | Relation selector | Base | Real | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 20260728 | Broad occurrence pair | Nested task-aware | 0.755736 | 0.761675 | 0.755729 | +0.005939 | +0.005947 | 3/3 | 3/3 |
| 20260728 | Broad occurrence pair | Nested shuffled-label | 0.755736 | 0.758332 | 0.757348 | +0.002596 | +0.000984 | 3/3 | 2/3 |
| 20260728 | Broad relation pair | Nested task-aware | 0.758018 | 0.762375 | 0.755966 | +0.004357 | +0.006409 | 3/3 | 3/3 |
| 20260728 | Broad relation pair | Nested shuffled-label | 0.758018 | 0.759346 | 0.757899 | +0.001328 | +0.001447 | 3/3 | 2/3 |
| 20260729 | Broad occurrence pair | Nested task-aware | 0.755736 | 0.757620 | 0.756789 | +0.001884 | +0.000831 | 3/3 | 2/3 |
| 20260729 | Broad occurrence pair | Nested shuffled-label | 0.755736 | 0.760141 | 0.755371 | +0.004404 | +0.004770 | 3/3 | 3/3 |
| 20260729 | Broad relation pair | Nested task-aware | 0.758018 | 0.758318 | 0.757188 | +0.000300 | +0.001130 | 1/3 | 3/3 |
| 20260729 | Broad relation pair | Nested shuffled-label | 0.758018 | 0.760862 | 0.756814 | +0.002844 | +0.004048 | 3/3 | 3/3 |

## 3. Stability diagnosis

- task-aware selected rows 两个 prototype seeds 在每个 fold 的 exact overlap 都是 `[0, 0, 0]`；optimal-matching cosine 为 `[0.5594431161880493, 0.5818025469779968, 0.607628345489502]`，mean `0.582958`。
- shuffled-label bank 的 matching cosine mean 为 `0.589655`。因此变化不是简单的 atom permutation，而是 candidate reservoir 改变后选出了不同的 vocabulary geometry。
- prototype seed `20260728` 上，broad occurrence + task-aware sidecar gain 为 `+0.005939`；seed `20260729` 仅为 `+0.001884`，assignment-specific margin 也降至 `+0.000831`。
- 更关键的是，seed `20260729` 上 shuffled-label selector 反而比 task-aware selector 更强。这说明当前 256-candidate nested selector 的 label-matched advantage 不稳定。
- 固定平均两个 task-aware sidecar logits 后：broad occurrence base mean `0.755736` → `0.759757`（gain `+0.004021`）；broad relation base mean `0.758018` → `0.760526`（gain `+0.002508`）。ensemble 降低了 seed 方差，但仍未达到 +0.005 gate。

## 4. Decision

- prototype-seed robustness gate：**False**。
- **不晋级 official valid/test。** 单 seed 的 `0.762375` 仍是有价值的 mechanism signal，但不能当作冻结配置的可靠预期。
- 需要保留的结论不是“当前 supervised selector 已成功”，而是：**task-aware relation bank 可能有用，但当前 256-candidate hard selection 太依赖 reservoir identity。**

## 5. Next optimization direction

下一轮应停止直接从随机 256-candidate pool hard-select 32 个 atoms，改成更稳定的 task matching：

1. 使用 broad deterministic outer-fit bank（每个 fit graph 一个 observed patch）作为冻结候选宇宙；
2. 不替换 broad occurrence vocabulary，只学习 nested outer-fit-only prototype scalar weights / relation gates；
3. 或对多个 candidate reservoirs 做 selector-score consensus，再从 union 中选 prototype；
4. 保持 broad occurrence base、exact distance 1+2、assignment-shuffled 和 shuffled-label controls 不变。

这比继续调 residual cap、增加 random-walk 长度或扩大 relation head 更符合当前证据。
