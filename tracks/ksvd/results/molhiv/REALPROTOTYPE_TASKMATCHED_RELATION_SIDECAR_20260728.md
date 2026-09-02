# Task-matched prototype relation sidecar（2026-07-28）

## 1. Protocol

- 8,000-graph development subset；只使用其中 6,400 个 official-train graphs 的三个 outer scaffold folds。
- task-aware selector 严格 nested：candidate 只由 outer-fit 构造，candidate source graph 不参与评价该 candidate 的 inner-valid evidence。
- `nested_taskaware` 与 label-shuffled `nested_shuffled` 使用相同 candidate bank、MMR 与模型协议。
- relation 仅使用已定位的 exact shortest-path distance 1+2 compact feature（140 维），并保留 node-assignment-shuffled control。
- broad occurrence base 是 frozen farthest + scaffold-facility probability pair；另测试 frozen broad relation pair。
- official valid/test 编码与评估均为 **0**。

## 2. Task-matched bank standalone

| Prototype selector | Occurrence base | Real d1+2 | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nested task-aware | 0.716416 | 0.725287 | 0.721694 | +0.008871 | +0.003593 | 3/3 | 3/3 |
| Nested shuffled-label | 0.716094 | 0.718380 | 0.718099 | +0.002286 | +0.000280 | 3/3 | 1/3 |

结论：task-aware selection **没有提升 standalone occurrence**（两种 selector 的 occurrence mean 几乎相同），但 task-aware bank 上的真实 d1+2 relation gain 为 `+0.008871`，明显高于 shuffled-label selector 的 `+0.002286`。这说明监督信息更可能改变“哪些 prototype 适合做关系组合”，而不是直接改善 occurrence readout。

## 3. Role-separated sidecar, residual cap 0.25

| Frozen base | Relation bank | Base | Real | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Broad occurrence pair | Nested task-aware | 0.755736 | 0.760658 | 0.756053 | +0.004922 | +0.004605 | 3/3 | 3/3 |
| Broad occurrence pair | Nested shuffled-label | 0.755736 | 0.757843 | 0.757262 | +0.002107 | +0.000582 | 3/3 | 2/3 |
| Broad relation pair | Nested task-aware | 0.758018 | 0.761753 | 0.756588 | +0.003735 | +0.005165 | 3/3 | 3/3 |
| Broad relation pair | Nested shuffled-label | 0.758018 | 0.759172 | 0.758214 | +0.001154 | +0.000958 | 3/3 | 2/3 |

cap 0.25 下，task-aware relation sidecar 在 broad occurrence base 上得到 `0.760658`，gain `+0.004922`，只差 `0.000078` 达到 +0.005 gate，但 real 同时以 3/3 folds 超过 base 和 assignment-shuffled。

## 4. One-step capacity confirmation, residual cap 0.3125

| Frozen base | Relation bank | Base | Real | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Broad occurrence pair | Nested task-aware | 0.755736 | 0.761675 | 0.755729 | +0.005939 | +0.005947 | 3/3 | 3/3 |
| Broad occurrence pair | Nested shuffled-label | 0.755736 | 0.758332 | 0.757348 | +0.002596 | +0.000984 | 3/3 | 2/3 |
| Broad relation pair | Nested task-aware | 0.758018 | 0.762375 | 0.755966 | +0.004357 | +0.006409 | 3/3 | 3/3 |
| Broad relation pair | Nested shuffled-label | 0.758018 | 0.759346 | 0.757899 | +0.001328 | +0.001447 | 3/3 | 2/3 |

- broad occurrence pair + task-aware relation sidecar：fold AUC `[0.7738151707639004, 0.7030345900827616, 0.8081766617806385]`，mean `0.761675`。
- 相对 broad occurrence pair gain `+0.005939`，real−shuffled `+0.005947`，两项均 3/3 wins。
- 最佳绝对结果是 broad relation pair + task-aware sidecar：`0.762375`；相对上一轮 broad relation pair `+0.004357`。
- shuffled-label selector 的增益明显更小，说明结果不是任意第三套 prototype bank 或额外 140 参数即可解释。

严格 development gate（以 broad occurrence pair + task-aware sidecar 为 target）：**True**。

注意：cap `0.3125` 是在 cap `0.25` near-gate 后进行的一步 development optimization，因此它不是 untouched confirmation。进入 official valid/test 前仍应做 prototype-bank seed / selector robustness；不能仅凭本轮调参后过线直接使用 official splits 做选择。

## 5. Research decision

1. **B 的更精确版本成立**：字典不应整体变成 supervised vocabulary；更合理的是 role separation：label-free broad banks 负责稳定 occurrence，nested task-matched bank 只负责 local relation composition。
2. 这也解释了前一轮矛盾：task-aware bank standalone occurrence 较弱，但其 d1+2 relation feature 在强 broad base 上有稳定、assignment-specific 的增益。
3. 当前最佳开发模型仍是 GNN-free：broad occurrence/relation + 140-d task-matched relation sidecar；不需要改回 GINE backbone。
4. 下一步不是继续扩 walk 长度或 relation matrix，而是验证 selector 稳定性：至少 3 个 candidate-bank/prototype seeds，并固定 cap 0.3125。只有 robustness 仍通过 gate，才晋级 full official valid/test。
