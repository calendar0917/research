# 连续重叠 patch cover 第一轮审计

> 日期：2026-08-01  
> 本轮不训练 KSVD、不使用 labels。

## 1. 判定

**FAIL_SMALL_PATCH_GRAPH_RECOVERY**

## 2. 全局均值

| method | node cover | edge cover | pair cover | full adj acc | consecutive overlap | Jaccard gap | center dist | fresh relabel set sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_walk | 0.9743 | 0.5879 | 0.4652 | 0.8353 | 1.9858 | -0.0015 | 1.6098 | 0.1182 |
| sliding_walk | 1.0000 | 0.5176 | 0.4317 | 0.8068 | 5.0000 | 0.2504 | 1.6077 | 0.1182 |
| frontier_cover | 0.9999 | 0.5977 | 0.4289 | 0.8382 | 5.0000 | 0.1363 | 1.5731 | 0.1780 |

所有方法在 observed pairs 上必须满足 consistency/accuracy=1，并且 mapped replay 的 patch adjacency 与 transition slot maps 必须完全一致。

## 3. Seed gates

| seed | frontier edge/pair gain | frontier overlap/gap | pass | sliding edge/pair gain | sliding overlap/gap | pass |
|---:|---:|---:|---:|---:|---:|---:|
| 810101 | 0.0109/-0.0337 | 5.0000/0.1387 | False | -0.0716/-0.0332 | 5.0000/0.2498 | False |
| 810102 | 0.0107/-0.0361 | 5.0000/0.1371 | False | -0.0697/-0.0344 | 5.0000/0.2504 | False |
| 810103 | 0.0076/-0.0393 | 5.0000/0.1331 | False | -0.0694/-0.0329 | 5.0000/0.2512 | False |

## 4. Family / degree breakdown

| family-degree | method | edge cover | pair cover | consecutive overlap | Jaccard gap |
|---|---|---:|---:|---:|---:|
| regular_d15 | independent_walk | 0.5030 | 0.3983 | 2.1538 | 0.0095 |
| regular_d15 | sliding_walk | 0.4427 | 0.3719 | 5.0000 | 0.2649 |
| regular_d15 | frontier_cover | 0.5422 | 0.3682 | 5.0000 | 0.1512 |
| regular_d20 | independent_walk | 0.5673 | 0.4896 | 1.9534 | -0.0008 |
| regular_d20 | sliding_walk | 0.4969 | 0.4467 | 5.0000 | 0.2481 |
| regular_d20 | frontier_cover | 0.5987 | 0.4512 | 5.0000 | 0.1629 |
| regular_d25 | independent_walk | 0.5963 | 0.5416 | 2.0229 | 0.0010 |
| regular_d25 | sliding_walk | 0.5293 | 0.4932 | 5.0000 | 0.2412 |
| regular_d25 | frontier_cover | 0.6315 | 0.5151 | 5.0000 | 0.1761 |
| small_world_d15 | independent_walk | 0.5318 | 0.4013 | 1.9487 | -0.0031 |
| small_world_d15 | sliding_walk | 0.4544 | 0.3716 | 5.0000 | 0.2655 |
| small_world_d15 | frontier_cover | 0.5744 | 0.3693 | 5.0000 | 0.1540 |
| small_world_d20 | independent_walk | 0.5817 | 0.4840 | 1.9191 | -0.0044 |
| small_world_d20 | sliding_walk | 0.5053 | 0.4400 | 5.0000 | 0.2452 |
| small_world_d20 | frontier_cover | 0.6343 | 0.4631 | 5.0000 | 0.1836 |
| small_world_d25 | independent_walk | 0.5961 | 0.5369 | 2.0250 | 0.0007 |
| small_world_d25 | sliding_walk | 0.5307 | 0.4957 | 5.0000 | 0.2414 |
| small_world_d25 | frontier_cover | 0.6477 | 0.5230 | 5.0000 | 0.1865 |
| block_d15 | independent_walk | 0.5646 | 0.3802 | 2.0475 | -0.0020 |
| block_d15 | sliding_walk | 0.4942 | 0.3655 | 5.0000 | 0.2677 |
| block_d15 | frontier_cover | 0.5613 | 0.3387 | 5.0000 | 0.0724 |
| block_d20 | independent_walk | 0.6425 | 0.4506 | 1.9073 | -0.0073 |
| block_d20 | sliding_walk | 0.5628 | 0.4274 | 5.0000 | 0.2466 |
| block_d20 | frontier_cover | 0.5922 | 0.3848 | 5.0000 | 0.0628 |
| block_d25 | independent_walk | 0.7079 | 0.5045 | 1.8945 | -0.0066 |
| block_d25 | sliding_walk | 0.6425 | 0.4734 | 5.0000 | 0.2334 |
| block_d25 | frontier_cover | 0.5965 | 0.4464 | 5.0000 | 0.0773 |

## 5. 机制解释与路线判断

1. **连续性实现成功。** sliding/frontier 的相邻 overlap 均严格为 5，Jaccard gap 为正；observed-pair consistency、mapped adjacency 和 transition slot maps 全部为 1。
2. **当前 frontier 没有通过恢复 gate。** 它相对 independent walk 的 edge coverage 只提高约 0.008–0.011，低于冻结的 0.03；pair coverage 损失约 0.034–0.039，虽在容许范围内，但没有换来足够的边恢复收益。
3. **失败具有明确的 graph-family 条件。** frontier 在 regular 与 small-world 的各 degree cell 中提高 edge coverage，但在 block 图上随 degree 增大而明显恶化。单条局部 frontier 会在社区内部持续获得高局部收益，因此无法把有限 patch 预算合理转移到另一个社区。
4. **sliding walk 不是候选。** 它提供最清晰的序列连续性，但在所有 family/degree cells 都降低 edge coverage；仅靠一条 walk 的连续窗口不足以形成高质量 edge cover。
5. **暂不进入 KSVD。** 当前 raw induced patches 尚只观察约 0.60 的真实边；此时比较 KSVD stitched reconstruction 会把采样缺失与字典误差混在一起。

下一轮应保持 patch size、overlap 和预算不变，只修改 frontier 的跨区域调度：显式选择低覆盖区域中的 uncovered target edge，并用最短 bridge 把下一 patch 引向该区域。必须同时比较 single-chain、edge-targeted bridge 和允许多个连续 segment 的 multi-chain cover；若仍不能提高 block edge coverage，再做 patch-size/overlap 容量曲线，而不是直接接 KSVD。

## 6. 边界

- `mapped replay=1` 只说明给定同一抽象 patch cover 后，节点重编号不会改变局部 adjacency 或 overlap correspondence。
- fresh resampling similarity 不是严格不变性；它会受到随机选择和结构 ties 影响。
- full adjacency accuracy 会受图的非边占多数影响，主恢复指标仍是 true-edge 与 node-pair coverage。
- 本轮通过也不等于 KSVD 或 Transformer 已经有效；下一轮才比较 raw patch 与 KSVD patch 的 stitched reconstruction。
