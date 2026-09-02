# Patch-chain arbitrariness、优化与 reconstruction error budget：完整结论

> 日期：2026-08-02  
> 范围：72 个 50-node synthetic graphs；无 graph labels；覆盖、压缩、拼接和 completion 分开审计。

## 1. 总结论

用户对“当前方法多少有点任意”的判断成立。系统审计后得到四个明确结论：

1. 旧 cover `s10/o5/m1.5` 不是 Pareto 最优，50% overlap 偏高；
2. `K24/T3` 是合理的 compression Pareto 点，不能无代价地通过调大 K/T 降低误差；
3. target-edge heuristic 离 marginal coverage 目标有明显差距；
4. 一个轻量 `Beam8_R1` candidate search 稳定修复了主要 cover 瓶颈，并在 RAW、KSVD 和 end-to-end completion 中保留收益。

因此推荐把 continuous-cover 主方案从 v1 更新为：

> **target-seeded marginal candidate beam + s=10/o=3/m=1.5 + construction ordering + explicit transition maps + optional K24/T3 KSVD + optional train-only structural completion**。

## 2. Cover 参数任意性

预注册 27 cells：

```text
s = 8 / 10 / 12
overlap ≈ 1/3 / 1/2 / 3/4
budget multiplier = 1.0 / 1.5 / 2.0
```

当前 `s10/o5/m1.5`：

```text
mean patches       17.65
raw pair slots     794.38
edge coverage      0.6870
pair coverage      0.4925
RAW full RMSE      0.3530
```

它被两个 cells 支配：

| cell | patches | raw pair slots | edge cover | pair cover | RAW full RMSE |
|---|---:|---:|---:|---:|---:|
| old `s10/o5/m1.5` | 17.65 | 794.38 | 0.6870 | 0.4925 | 0.3530 |
| `s10/o3/m1.5` | 17.65 | 794.38 | 0.7629 | 0.5599 | 0.3064 |
| `s12/o4/m1.5` | 12.00 | 792.00 | 0.7424 | 0.5492 | 0.3177 |

结论：固定一半 overlap 没有被数据支持。约三分之一 overlap 用相同观察成本探索更多新区域。

## 3. KSVD capacity 是否任意

冻结旧 cover，比较 `K=16/24/32`、`T=2/3/4`、25 updates。

当前 `K24/T3`：

```text
dictionary scalars      1080
code scalars / graph    52.96
observed RMSE            0.36273
full RMSE                0.43582
```

没有任何 cell 能在 dictionary 不更大、code 不更多的同时得到更低 RMSE。所有 K/T 主网格点形成不同的容量–误差 tradeoff：

- `K16/T4` 用更小字典、更长 code 得到近似当前误差；
- `K24/T4` observed RMSE 降到 `0.3458`，但 code 增加 33%；
- `K32/T4` 降到 `0.3368`，同时 dictionary 和 code 都更大。

25→50 updates 只把 observed RMSE 从 `0.36273` 降到 `0.36248`，约 `0.07%`。训练轮数不是主要瓶颈。

判定：`CURRENT_KSVD_CAPACITY_PARETO_SUPPORTED`。

## 4. Target heuristic 的局部最优差距

18-node 小图上，每一步枚举所有 exact-overlap、connected next patches：

| branch | edge cover | pair cover |
|---|---:|---:|
| target-edge heuristic | 0.7358 | 0.4420 |
| exhaustive one-step marginal | 0.8737 | 0.4458 |

edge coverage gap 为 `+0.1379`，24/24 graphs exhaustive branch 都更好；pair coverage基本相同。

判定：`HEURISTIC_HAS_MATERIAL_ORACLE_GAP`。

这证明 target-edge 的方向虽然比 frontier 好，但“选一条目标边再走过去”仍不是高效利用下一张 patch 容量的方式。

## 5. Scalable marginal candidate beam

Beam 方法：

1. 第一 patch 保留 target-edge seed；
2. 枚举 previous patch 的 connected retained subsets；
3. 根据通向未覆盖边的 boundary potential 保留少量候选；
4. 每个 retained candidate 沿 graph frontier greedy fill；
5. 对完整 next patches 按 `new edges → new pairs → new nodes → induced edges` 选择。

### Beam sensitivity

固定 `s10/o3/m1.5`：

| cell | sec/graph | edge cover | pair cover | RAW full RMSE |
|---|---:|---:|---:|---:|
| TARGET | 0.025 | 0.7650 | 0.5605 | 0.3054 |
| B8_R1 | 0.279 | 0.8734 | 0.5288 | 0.2205 |
| B16_R1 | 0.505 | 0.8803 | 0.5298 | 0.2139 |
| B32_R1 | 0.954 | 0.8844 | 0.5307 | 0.2102 |
| B32_R2 | 1.849 | 0.8845 | 0.5309 | 0.2101 |

预注册 knee rule 选择 `B8_R1`：相对 TARGET edge coverage `+0.1083`、RAW full RMSE reduction `27.81%`，但比 B32_R2 快约 6.6 倍。

代价是 pair coverage下降约 `0.032`。Beam8 更偏向观察真实边，而不是广泛观察非边 pairs。

### Three-seed robustness

三个 cover seeds：

```text
edge coverage gain       +0.1083 / +0.1085 / +0.1092
RAW full RMSE reduction   27.81% / 28.03% / 28.17%
```

3/3 seeds 全通过，所有 exact overlap、connected、single-chain invariants 通过。

判定：`PASS_BEAM8_THREE_SEED_ROBUSTNESS`。

## 6. Beam8 的 KSVD compression

相同 `s=10,K24/T3/u25`、相同 patch count/code cost：

| branch | patch error | observed RMSE | full RMSE | observed F1 | full recall/F1 | disagreement |
|---|---:|---:|---:|---:|---:|---:|
| old target o5 | 0.4928 | 0.3653 | 0.4367 | 0.8387 | 0.5909/0.6854 | 0.1409 |
| target o3 | 0.5021 | 0.3760 | 0.4162 | 0.8211 | 0.6442/0.7127 | 0.1470 |
| Beam8 o3 | **0.4319** | **0.3489** | **0.3349** | **0.8780** | **0.8137/0.8224** | **0.1247** |

Beam8 相对 target o3：

- observed RMSE 改善 `7.20%`；
- full RMSE 改善 `19.52%`；
- full recall 从 `0.6442` 提高到 `0.8137`；
- patch error、F1 和 disagreement 也同时改善；
- FINAL 优于自己的 PCA3；
- 3/3 folds FINAL patch error < INIT。

判定：`ADOPT_MARGINAL_BEAM_COVER_V2`。

KSVD INIT→FINAL patch reduction仍约 5–6%，没有达到历史 10% 强 gate。因此 KSVD 继续定位为可用 compressor，而不是主要方法贡献。

## 7. Stitching 与 completion

### Slot reliability weighting

旧 target cover：observed RMSE 只改善 `0.55%`；Beam8 上只改善 `0.03%`。两者都未过 1% gate，且没有稳定 F1收益。

结论：删除 slot weighting，不进入 v2 默认方案。

### Train-only structural completion

旧 target RAW：

```text
ZERO full RMSE          0.3544
TRAIN_DENSITY           0.3079
STRUCTURAL_RIDGE        0.2900
SHUFFLED_STRUCTURE      0.2951
```

structural ridge 相对 density 改善 `5.81%`，相对 shuffled 改善 `1.74%`，registered completion gate 通过。

Beam8：

```text
RAW zero                0.2200
RAW structural          0.2031
KSVD zero               0.3349
KSVD structural         0.3241
```

最终 Beam8 + K24/T3 + structural completion：

```text
full RMSE       0.3241
full recall     0.8138
full F1         0.8224
```

相对旧 target-chain + KSVD + completion 的 `0.3845`，full RMSE 改善约 `15.7%`。

completion 是可选 graph-completion 后处理，不是 KSVD compressor 的内生能力；它使用 train graphs 的 unseen adjacency targets，但不使用 graph labels 或 test truth。

## 8. Error budget：旧方案 vs v2

| pipeline | full RMSE |
|---|---:|
| old target RAW zero | 0.3544 |
| old target KSVD zero | 0.4358 |
| old target KSVD + completion | 0.3845 |
| Beam8 RAW zero | 0.2200 |
| Beam8 RAW + completion | 0.2031 |
| Beam8 KSVD zero | 0.3349 |
| Beam8 KSVD + completion | 0.3241 |

Beam8 下：

```text
compression RMSE increment     +0.1150
slot weighting change          -0.00005
completion change              -0.0108
```

这说明 v2 的主要进步来自 cover/sampler，而不是更复杂 decoder：

> 先选择信息密度更高的连续 patches，比在旧 cover 上继续修补 stitching 更有效。

## 9. 推荐的 v2 方法

```text
patch size                  10
overlap                      3
budget multiplier            1.5
first patch                  target-edge seed
next-patch search            retained Beam8, one greedy fill each
candidate objective          new edges → new pairs → new nodes → induced edges
ordering                     construction ordering
transition representation    explicit slot-to-slot maps
compression                  optional K=24,T=3,updates=25
stitching                    uniform mean
completion                   optional train-only structural ridge
```

不保留：

- 50% overlap；
- Beam32/restarts2 默认配置；
- slot-persistent ordering；
- Fiedler total ordering；
- linear transition decoder；
- slot reliability weighting。

## 10. 尚未解决的边界

1. Beam8 计算约为 target sampler 的 11 倍，虽然当前绝对值约 0.28 sec/graph；
2. pair coverage下降约 3.2 个百分点，它更偏 edge reconstruction；
3. sampler 读取完整输入图的真实边，适用于 graph compression/representation，不是缺边条件下的 blind link prediction；
4. synthetic graph 结果仍需在导师真实 50-node 数据上重跑；
5. 若真实图之间有共享 node identity，必须加入 whole aligned-adjacency baseline；
6. 本轮不包含下游分类。
7. 后续 identity/rate 审计发现，重新编号后重新运行 Beam8 并不会复现同一 chain/slot substrate；把 global node identity sidecar 计入后，当前 RAW 显式 payload 平均也大于 1225-bit upper-triangle adjacency。

因此 v2 可以作为当前 synthetic continuous-cover sampler 的最终方案，但不能直接升格为编号无关图表示或已经证明 bit-competitive 的压缩格式。完整结果见 `PATCH_CHAIN_IDENTITY_RATE_AUDIT_20260802.md`；整个项目的最终方案仍取决于真实数据、下游目标以及 labeled compression / ID-free representation 的路线选择。
