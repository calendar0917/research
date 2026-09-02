# Rooted-canonical KSVD 下游归因：三种子结论

> 日期：2026-08-02  
> 范围：3 graph-bank seeds × 3 cover seeds；72张50-node synthetic graphs/seed；9-class `family × degree`；相同 train-only standardization + 12D PCA + fixed ridge。

## 1. 判定

三次独立审计全部得到：

```text
DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS
```

当前 synthetic graph-level downstream 上，rooted-canonical KSVD 不优于直接 whole-graph statistics，也不优于未压缩 raw canonical patches + relations。

## 2. 三种子均值

| branch | joint-9 acc | family acc | degree acc | relabel-resample cosine |
|---|---:|---:|---:|---:|
| GLOBAL_STATS | **0.8004** | **1.0000** | **0.8374** | **1.0000** |
| ROOTED_RAW_BAG | 0.4835 | 0.6523 | 0.7901 | 0.6344 |
| ROOTED_RAW_TRUE_RELATION | **0.5802** | 0.7284 | 0.8230 | 0.6974 |
| ROOTED_KSVD_BAG | 0.4136 | 0.5967 | 0.6132 | 0.3376 |
| ROOTED_KSVD_TRUE_RELATION | 0.4527 | 0.5823 | 0.7037 | 0.4008 |
| ROOTED_KSVD_SHUFFLED_RELATION | 0.4115 | 0.5761 | 0.6975 | 0.4010 |

GLOBAL_STATS joint accuracy by seed：`0.8210 / 0.7901 / 0.7901`。  
RAW TRUE：`0.6605 / 0.4753 / 0.6049`。  
KSVD TRUE：`0.4815 / 0.4198 / 0.4568`。

## 3. 归因

### Relations 不是完全无用

三种子均值：

```text
RAW TRUE - RAW BAG       +0.0967
KSVD TRUE - KSVD BAG     +0.0391
KSVD TRUE - SHUFFLED     +0.0412
```

因此 overlap/distance binding 含有下游信号。但 KSVD TRUE 对 BAG/SHUFFLED 的 fold-level simultaneous wins 不稳定，不能升级为可靠下游增益。

### KSVD compression 丢失下游可读信号

在相同 true-relation mechanism 与相同12D最终容量下：

```text
RAW TRUE   0.5802
KSVD TRUE  0.4527
difference -0.1275
```

所以 dictionary/code 不是本任务 relation benefit 的来源；K24/T3 sparse bottleneck 对 generator-factor classification 过于有损。

### 直接统计为何胜出

当前 targets 本身是 graph generator 的宏观因素：family 与 target degree。sorted degree sequence、whole-graph spectrum、density、triangle density 与这些因素高度对齐，family 在三种子中均达到1.0。patch sampler只观察部分重复局部区域，经过bag/relations和12D读出后无法胜过直接访问完整图的宏观统计。

## 4. 方法定位更新

当前证据支持拆分：

```text
graph reconstruction
    rooted-canonical 45D → KSVD → stitching

synthetic family/degree downstream
    direct graph statistics

relation mechanism research
    raw canonical patch tokens + true relations
```

不再支持：

> 对 KSVD activation做 mean/std/max 就天然形成优于直接统计的 graph representation。

若最终 readout 只是 code histogram/statistics，必须把 whole-graph statistics 作为强 baseline；在当前任务上应直接选后者。

## 5. 边界

- synthetic generator factors刻意接近宏观统计，不能外推到真实任务；
- 当前 relation readout是手工 weighted summaries，不是attention/message passing；
- 12D PCA公平控制最终容量，但可能不利于某些非线性 patch信号；
- 该负结果不影响 rooted-canonical KSVD 的 adjacency reconstruction结论；
- 真实50-node labels到位后，必须重新做 `GLOBAL_STATS / RAW_RELATION / KSVD_RELATION` matched comparison。

## 6. 产物

- 协议：`KSVD_ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_PROTOCOL_20260802.md`；
- runner：`run_rooted_canonical_downstream_attribution.py`；
- test：`test_rooted_canonical_downstream_attribution.py`；
- 三次结果：`ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_20260802.md`、`ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_SEED2_20260802.md`、`ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_SEED3_20260802.md`。
