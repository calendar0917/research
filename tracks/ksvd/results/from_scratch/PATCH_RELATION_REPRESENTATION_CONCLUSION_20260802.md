# Patch relations 到稳定图表示：两轮结论

> 日期：2026-08-02  
> 范围：72 synthetic graphs；3-fold graph isolation；无 graph labels。

## 1. 核心答案

当前 patches 可以形成有效的 relation-aware representation substrate，但必须把两个问题分开：

1. **patch relations 有真实的 held-out structural signal**；
2. **当前 ordered-adjacency KSVD token 不满足重编号稳定性**。

因此关系机制应保留，numeric-ID/construction-slot 依赖不应进入最终 local token。

## 2. KSVD token 的 relation audit

Masked target patch 不读取自己的 raw vector/code，只用其他 patches 预测其 22D permutation-invariant structure。

| branch | overall RMSE |
|---|---:|
| RAW_BAG | 0.22443 |
| KSVD_BAG | 0.18916 |
| KSVD_TRUE_RELATION | **0.11869** |
| KSVD_SHUFFLED_RELATION | 0.13980 |

TRUE relation：

- 相对 KSVD_BAG 改善 `37.25%`；
- 相对 SHUFFLED binding 改善 `15.10%`；
- 3/3 folds 同时更好。

判定：`PATCH_RELATIONS_ADD_MASKED_VALUE`。

但 node relabel + Beam8 resampling 的 graph-embedding cosine：

```text
RAW_BAG              0.9908
KSVD_BAG             0.6953
KSVD_TRUE_RELATION   0.7221
```

relation 略微缓解但没有修复 ordered KSVD token 的不稳定性。

## 3. ID-free invariant token follow-up

把 local token 替换为不读取 global ID/slot 的 22D patch descriptor：sorted degrees、sorted spectrum、density、triangle density。

| branch | overall RMSE | relabel cosine |
|---|---:|---:|
| INVARIANT_BAG | 0.16831 | 0.9988 |
| INVARIANT_TRUE_RELATION | **0.12474** | **0.9983** |
| INVARIANT_SHUFFLED_RELATION | 0.14263 | 0.9982 |

TRUE relation：

- 相对 invariant BAG 改善 `25.89%`；
- 相对 invariant SHUFFLED 改善 `12.55%`；
- 3/3 folds 同时更好；
- stability gate 通过。

判定：`ID_FREE_PATCH_RELATION_SUBSTRATE_SUPPORTED`。

## 4. 方法含义

当前证据支持：

```text
Beam8 overlapping patches
    ↓
ID-free local patch token
    +
overlap / center-distance relative relations
    ↓
relation-aware graph encoder
```

不再支持直接把 construction-order KSVD codes 当作稳定 patch tokens。

KSVD_TRUE 的 masked RMSE 比 invariant TRUE 低约 `4.85%`，说明 KSVD 仍含有额外结构信息；但它的 relabel cosine 低约 `0.276`。因此 KSVD 更适合作为待稳定化的辅助通道，而不是当前唯一主 token。

## 5. 下一模型的冻结建议

第一版 relation-aware model 应使用：

```text
primary token       ID-free invariant or learned equivariant patch encoder
auxiliary token     KSVD code（可选）
relations           overlap fraction + center shortest-path distance
aggregator          relation-aware attention / patch-graph message passing
graph readout       CLS or attention pooling
```

KSVD auxiliary channel 必须加入 relabel/permutation consistency loss，或经过 invariant projection。所有 learned variants 继续使用：

1. true vs shuffled relation gate；
2. node relabel + resampling stability；
3. held-out graph isolation；
4. 最终真实数据 downstream linear probe/fine-tuning。

## 6. 边界

- masked target 是手工 invariant structure，不是 graph classification；
- invariant descriptor 是 control，不是最终 learned encoder；
- ridge added value 证明关系信号存在，但不保证 Transformer 一定获得同等收益；
- 真实 50-node 数据仍需要确认节点身份是否跨图对齐。

## 7. KSVD 的 ID/slot 分解与 invariant-space 修正

后续 matched audit 把 global identity、patch selection 和 patch 内 local slot 分开。结果为：

| intervention | graph cosine | patch-vector match | patch-code cosine |
|---|---:|---:|---:|
| mapped global relabel（同时搬运原 local order） | 1.0000 | 1.0000 | 1.0000 |
| frozen patch set，仅打乱 local slots | 0.6359 | 0.0372 | 0.1225 |
| frozen patch set，重新按 numeric ID 排序 | 0.6446 | 0.0353 | 0.1449 |
| relabel 后重新采样 | 0.6718 | — | — |

所以问题不是 global node ID 仅作为身份记账，而是 adjacency entries 被放入没有稳定语义的 local slots，再展开成固定45D坐标。numeric-ID sort 只是另一种任意排列，不能修复。

修正方案先把 patch 转成22D permutation-invariant descriptor，再在该空间学习 KSVD。容量审计结果：

| token | TRUE RMSE | vs raw invariant TRUE | relabel cosine |
|---|---:|---:|---:|
| raw invariant TRUE | 0.12474 | baseline | 0.9983 |
| invariant KSVD K16/T3 TRUE | **0.09945** | **-20.27%** | 0.9673 |
| invariant KSVD K24/T3 TRUE | 0.13625 | +9.23% | 0.9506 |

`K16/T3` 在3/3 folds 中均优于 bag 与 shuffled relation，16/16 atoms 有效。更大字典虽然降低 patch reconstruction error，却恶化 masked downstream RMSE，说明这里的最佳 KSVD 作用是低容量结构去噪，而不是尽可能精确地复刻每个 patch descriptor。

更新后的建议是：

```text
primary compact token  invariant-space KSVD K16/T3
high-stability sidecar raw invariant token（可选）
relations              overlap fraction + center shortest-path distance
```

该选择修复了 ordered-adjacency KSVD 的主要 local-slot 缺陷；剩余 `1 - 0.9673` 的不稳定主要来自重编号后 Beam8 可能选择不同 patch sets，而不是 KSVD 再次读取 numeric ID。

## 8. 更正：完整邻接 canonical-slot 路线

第7节的 invariant-space KSVD 回答的是 compact graph representation，不是用户原本提出的“保留45D完整邻接、只改变 slot 生成规则”。后续 direct audit 已完成这一更直接的路线。

三组独立 graph/cover seeds 中，`ROOTED_CANONICAL` 均被选中：把 patch center 作为 singleton color，对完整 induced adjacency 做 exact canonical labeling，再展开45D upper triangle。

```text
construction mean observed/full RMSE  0.34842 / 0.33422
rooted canonical                     0.31248 / 0.31477
relative improvement                 10.32% / 5.82%
```

同一 frozen patch-set 重编号后，rooted canonical 的45D vector match、matched KSVD-code cosine 和 pooled-code cosine 在三组 seeds 中均为 `1.0000`。因此完整邻接 KSVD 的主要 slot instability 已经可以直接修复，不必通过22D summary 绕开。

更新后的双路线定位：

```text
adjacency reconstruction  rooted-canonical 45D + K24/T3 KSVD + stitching
compact graph token       invariant descriptor / invariant-space KSVD
```

rooted canonical 在 relabel+Beam8 resampling 后的 code cosine仍约 `0.7045`，因为 sampler 选择了不同 patch sets；但 full RMSE 平均 `0.31477 → 0.31482`，重构质量几乎不变。详细结论见 `STRUCTURAL_CANONICAL_SLOT_MULTISEED_CONCLUSION_20260802.md`。

## 9. 直接统计 vs rooted patch/KSVD 下游读出

为回答“最终仍需 pooling 时，KSVD 是否只是换一种统计”，使用三组 seeds 做9-class `family × degree` graph-level attribution。所有 branches 均经过相同 train-only standardization、12D PCA和fixed ridge。

| branch | joint-9 acc |
|---|---:|
| GLOBAL_STATS | **0.8004** |
| ROOTED_RAW_BAG | 0.4835 |
| ROOTED_RAW_TRUE_RELATION | **0.5802** |
| ROOTED_KSVD_BAG | 0.4136 |
| ROOTED_KSVD_TRUE_RELATION | 0.4527 |
| ROOTED_KSVD_SHUFFLED_RELATION | 0.4115 |

三次均判定 `DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS`。relations 对 raw/KSVD bag 分别有约 `+0.0967/+0.0391` 平均增量，但 KSVD TRUE 比 RAW TRUE 低 `0.1275`，且远低于 direct stats。

因此当前不能把 KSVD activation statistics 作为优于直接统计的下游表示。KSVD 保留为 adjacency reconstruction compressor；synthetic family/degree readout 直接使用 whole-graph statistics；若继续研究 relations，优先保留未压缩 raw canonical patch tokens。详细结果见 `ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_MULTISEED_CONCLUSION_20260802.md`。
