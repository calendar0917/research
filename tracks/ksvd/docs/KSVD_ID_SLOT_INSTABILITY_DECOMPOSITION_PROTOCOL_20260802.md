# KSVD token 不稳定性：sampler ID 与 local slot 分解协议

> 日期：2026-08-02  
> 状态：结果前冻结。

## 1. 问题

当前 KSVD patch token 来自 construction-ordered `10×10` adjacency 的 45D upper vector。node relabel + resampling 后 graph-code cosine 约为 `0.70`。可能来源包括：

1. global node IDs 改变 sampler tie-breaking，使 patch node sets 改变；
2. 同一 patch node set 的 local slots 改变，使45D坐标置换；
3. patch sequence 改变；
4. sparse support 对上述扰动放大。

本轮在同一冻结 dictionary 下分离这些来源。

## 2. 设置

- 沿用72图 Beam8/R1、`s10/o3/m1.5`；
- graph bank `810001`，cover seed `940101`；
- 3-fold graph isolation；KSVD `K24/T3/u25`；
- graph embedding 为 patch codes 的 `mean/std/max` pooling。

## 3. 对照

### MAPPED_GLOBAL_RELABEL

随机重编号全图，但把原 cover node order、patch order 和 transition substrate 一起映射过去。它只改变 numeric IDs，不改变 local adjacency coordinates。

### FROZEN_SET_SLOT_SHUFFLE

保持每个 patch 的 node set、patch sequence 和 global graph 完全不变，只对每个 patch 的10个 local slots 做独立随机 permutation。

### FROZEN_SET_ID_SORT

保持 patch node sets 不变，把每个 patch 直接按 numeric global ID 排序。随后对重编号图的映射 node sets 再按新 ID 排序，模拟把 ID 当 local coordinate 的方案。

### PATCH_SEQUENCE_SHUFFLE

只改变 patch rows 的顺序；bag pooling 理论上应完全不变。

### RELABEL_RESAMPLE

随机重编号后用相同 scalar sampler seed 重新运行 Beam8，包含 sampler tie-breaking、patch-set 和 slot 的联合变化。

## 4. 指标

- graph-code cosine 与 relative L2；
- matched-patch signed code cosine；
- sparse support Jaccard；
- patch-vector exact match；
- sequence-shuffle graph embedding replay。

## 5. 分类

- slot shuffle cosine `<0.90`、relabel-resample `>=0.90`：`LOCAL_SLOT_INSTABILITY`；
- slot shuffle cosine `>=0.90`、relabel-resample `<0.90`：`SAMPLER_SELECTION_INSTABILITY`；
- 两者都 `<0.90`：`MIXED_SAMPLER_AND_SLOT_INSTABILITY`，并单独报告 local slot 已被 matched control 确认；
- 两者都 `>=0.90`：`KSVD_TOKEN_STABILITY_SUPPORTED`。

Mapped global relabel 必须达到 `>=0.999`，sequence shuffle 必须 replay 到 `>=0.999999`，否则优先判为实现不变量失败。ID-sort 单独作为反例，不参与主分类。

## 6. 跟进规则

若 local slot 是主因，优先尝试 permutation-ensemble/orbit-pooled KSVD token；若 sampler 是主因，优先修复 equivariant tie-breaking 或做 multi-cover consistency。未经本轮分解不直接修改 Beam8 objective。
