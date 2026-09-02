# Structural canonical slots：完整邻接 KSVD 审计

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_STRUCTURAL_CANONICAL_SLOT_PROTOCOL_20260802.md`  
> 判定：`ADOPT_STRUCTURAL_CANONICAL_SLOTS`

## 1. Frozen patch-set relabel stability

| branch | vector match | code cosine | support Jaccard | pooled cosine | transition match | node-order equivariance |
|---|---:|---:|---:|---:|---:|---:|
| CONSTRUCTION | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ID_SORT | 0.0367 | 0.1285 | 0.1345 | 0.6576 | 0.0000 | 0.0000 |
| SIGNATURE | 0.9209 | 0.9564 | 0.9422 | 0.9831 | 0.6104 | 0.6308 |
| CANONICAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8555 | 0.9128 |
| ROOTED_CANONICAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9093 | 0.9483 |
| OVERLAP_CANONICAL | 0.9979 | 0.9986 | 0.9982 | 0.9996 | 0.9934 | 0.9912 |

## 2. Held-out KSVD + stitched reconstruction

| branch | patch error | observed RMSE | observed F1 | full RMSE | relabel-resample full RMSE | abs delta | full recall | full F1 | resample cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CONSTRUCTION | 0.4305 | 0.3476 | 0.8803 | 0.3337 | 0.3341 | 0.0005 | 0.8182 | 0.8250 | 0.6994 |
| ID_SORT | 0.4512 | 0.3680 | 0.8567 | 0.3448 | 0.3559 | 0.0111 | 0.8033 | 0.8038 | 0.6655 |
| SIGNATURE | 0.3917 | 0.3162 | 0.8999 | 0.3167 | 0.3166 | 0.0000 | 0.8203 | 0.8423 | 0.6994 |
| CANONICAL | 0.3893 | 0.3155 | 0.9059 | 0.3162 | 0.3176 | 0.0014 | 0.8388 | 0.8490 | 0.6736 |
| ROOTED_CANONICAL | 0.3882 | 0.3132 | 0.9030 | 0.3151 | 0.3149 | 0.0002 | 0.8270 | 0.8455 | 0.7001 |
| OVERLAP_CANONICAL | 0.4151 | 0.3358 | 0.8890 | 0.3271 | 0.3277 | 0.0006 | 0.8252 | 0.8331 | 0.6857 |

## 3. Ordering diagnostics

| branch | signature tie rate | canonical ambiguity | mean/max search leaves | decision |
|---|---:|---:|---:|---|
| CONSTRUCTION | 0.0000 | 0.0000 | 0.00/0 | CONTROL |
| ID_SORT | 0.0000 | 0.0000 | 0.00/0 | CONTROL |
| SIGNATURE | 0.5090 | 0.0000 | 0.00/0 | CONTROL |
| CANONICAL | 0.0000 | 0.5468 | 1.27/24 | STRUCTURAL_CANONICAL_SLOT_SUPPORTED |
| ROOTED_CANONICAL | 0.0000 | 0.4972 | 1.13/6 | STRUCTURAL_CANONICAL_SLOT_SUPPORTED |
| OVERLAP_CANONICAL | 0.0000 | 0.3501 | 1.02/6 | CANONICAL_SLOT_STABILITY_GATE_FAIL |

Passing branches：`['CANONICAL', 'ROOTED_CANONICAL']`。  
Selected branch：`ROOTED_CANONICAL`。

Exact canonical vector 的稳定性与 canonical node-map 的唯一性是两件事。若 vector/code 为1而 node-order/transition低于1，差异来自 patch automorphism 中结构等价节点无法由纯结构唯一命名。
