# Structural canonical slots：完整邻接 KSVD 审计

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_STRUCTURAL_CANONICAL_SLOT_PROTOCOL_20260802.md`  
> 判定：`ADOPT_STRUCTURAL_CANONICAL_SLOTS`

## 1. Frozen patch-set relabel stability

| branch | vector match | code cosine | support Jaccard | pooled cosine | transition match | node-order equivariance |
|---|---:|---:|---:|---:|---:|---:|
| CONSTRUCTION | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ID_SORT | 0.0262 | 0.1201 | 0.1200 | 0.6404 | 0.0006 | 0.0000 |
| SIGNATURE | 0.9060 | 0.9451 | 0.9324 | 0.9780 | 0.6183 | 0.6372 |
| CANONICAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8329 | 0.8995 |
| ROOTED_CANONICAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8786 | 0.9319 |
| OVERLAP_CANONICAL | 0.9932 | 0.9954 | 0.9946 | 0.9981 | 0.9900 | 0.9808 |

## 2. Held-out KSVD + stitched reconstruction

| branch | patch error | observed RMSE | observed F1 | full RMSE | relabel-resample full RMSE | abs delta | full recall | full F1 | resample cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CONSTRUCTION | 0.4301 | 0.3493 | 0.8787 | 0.3334 | 0.3332 | 0.0002 | 0.8191 | 0.8249 | 0.6894 |
| ID_SORT | 0.4533 | 0.3707 | 0.8554 | 0.3450 | 0.3559 | 0.0110 | 0.8052 | 0.8040 | 0.6317 |
| SIGNATURE | 0.3932 | 0.3181 | 0.9007 | 0.3162 | 0.3161 | 0.0001 | 0.8223 | 0.8446 | 0.6881 |
| CANONICAL | 0.3898 | 0.3162 | 0.9048 | 0.3151 | 0.3155 | 0.0004 | 0.8421 | 0.8497 | 0.7043 |
| ROOTED_CANONICAL | 0.3856 | 0.3113 | 0.9046 | 0.3126 | 0.3137 | 0.0011 | 0.8301 | 0.8484 | 0.7153 |
| OVERLAP_CANONICAL | 0.4133 | 0.3351 | 0.8900 | 0.3254 | 0.3263 | 0.0009 | 0.8298 | 0.8356 | 0.6890 |

## 3. Ordering diagnostics

| branch | signature tie rate | canonical ambiguity | mean/max search leaves | decision |
|---|---:|---:|---:|---|
| CONSTRUCTION | 0.0000 | 0.0000 | 0.00/0 | CONTROL |
| ID_SORT | 0.0000 | 0.0000 | 0.00/0 | CONTROL |
| SIGNATURE | 0.5035 | 0.0000 | 0.00/0 | CONTROL |
| CANONICAL | 0.0000 | 0.5419 | 1.33/24 | STRUCTURAL_CANONICAL_SLOT_SUPPORTED |
| ROOTED_CANONICAL | 0.0000 | 0.4894 | 1.20/24 | STRUCTURAL_CANONICAL_SLOT_SUPPORTED |
| OVERLAP_CANONICAL | 0.0000 | 0.3469 | 1.03/8 | CANONICAL_SLOT_STABILITY_GATE_FAIL |

Passing branches：`['CANONICAL', 'ROOTED_CANONICAL']`。  
Selected branch：`ROOTED_CANONICAL`。

Exact canonical vector 的稳定性与 canonical node-map 的唯一性是两件事。若 vector/code 为1而 node-order/transition低于1，差异来自 patch automorphism 中结构等价节点无法由纯结构唯一命名。
