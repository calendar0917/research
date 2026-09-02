# Structural canonical slots：完整邻接 KSVD 审计

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_STRUCTURAL_CANONICAL_SLOT_PROTOCOL_20260802.md`  
> 判定：`ADOPT_STRUCTURAL_CANONICAL_SLOTS`

## 1. Frozen patch-set relabel stability

| branch | vector match | code cosine | support Jaccard | pooled cosine | transition match | node-order equivariance |
|---|---:|---:|---:|---:|---:|---:|
| CONSTRUCTION | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ID_SORT | 0.0304 | 0.1134 | 0.1301 | 0.6370 | 0.0009 | 0.0000 |
| SIGNATURE | 0.9136 | 0.9495 | 0.9361 | 0.9755 | 0.6510 | 0.6461 |
| CANONICAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8549 | 0.9153 |
| ROOTED_CANONICAL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8988 | 0.9435 |
| OVERLAP_CANONICAL | 0.9970 | 0.9978 | 0.9976 | 0.9999 | 0.9947 | 0.9904 |

## 2. Held-out KSVD + stitched reconstruction

| branch | patch error | observed RMSE | observed F1 | full RMSE | relabel-resample full RMSE | abs delta | full recall | full F1 | resample cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CONSTRUCTION | 0.4312 | 0.3483 | 0.8796 | 0.3356 | 0.3339 | 0.0017 | 0.8167 | 0.8236 | 0.6849 |
| ID_SORT | 0.4543 | 0.3711 | 0.8547 | 0.3480 | 0.3562 | 0.0082 | 0.7985 | 0.8011 | 0.6299 |
| SIGNATURE | 0.3920 | 0.3155 | 0.9010 | 0.3180 | 0.3175 | 0.0005 | 0.8184 | 0.8426 | 0.6798 |
| CANONICAL | 0.3941 | 0.3187 | 0.9032 | 0.3195 | 0.3181 | 0.0014 | 0.8352 | 0.8458 | 0.6743 |
| ROOTED_CANONICAL | 0.3880 | 0.3130 | 0.9033 | 0.3166 | 0.3159 | 0.0008 | 0.8244 | 0.8449 | 0.6980 |
| OVERLAP_CANONICAL | 0.4135 | 0.3342 | 0.8896 | 0.3279 | 0.3280 | 0.0001 | 0.8235 | 0.8329 | 0.6996 |

## 3. Ordering diagnostics

| branch | signature tie rate | canonical ambiguity | mean/max search leaves | decision |
|---|---:|---:|---:|---|
| CONSTRUCTION | 0.0000 | 0.0000 | 0.00/0 | CONTROL |
| ID_SORT | 0.0000 | 0.0000 | 0.00/0 | CONTROL |
| SIGNATURE | 0.4863 | 0.0000 | 0.00/0 | CONTROL |
| CANONICAL | 0.0000 | 0.5357 | 1.27/12 | STRUCTURAL_CANONICAL_SLOT_SUPPORTED |
| ROOTED_CANONICAL | 0.0000 | 0.4768 | 1.15/8 | STRUCTURAL_CANONICAL_SLOT_SUPPORTED |
| OVERLAP_CANONICAL | 0.0000 | 0.3543 | 1.02/2 | CANONICAL_SLOT_STABILITY_GATE_FAIL |

Passing branches：`['CANONICAL', 'ROOTED_CANONICAL']`。  
Selected branch：`ROOTED_CANONICAL`。

Exact canonical vector 的稳定性与 canonical node-map 的唯一性是两件事。若 vector/code 为1而 node-order/transition低于1，差异来自 patch automorphism 中结构等价节点无法由纯结构唯一命名。
