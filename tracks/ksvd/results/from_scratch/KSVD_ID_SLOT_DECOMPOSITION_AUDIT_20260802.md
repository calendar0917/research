# KSVD ID/slot 不稳定性分解审计

> 日期：2026-08-02  
> 协议：`tracks/ksvd/docs/KSVD_ID_SLOT_INSTABILITY_DECOMPOSITION_PROTOCOL_20260802.md`  
> 判定：`MIXED_SAMPLER_AND_SLOT_INSTABILITY`

## 1. Matched decomposition

| condition | graph cosine | relative L2 | vector row match | patch-code cosine | support Jaccard |
|---|---:|---:|---:|---:|---:|
| MAPPED_GLOBAL_RELABEL | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| FROZEN_SET_SLOT_SHUFFLE | 0.6359 | 0.8435 | 0.0372 | 0.1225 | 0.1253 |
| FROZEN_SET_ID_SORT | 0.6446 | 0.8450 | 0.0353 | 0.1449 | 0.1390 |
| PATCH_SEQUENCE_SHUFFLE | 1.0000 | 0.0000 | - | - | - |
| RELABEL_RESAMPLE | 0.6718 | 0.8084 | - | - | - |

## 2. 结论

- mapped/sequence replay gate：`True`；
- local slot instability confirmed：`True`；
- relabel-resample instability confirmed：`True`。

MAPPED_GLOBAL_RELABEL 只改变 numeric IDs；若它保持 1.0，说明 ID 作为节点身份记账本身不会改变 KSVD。FROZEN_SET_SLOT_SHUFFLE 在完全相同 node sets 上直接测量45D local coordinates 的敏感性。RELABEL_RESAMPLE 再加入 sampler tie-breaking 与 patch-set 变化。
