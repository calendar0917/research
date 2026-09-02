# Attributed Beam8/Mutagenicity anchor invariance audit

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_ANCHOR_INVARIANCE_PROTOCOL_20260813.md`  
> 判定：`ATTRIBUTED_ANCHOR_INVARIANCE_PASS`

| family | invariant | match rate | gate |
|---|---|---:|---:|
| base | chain_exact_match | 0.595703 | diagnostic |
| base | token_row_match | 1.000000 | required |
| base | token_multiset_match | 1.000000 | required |
| base | readout_match | 1.000000 | required |
| anchor | chain_exact_match | 0.565104 | diagnostic |
| anchor | token_row_match | 1.000000 | required |
| anchor | token_multiset_match | 1.000000 | required |
| anchor | readout_match | 1.000000 | required |

## Boundary

- concrete node IDs may differ inside attributed automorphisms; tokens and compact readout may not.
- anchor gate fails时，classification 只保留为实现诊断，不作为方法证据。
