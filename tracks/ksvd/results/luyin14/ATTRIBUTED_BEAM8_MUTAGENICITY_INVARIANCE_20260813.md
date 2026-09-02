# Attributed Beam8/Mutagenicity relabel-invariance audit

> 协议：`tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_INVARIANCE_PROTOCOL_20260813.md`  
> 判定：`ATTRIBUTED_BEAM8_INVARIANCE_PASS`

| invariant | match rate | gate |
|---|---:|---:|
| chain_exact_match | 0.589844 | diagnostic |
| token_row_match | 1.000000 | required |
| token_multiset_match | 1.000000 | required |
| readout_match | 1.000000 | required |

## Boundary

- exact chain node IDs may differ only inside attributed automorphisms; token and readout must remain exact.
- classification is permitted only when all three representation gates equal 1.0.
