# ENZYMES Beam8 relabel-invariance audit

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_INVARIANCE_PROTOCOL_20260814.md`  
> 判定：`ENZYMES_BEAM8_INVARIANCE_FAILED`

| invariant | match rate | role |
|---|---:|---:|
| chain_exact_match | 0.697917 | diagnostic |
| token_row_match | 0.697917 | required |
| token_multiset_match | 0.713542 | required |
| readout_match | 0.713542 | required |
| incidence_equivariance | 0.713542 | required |

## Boundary

- exact chain node IDs may differ only inside discrete attributed automorphisms.
- token/readout invariance and mapped node-incidence equivariance are required.
- Passing confirms that continuous patch content did not break relabel legality.
