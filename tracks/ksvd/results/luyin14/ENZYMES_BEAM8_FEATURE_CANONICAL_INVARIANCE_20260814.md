# ENZYMES full-feature-canonical Beam8 relabel-invariance audit

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_FEATURE_CANONICAL_INVARIANCE_PROTOCOL_20260814.md`  
> 判定：`ENZYMES_FEATURE_CANONICAL_CONTINUOUS_CONTENT_INVARIANCE_PASS`

| invariant | match rate | role |
|---|---:|---:|
| chain_exact_match | 0.979167 | diagnostic |
| token_row_match | 1.000000 | required |
| token_multiset_match | 1.000000 | required |
| readout_match | 1.000000 | required |
| incidence_equivariance | 1.000000 | required |

## Boundary

- complete feature-row colors define attributed automorphisms; structural vectors still use discrete labels.
- token/readout invariance and mapped node-incidence equivariance are required.
- Passing confirms that continuous patch content did not break relabel legality.
