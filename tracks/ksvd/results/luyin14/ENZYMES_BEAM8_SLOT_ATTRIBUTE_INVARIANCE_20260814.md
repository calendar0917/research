# ENZYMES canonical-slot attribute relabel-invariance audit

> 协议：`tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_SLOT_ATTRIBUTE_INVARIANCE_PROTOCOL_20260814.md`  
> 判定：`ENZYMES_CANONICAL_SLOT_ATTRIBUTE_INVARIANCE_PASS`

| family | invariant | match rate | role |
|---|---|---:|---|
| slot_true | chain_exact_match | 0.979167 | diagnostic |
| slot_true | token_row_match | 1.000000 | required |
| slot_true | token_multiset_match | 1.000000 | required |
| slot_true | readout_match | 1.000000 | required |
| slot_true | incidence_equivariance | 1.000000 | required |
| slot_within_patch_shuffled | chain_exact_match | 0.979167 | diagnostic |
| slot_within_patch_shuffled | token_row_match | 1.000000 | required |
| slot_within_patch_shuffled | token_multiset_match | 1.000000 | required |
| slot_within_patch_shuffled | readout_match | 1.000000 | required |
| slot_within_patch_shuffled | incidence_equivariance | 1.000000 | required |

## Boundary

- complete feature-row colors define cover/order; structural vectors still use discrete labels。
- slot tensor preserves canonical slot-to-continuous-attribute correspondence and an occupancy mask。
- within-patch shuffle is recomputed after relabeling from the same fixed family seed。
- both TRUE and within-patch-shuffled controls must pass every required check。
