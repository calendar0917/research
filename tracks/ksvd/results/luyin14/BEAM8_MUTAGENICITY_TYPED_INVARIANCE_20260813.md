# Beam8/Mutagenicity typed relabel-invariance audit

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_INVARIANCE_PROTOCOL_20260813.md`  
> 判定：`TYPED_BEAM8_INVARIANCE_FAILED_REIMPLEMENT_REQUIRED`

- graphs/permutations：`512` / `3`；

| invariant | match rate |
|---|---:|
| chain_match | 0.477214 |
| token_row_match | 0.750651 |
| token_multiset_match | 0.764974 |
| readout_match | 0.765625 |

## Boundary

- permutation 使用 new-index → old-index 映射；node features 与 typed adjacency 同步重排。
- gate 要求所有 trial 四项严格为 1.0。
