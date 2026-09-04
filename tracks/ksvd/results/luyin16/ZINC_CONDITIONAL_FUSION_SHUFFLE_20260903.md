# luyin16-zinc-conditional-fusion-shuffle-confirmation-v1

Explicit centre-level conditional fusion and bounded centre-relation propagation on official PyG ZINC.

- split sizes: `{'train': 10000, 'valid': 1000, 'test': 1000}`
- representation: topology-only rooted-WL radius `3`, relation radius `3`
- device: `cpu`; epoch budget `100` with early stopping

| mode | valid MAE | selected epoch | test MAE after train+valid refit |
|---|---:|---:|---:|
| `conditional_fusion` | 0.216337 | 86.7 | 0.212565 |
| `conditional_fusion_shuffle` | 0.326837 | 63.3 | 0.318601 |

Interpretation: `center_concat` tests centre alignment, `conditional_fusion` adds the same-centre multiplicative interaction, and `conditional_relation` additionally propagates over explicit distance/overlap/bond relations.

Runtime: `1156.6s`.
