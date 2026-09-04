# luyin16-zinc-center-relation-network-official-fast-v1

Explicit centre-level conditional fusion and bounded centre-relation propagation on official PyG ZINC.

- split sizes: `{'train': 10000, 'valid': 1000, 'test': 1000}`
- representation: topology-only rooted-WL radius `3`, relation radius `3`
- device: `cpu`; epoch budget `60` with early stopping

| mode | valid MAE | selected epoch | test MAE after train+valid refit |
|---|---:|---:|---:|
| `attribute_only` | 0.494290 | 25.0 | 0.530752 |
| `structure_only` | 1.119067 | 60.0 | 1.158670 |
| `center_concat` | 0.269540 | 50.0 | 0.272167 |
| `conditional_fusion` | 0.238657 | 55.0 | 0.209998 |
| `conditional_relation` | 0.219294 | 60.0 | 0.238209 |

Interpretation: `center_concat` tests centre alignment, `conditional_fusion` adds the same-centre multiplicative interaction, and `conditional_relation` additionally propagates over explicit distance/overlap/bond relations.

Runtime: `721.4s`.
