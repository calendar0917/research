# luyin16-zinc-center-relation-network-dev-v1

Explicit centre-level conditional fusion and bounded centre-relation propagation on official PyG ZINC.

- split sizes: `{'train': 2000, 'valid': 200, 'test': 200}`
- representation: topology-only rooted-WL radius `3`, relation radius `3`
- device: `cpu`; epoch budget `40` with early stopping

| mode | valid MAE | selected epoch | test MAE after train+valid refit |
|---|---:|---:|---:|
| `attribute_only` | 0.638694 | 10.0 | 0.550697 |
| `structure_only` | 1.210940 | 15.0 | 1.292168 |
| `center_concat` | 0.471925 | 25.0 | 0.422949 |
| `conditional_fusion` | 0.469370 | 30.0 | 0.421057 |
| `conditional_relation` | 0.482359 | 20.0 | 0.435548 |

Interpretation: `center_concat` tests centre alignment, `conditional_fusion` adds the same-centre multiplicative interaction, and `conditional_relation` additionally propagates over explicit distance/overlap/bond relations.

Runtime: `59.9s`.
