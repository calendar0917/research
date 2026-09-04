# luyin16-zinc-center-relation-network-smoke-v1

Explicit centre-level conditional fusion and bounded centre-relation propagation on official PyG ZINC.

- split sizes: `{'train': 80, 'valid': 20, 'test': 20}`
- representation: topology-only rooted-WL radius `3`, relation radius `3`
- device: `cpu`; epoch budget `3` with early stopping

| mode | valid MAE | selected epoch | test MAE after train+valid refit |
|---|---:|---:|---:|
| `center_concat` | 1.464023 | 1.0 | 1.884959 |
| `conditional_fusion` | 1.499688 | 2.0 | 1.984197 |
| `conditional_relation` | 1.479993 | 2.0 | 1.921724 |

Interpretation: `center_concat` tests centre alignment, `conditional_fusion` adds the same-centre multiplicative interaction, and `conditional_relation` additionally propagates over explicit distance/overlap/bond relations.

Runtime: `0.8s`.
