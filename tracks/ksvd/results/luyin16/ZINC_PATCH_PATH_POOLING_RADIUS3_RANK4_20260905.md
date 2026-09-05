# luyin16-zinc-exact-r3-patch-path-pooling-factorized-rank4-v1

Exact rooted typed patch tokens with shortest-path-conditioned pair pooling; one MLP, no message passing.

- split: `PyG ZINC subset=True official train/val/test`; sizes `{'train': 10000, 'valid': 1000, 'test': 1000}`
- patch shell descriptor: `190D`; relation descriptor: `23D`
- readout: `configurable invariant unary/pair pooling with log mass`
- message passing: `False`; attention: `False`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.186123 | 0.148832 | 48 |

- train-only valid typed token coverage: `0.8449`
- train+valid test typed token coverage: `0.8495`
- trainable parameters: `201265`
- runtime: `554.8s`
