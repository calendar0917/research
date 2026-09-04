# luyin16-zinc-exact-r2-token-r3-outer-context-patch-path-v1

Exact rooted typed patch tokens with shortest-path-conditioned pair pooling; one MLP, no message passing.

- split: `PyG ZINC subset=True official train/val/test`; sizes `{'train': 10000, 'valid': 1000, 'test': 1000}`
- patch shell descriptor: `146D`; relation descriptor: `23D`
- readout: `configurable invariant unary/pair pooling with log mass`
- message passing: `False`; attention: `False`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.181949 | 0.159699 | 46 |

- train-only valid typed token coverage: `0.9880`
- train+valid test typed token coverage: `0.9899`
- trainable parameters: `231549`
- runtime: `553.6s`
