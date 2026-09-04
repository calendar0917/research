# luyin16-zinc-exact-patch-path-conditioned-pooling-factorized-rank20-meanstd-v1

Exact rooted typed patch tokens with shortest-path-conditioned pair pooling; one MLP, no message passing.

- split: `PyG ZINC subset=True official train/val/test`; sizes `{'train': 10000, 'valid': 1000, 'test': 1000}`
- patch shell descriptor: `146D`; relation descriptor: `23D`
- readout: `configurable invariant unary/pair pooling with log mass`
- message passing: `False`; attention: `False`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.187110 | 0.152431 | 49 |

- train-only valid typed token coverage: `0.9880`
- train+valid test typed token coverage: `0.9899`
- trainable parameters: `228861`
- runtime: `562.3s`
