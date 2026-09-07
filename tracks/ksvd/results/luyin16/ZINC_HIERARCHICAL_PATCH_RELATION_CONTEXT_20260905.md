# luyin16-zinc-hierarchical-patch-relation-context-v1

Exact rooted typed patch tokens with shortest-path-conditioned pair pooling, one zero-initialised centre-context relation update, and one MLP.

- split: `PyG ZINC subset=True official train/val/test`; sizes `{'train': 10000, 'valid': 1000, 'test': 1000}`
- patch shell descriptor: `146D`; relation descriptor: `23D`
- readout: `configurable invariant unary/pair pooling with log mass`
- message passing: `True`; attention: `False`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.181589 | 0.134610 | 59 |

- train-only valid typed token coverage: `0.9880`
- train+valid test typed token coverage: `0.9899`
- trainable parameters: `277053`
- runtime: `733.1s`
