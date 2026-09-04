# luyin16-molhiv-exact-patch-path-conditioned-pooling-v1

Exact rooted OGB-typed patch tokens with shortest-path-conditioned pair pooling; one MLP, no message passing.

- split: `OGB ogbg-molhiv official scaffold train/valid/test`; sizes `{'train': 32901, 'valid': 4113, 'test': 4113}`
- patch shell descriptor: `793D`; relation descriptor: `42D`
- readout: `unary sum and sum-of-squares plus distance-conditioned pair sum and sum-of-squares with log pair mass`
- message passing: `False`; attention: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.802840 | 0.785195 | 4 |

- train-only valid typed token coverage: `0.9263`
- train+valid test typed token coverage: `0.9414`
- trainable parameters: `1001809`
- runtime: `1686.0s`
