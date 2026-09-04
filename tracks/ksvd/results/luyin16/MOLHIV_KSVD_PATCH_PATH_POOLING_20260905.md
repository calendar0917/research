# luyin16-molhiv-ksvd-compressed-patch-path-pooling-v1

Train-only K-SVD sparse patch code with explicit shortest-path pair pooling; one MLP, no message passing.

- split: `OGB ogbg-molhiv official scaffold train/valid/test`; sizes `{'train': 32901, 'valid': 4113, 'test': 4113}`
- patch descriptor: `793D`; dictionary: `K=64`, `T=4`
- pair relation: `42D`; readout: `unary sum and sum-of-squares plus distance-conditioned pair sum and sum-of-squares with log pair mass`
- message passing: `False`; attention: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.828952 | 0.780345 | 12 |

- trainable parameters: `115201`
- total parameters including frozen dictionary: `165953`
- dictionary final relative reconstruction: `0.416541`
- runtime: `1733.9s`
