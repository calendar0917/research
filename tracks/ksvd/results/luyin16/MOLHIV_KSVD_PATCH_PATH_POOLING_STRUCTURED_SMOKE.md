# luyin16-molhiv-ksvd-structured-patch-path-pooling-smoke-v1

Train-only K-SVD sparse patch code with explicit shortest-path pair pooling; one MLP, no message passing.

- split: `OGB ogbg-molhiv official scaffold train/valid/test`; sizes `{'train': 480, 'valid': 60, 'test': 60}`
- patch descriptor: `1845D`; dictionary: `K=16`, `T=3`
- pair relation: `42D`; readout: `unary sum and sum-of-squares plus distance-conditioned pair sum and sum-of-squares with log pair mass`
- message passing: `False`; attention: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.847458 | 0.974138 | 1 |

- trainable parameters: `52193`
- total parameters including frozen dictionary: `81713`
- dictionary final relative reconstruction: `0.705117`
- runtime: `17.8s`
