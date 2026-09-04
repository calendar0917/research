# luyin16-zinc-exact-canonical-ksvd-patch-path-pooling-v1

Exact invariant rooted patch descriptor compressed by train-only K-SVD; explicit shortest-path pair pooling and one MLP, no message passing.

- split: `PyG ZINC subset=True official train/val/test`; sizes `{'train': 10000, 'valid': 1000, 'test': 1000}`
- patch descriptor: `840D exact canonical`; dictionary: `K=64`, `T=4`
- pair relation: `23D`; readout: `moments`
- message passing: `False`; attention: `False`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.258168 | 0.243949 | 32 |

- trainable parameters: `101281`
- total parameters including frozen dictionary: `155041`
- dictionary `final` relative reconstruction: `0.639896`
- runtime: `504.4s`
