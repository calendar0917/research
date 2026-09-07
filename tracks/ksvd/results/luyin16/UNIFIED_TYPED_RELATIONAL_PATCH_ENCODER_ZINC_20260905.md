# luyin16-zinc-unified-typed-relational-patch-encoder-v2

Trainable patch/pair encoder with explicit symmetric relations and one invariant readout head; no message passing, attention, K-SVD, or ensemble.

- dataset: `zinc`; split: `PyG ZINC subset=True official train/valid/test`
- patch input: structure `448D` + attribute `392D` + auxiliary mass `4D`
- exact/backoff: `exact rooted typed certificate embedding for ZINC; OOV-only for MolHIV`; `coarse invariant descriptor embedding for ZINC; OOV-only for MolHIV`
- pair input: relation `23D`, `5` distance buckets
- readout: `patch/pair MLP before sum + sum-of-squares + max + log-count invariant pooling`
- message passing: `False`; attention: `False`; K-SVD: `False`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `one_head` | 0.192960 | 0.163967 | 50 |

- trainable parameters: `166509`
- valid train-only exact token coverage: `1.0000`
- runtime: `556.3s`
