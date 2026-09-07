# luyin16-molhiv-unified-typed-relational-patch-encoder-v2

Trainable patch/pair encoder with explicit symmetric relations and one invariant readout head; no message passing, attention, K-SVD, or ensemble.

- dataset: `molhiv`; split: `OGB ogbg-molhiv official scaffold train/valid/test`
- patch input: structure `84D` + attribute `709D` + auxiliary mass `4D`
- exact/backoff: `exact rooted typed certificate embedding for ZINC; OOV-only for MolHIV`; `coarse invariant descriptor embedding for ZINC; OOV-only for MolHIV`
- pair input: relation `42D`, `6` distance buckets
- readout: `patch/pair MLP before sum + sum-of-squares + max + log-count invariant pooling`
- message passing: `False`; attention: `False`; K-SVD: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `one_head` | 0.838294 | 0.772667 | 11 |

- trainable parameters: `115693`
- valid train-only exact token coverage: `0.0000`
- runtime: `805.7s`
