# luyin16-molhiv-unified-exact-rooted-relational-patch-encoder-v2

Trainable patch/pair encoder with explicit symmetric relations and one invariant readout head; no message passing, attention, K-SVD, or ensemble.

- dataset: `molhiv`; split: `OGB ogbg-molhiv official scaffold train/valid/test`
- patch input: structure `84D` + attribute `709D` + auxiliary mass `4D`
- exact/backoff: `exact rooted typed certificate embedding`; `exact radius-1 parent certificate embedding`
- pair input: relation `42D`, `6` distance buckets
- readout: `patch/pair MLP before sum + sum-of-squares + max + log-count invariant pooling`
- message passing: `False`; attention: `False`; K-SVD: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `one_head` | 0.805203 | 0.774638 | 4 |

- trainable parameters: `325957`
- valid train-only exact token coverage: `1.0000`
- runtime: `1003.3s`
