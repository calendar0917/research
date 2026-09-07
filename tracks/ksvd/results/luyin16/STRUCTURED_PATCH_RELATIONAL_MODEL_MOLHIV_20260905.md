# luyin16-molhiv-structured-patch-relational-model-v1

Structured rooted typed patch model: separate topology/attribute encoders, low-rank within-centre binding, distance-conditioned pair products, invariant moments, and one linear graph head.

- dataset: `molhiv`; split: `OGB ogbg-molhiv official scaffold train/valid/test`
- patch input: structure `84D` + attribute `709D` + geometry `4D`
- pair input: relation `42D`, `6` distance buckets
- readout: `separate T/A encoders + low-rank binding; pair products by distance; sum/sumsq/max/log-count`
- WL/K-SVD/PCA upstream statistics: `none` / `none` / `none`
- message passing/attention: `False` / `False`

| model | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `structured_single_head` | 0.794787 | 0.753047 | 6 |

- trainable parameters: `66737`
- runtime: `617.3s`

The linear head makes the prediction an exact additive decomposition over structure, attribute, within-patch binding, geometry, cross-centre structure/attribute/binding products, relation descriptors and global context.
