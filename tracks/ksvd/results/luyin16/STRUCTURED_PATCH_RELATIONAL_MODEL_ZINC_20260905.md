# luyin16-zinc-structured-patch-relational-model-v1

Structured rooted typed patch model: separate topology/attribute encoders, low-rank within-centre binding, distance-conditioned pair products, invariant moments, and one linear graph head.

- dataset: `zinc`; split: `PyG ZINC subset=True official train/valid/test`
- patch input: structure `448D` + attribute `392D` + geometry `4D`
- pair input: relation `23D`, `5` distance buckets
- readout: `separate T/A encoders + low-rank binding; pair products by distance; sum/sumsq/max/log-count`
- WL/K-SVD/PCA upstream statistics: `none` / `none` / `none`
- message passing/attention: `False` / `False`

| model | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `structured_single_head` | 0.244129 | 0.205874 | 50 |

- trainable parameters: `64005`
- runtime: `504.9s`

The linear head makes the prediction an exact additive decomposition over structure, attribute, within-patch binding, geometry, cross-centre structure/attribute/binding products, relation descriptors and global context.
