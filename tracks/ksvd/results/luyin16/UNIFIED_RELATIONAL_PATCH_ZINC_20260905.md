# luyin16-zinc-unified-relational-patch-readout-v1

Unified relational patch readout: hierarchical patch keys, train-only K-SVD-init prototype code, invariant unary/joint moments, explicit distance-conditioned cross-centre moments, and fixed signed count sketches; one MLP head.

- dataset: `zinc`; split: `PyG ZINC official train/valid/test`
- descriptor: `840D`; shared graph feature width: `4473D`
- prototype: `K=16`, `T=2`, train-only K-SVD initialization
- readout: `unary mean/std/max + within-patch centered joint + distance-conditioned cross-centre centered joint + exact/parent unary sketches + unordered patch-pair/relation sketch + relation moments`
- message passing: `False`; attention: `False`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.658220 | 0.666607 | 27 |

- trainable parameters: `143777`
- valid-phase train-only descriptor sample: `20000` patch rows
- valid-phase train-only exact/descriptor patch hashes: `6784`
- runtime: `59.8s`
