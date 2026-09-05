# luyin16-molhiv-unified-relational-patch-readout-v1

Unified relational patch readout: hierarchical patch keys, train-only K-SVD-init prototype code, invariant unary/joint moments, explicit distance-conditioned cross-centre moments, and fixed signed count sketches; one MLP head.

- dataset: `molhiv`; split: `OGB ogbg-molhiv official scaffold train/valid/test`
- descriptor: `793D`; shared graph feature width: `4746D`
- prototype: `K=16`, `T=2`, train-only K-SVD initialization
- readout: `unary mean/std/max + within-patch centered joint + distance-conditioned cross-centre centered joint + exact/parent unary sketches + unordered patch-pair/relation sketch + relation moments`
- message passing: `False`; attention: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.806790 | 0.761859 | 1 |

- trainable parameters: `152513`
- valid-phase train-only descriptor sample: `20000` patch rows
- valid-phase train-only exact/descriptor patch hashes: `79844`
- runtime: `202.1s`
