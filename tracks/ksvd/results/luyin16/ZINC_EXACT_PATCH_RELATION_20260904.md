# luyin16-zinc-exact-rooted-typed-patch-direct-relation-v1

Exact rooted typed radius-2 patches with direct explicit pair-relation readout; no message passing.

- split: `PyG ZINC subset=True official train/val/test`; sizes `{'train': 10000, 'valid': 1000, 'test': 1000}`
- descriptor width: `840`; graph feature width: `7636`
- canonicalizer: `pynauty 2.8.8.1 colored incidence-graph certificate + canonical label`
- pair relation: `unordered centre pairs: distance, four overlap ratios, root-atom equality, exact typed-patch equality, adjacent bond type, cosine and shared descriptor bits`

| head | valid MAE | test MAE after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `xgboost` | 0.515355 | 0.545977 | fixed n_estimators |
| `mlp` | 0.456223 | 0.454443 | 19 |

The feature matrix is label-free and uses all centres and all unordered centre pairs. For each patch and relation block, sum/mean/max are concatenated; the heads are the only predictors.

- feature build cache: `/home/calendar/code/research/tracks/ksvd/results/luyin16/zinc_exact_patch_relation/feature_cache.npz`; cache hit: `False`
- runtime: `147.2s`
