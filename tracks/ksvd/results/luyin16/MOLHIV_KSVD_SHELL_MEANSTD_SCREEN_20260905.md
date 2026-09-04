# luyin16-molhiv-ksvd-shell-meanstd-screen-v1

Train-only K-SVD sparse patch code with explicit shortest-path pair pooling; one MLP, no message passing.

- split: `OGB ogbg-molhiv official scaffold train/valid/test`; sizes `{'train': 9600, 'valid': 1200, 'test': 1200}`
- patch descriptor: `793D`; dictionary: `K=64`, `T=4`
- pair relation: `42D`; readout: `mean_std`
- message passing: `False`; attention: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.769452 | 0.824667 | 2 |

- trainable parameters: `76753`
- total parameters including frozen dictionary: `127505`
- dictionary `final` relative reconstruction: `0.414188`
- runtime: `125.7s`
