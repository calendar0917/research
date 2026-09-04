# luyin16-molhiv-ksvd-shell-distribution-screen-v1

Train-only K-SVD sparse patch code with explicit shortest-path pair pooling; one MLP, no message passing.

- split: `OGB ogbg-molhiv official scaffold train/valid/test`; sizes `{'train': 9600, 'valid': 1200, 'test': 1200}`
- patch descriptor: `793D`; dictionary: `K=64`, `T=4`
- pair relation: `42D`; readout: `distribution`
- message passing: `False`; attention: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.807929 | 0.831959 | 12 |

- trainable parameters: `113617`
- total parameters including frozen dictionary: `164369`
- dictionary `final` relative reconstruction: `0.414188`
- runtime: `238.2s`
