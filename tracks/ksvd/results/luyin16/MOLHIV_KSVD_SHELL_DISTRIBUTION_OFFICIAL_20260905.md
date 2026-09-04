# luyin16-molhiv-ksvd-shell-distribution-official-v1

Train-only K-SVD sparse patch code with explicit shortest-path pair pooling; one MLP, no message passing.

- split: `OGB ogbg-molhiv official scaffold train/valid/test`; sizes `{'train': 32901, 'valid': 4113, 'test': 4113}`
- patch descriptor: `793D`; dictionary: `K=64`, `T=4`
- pair relation: `42D`; readout: `distribution`
- message passing: `False`; attention: `False`

| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |
|---|---:|---:|---:|
| `single_mlp` | 0.796909 | 0.794052 | 7 |

- trainable parameters: `180737`
- total parameters including frozen dictionary: `231489`
- dictionary `final` relative reconstruction: `0.416541`
- runtime: `1581.6s`
