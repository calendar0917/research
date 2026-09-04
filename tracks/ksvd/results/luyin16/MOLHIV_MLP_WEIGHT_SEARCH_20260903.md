# luyin16-molhiv-mlp-lightweight-pos-weight-search-v1

MolHIV search over lightweight positive-class weights for the MLP controls.

- split sizes: {'train': 32901, 'valid': 4113, 'test': 4113}
- candidates: `[2.0, 5.0, 10.0]`
- device: cpu; budget: 50 epochs; seed: [0]
- loss: BCE-with-logits with the candidate `pos_weight`; metric: ROC-AUC
- selection: highest official-valid AUC per model; test is run only for the selected weight

| model | weight | valid best AUC | epoch | test AUC |
|---|---:|---:|---:|---:|
| `s_marginal_mlp` | 10.0 | 0.830666 | 9 | 0.732772 |
| `center_fusion_plus_s_marginal_mlp` | 10.0 | 0.827075 | 6 | 0.741370 |

## Validation search

| model | weight | valid best AUC | epoch |
|---|---:|---:|---:|
| `s_marginal_mlp` | 2.0 | 0.809288 | 6 |
| `s_marginal_mlp` | 5.0 | 0.809450 | 11 |
| `s_marginal_mlp` | 10.0 | 0.830666 | 9 |
| `center_fusion_plus_s_marginal_mlp` | 2.0 | 0.814135 | 5 |
| `center_fusion_plus_s_marginal_mlp` | 5.0 | 0.819004 | 6 |
| `center_fusion_plus_s_marginal_mlp` | 10.0 | 0.827075 | 6 |

Runtime: 703.8s.
