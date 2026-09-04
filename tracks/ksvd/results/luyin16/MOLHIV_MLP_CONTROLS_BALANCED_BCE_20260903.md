# luyin16-molhiv-mlp-downstream-controls-balanced-bce-v1

MolHIV downstream-control experiment for the frozen S+marginal base.

- split sizes: {'train': 32901, 'valid': 4113, 'test': 4113}
- global input: 508D (S_v1 + radius-2 centre marginal mean/std + five graph context values)
- device: cpu; budget: 50 epochs; seed: [0]
- loss: balanced BCE-with-logits (`pos_weight=N_negative/N_positive`); metric: ROC-AUC
- test policy: best official-valid epoch, then train+valid refit; test was not used for selection

| model | valid best AUC | selected epoch | test AUC |
|---|---:|---:|---:|
| `s_marginal_mlp` | 0.816225 | 22 | 0.745364 |
| `center_fusion_plus_s_marginal_mlp` | 0.805118 | 8 | 0.744364 |

The second model uses centre-level `[s_v, a_v, s_v*a_v]` fusion before sum/mean/std pooling, then concatenates the pooled centre state with the graph-level S+marginal representation.

Runtime: 269.4s.
