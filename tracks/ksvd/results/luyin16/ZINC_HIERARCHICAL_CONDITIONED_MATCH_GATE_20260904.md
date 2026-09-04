# ZINC hierarchical topology-conditioned match gate

Topology-only rooted-WL conditions a train-only centre-attribute prototype bank; the resulting same-centre agreement/residual readout is concatenated with hierarchical typed-WL backoff counts and sent to one XGBoost.

## Gate protocol

- scope: `official-train` only; three shuffled folds
- fixed objective: `reg:absoluteerror`; model seed: `0`
- topology/attribute shuffle: in-graph row permutation, seed `20260904`; prototype bank is not refit
- each fold fits the hierarchical vocabulary and conditional prototype bank on fold-train only

| fold | hierarchical MAE | true match MAE | shuffle MAE | baseline − true | shuffle − true | win |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.369550 | 0.402706 | 0.398103 | -0.033156 | -0.004603 | no |
| 1 | 0.358297 | 0.388811 | 0.388404 | -0.030514 | -0.000407 | no |
| 2 | 0.391483 | 0.413617 | 0.420789 | -0.022134 | +0.007173 | no |

- mean hierarchical MAE: `0.373110`
- mean true-match MAE: `0.401711`
- mean shuffle MAE: `0.402432`
- mean reduction: `-0.028601`; fold wins: `0/3`; true-vs-shuffle gap: `+0.000721`
- gate: **failed**

No Optuna search was run and official valid/test were not evaluated because the representation gate did not pass.

## Leakage boundary

- official valid vocabulary/prototype scope: `not evaluated unless gate passes; then official train-only vocabulary/prototypes and labels`
- official test vocabulary/prototype scope: `not evaluated unless gate passes; then official train+valid-only vocabulary/prototypes and labels`
- test labels used for selection: `False`

Runtime: `154.9s`; script SHA-256: `086e00843c429356ed3cc55fe1584cc309b9bc4fedc98c7b9153b214c8c530cf`.
