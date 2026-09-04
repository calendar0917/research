# luyin16-zinc-s-marginal-xgb-v1

ZINC transfer of `S + marginal -> XGBoost`; no K-SVD or interaction block.

## Protocol

- split: `PyG ZINC subset=True official train/val/test`; sizes `{'train': 10000, 'valid': 1000, 'test': 1000}`
- target: penalized logP / constrained solubility; metric: MAE
- tuning: 3-fold shuffled KFold inside official train, 20 Optuna trials per view
- official valid: evaluated after train-only tuning; official test: train+valid refit

## Representation

- `S`: 62D global structure + atom/bond composition
- `marginal`: 261D local mean/std + context
- `S+marginal`: 323D
- local object: invariant all-centre radius-2 rooted-WL role/attribute marginals

## Results

| view | dim | train CV MAE | valid MAE mean ± std | valid seed ensemble | test MAE mean ± std | test seed ensemble |
|---|---:|---:|---:|---:|---:|---:|
| `s` | 62 | 0.607075 | 0.601088 ± 0.005351 | 0.594622 | 0.634988 ± 0.001922 | 0.630417 |
| `s_marginal` | 323 | 0.581493 | 0.587555 ± 0.003635 | 0.581112 | 0.573417 ± 0.002913 | 0.567567 |

## Increment over S

- valid: `S+marginal - S = -0.013534 MAE`
- test after train+valid refit: `S+marginal - S = -0.061571 MAE`

## Selected parameters

```json
{
  "s": {
    "colsample_bytree": 0.9807483610653525,
    "gamma": 2.07076207956309,
    "learning_rate": 0.06230855746123861,
    "max_depth": 4,
    "min_child_weight": 1.1554594083648573,
    "n_estimators": 885,
    "reg_alpha": 0.0003157450913066466,
    "reg_lambda": 2.2960366637049447,
    "subsample": 0.7720399610700952
  },
  "s_marginal": {
    "colsample_bytree": 0.3935113145666825,
    "gamma": 0.07487535371316834,
    "learning_rate": 0.02306031601005054,
    "max_depth": 7,
    "min_child_weight": 1.0426282920392735,
    "n_estimators": 896,
    "reg_alpha": 1.196813296539454e-05,
    "reg_lambda": 0.013646380645274226,
    "subsample": 0.8334165542204919
  }
}
```

Runtime: `392.9s`; feature cache hit: `False`.
