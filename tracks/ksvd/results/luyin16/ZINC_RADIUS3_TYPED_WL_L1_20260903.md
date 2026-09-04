# luyin16-zinc-radius3-typed-wl-l1-xgb-v1

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
- local object: invariant all-centre radius-3 typed-WL (3 rounds) role/attribute marginals
- XGBoost objective: `reg:absoluteerror`; eval metric: `mae`

## Results

| view | dim | train CV MAE | valid MAE mean ± std | valid seed ensemble | test MAE mean ± std | test seed ensemble |
|---|---:|---:|---:|---:|---:|---:|
| `s` | 62 | 0.559860 | 0.560937 ± 0.002460 | 0.555685 | 0.595750 ± 0.002063 | 0.590888 |
| `s_marginal` | 323 | 0.554267 | 0.544409 ± 0.003132 | 0.538941 | 0.551926 ± 0.003951 | 0.545091 |

## Increment over S

- valid: `S+marginal - S = -0.016527 MAE`
- test after train+valid refit: `S+marginal - S = -0.043824 MAE`

## Selected parameters

```json
{
  "s": {
    "colsample_bytree": 0.6964070171094447,
    "gamma": 4.530150399962255,
    "learning_rate": 0.13407942947852833,
    "max_depth": 2,
    "min_child_weight": 10.962961341506011,
    "n_estimators": 728,
    "reg_alpha": 0.03449430931449621,
    "reg_lambda": 10.064551714457442,
    "subsample": 0.7347758355379018
  },
  "s_marginal": {
    "colsample_bytree": 0.35431259596012277,
    "gamma": 3.4740910790425223,
    "learning_rate": 0.06261605664777106,
    "max_depth": 4,
    "min_child_weight": 1.4934658205656586,
    "n_estimators": 896,
    "reg_alpha": 0.00848516031455313,
    "reg_lambda": 0.010269463911391752,
    "subsample": 0.9718799667152529
  }
}
```

Runtime: `462.8s`; feature cache hit: `False`.
