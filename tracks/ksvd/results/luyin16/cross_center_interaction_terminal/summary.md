# Centre-level structure--attribute frozen official-valid evaluation

Protocol: `luyin16-molhiv-cross-center-interaction-terminal-valid-v1`

Hyperparameters were searched only on official-train scaffold folds. Official-test was not encoded or evaluated.

Official train/valid: 32901/4113 graphs.

| view | scaffold CV after tuning | fixed valid mean | tuned valid mean | tuned valid ensemble |
|---|---:|---:|---:|---:|
| `s` | 0.761323 | 0.779013 | 0.791586 | 0.793498 |
| `s_marginal` | 0.787106 | 0.818908 | 0.841285 | 0.845047 |
| `s_cross_cov` | 0.789985 | 0.818362 | 0.839257 | 0.842530 |
| `s_binding` | 0.788789 | 0.817119 | 0.838859 | 0.841449 |
| `s_both` | 0.790662 | 0.824530 | 0.838222 | 0.841259 |

Selected view by official-train scaffold CV: `s_both`.

## Frozen parameters

```json
{
  "n_estimators": 420,
  "max_depth": 7,
  "learning_rate": 0.0288815572,
  "min_child_weight": 11.0,
  "subsample": 0.7669478928,
  "colsample_bytree": 0.7852959683,
  "reg_lambda": 29.1674476974,
  "reg_alpha": 0.09251054396,
  "gamma": 0.0,
  "objective": "binary:logistic",
  "eval_metric": "auc",
  "tree_method": "hist",
  "random_state": 0,
  "n_jobs": 8,
  "max_bin": 256
}
```

No test score is reported by this protocol.
