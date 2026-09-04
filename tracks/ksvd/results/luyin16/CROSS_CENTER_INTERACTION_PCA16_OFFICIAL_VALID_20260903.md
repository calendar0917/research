# Centre-level structure--attribute frozen official-valid evaluation

Protocol: `luyin16-molhiv-cross-center-interaction-terminal-valid-pca16-v1`

Hyperparameters were searched only on official-train scaffold folds. Official-test was not encoded or evaluated.

Official train/valid: 32901/4113 graphs.

| view | scaffold CV after tuning | fixed valid mean | tuned valid mean | tuned valid ensemble |
|---|---:|---:|---:|---:|
| `s` | 0.761323 | 0.779013 | 0.791586 | 0.793498 |
| `s_marginal` | 0.787106 | 0.818908 | 0.841285 | 0.845047 |
| `s_cross_cov` | 0.790890 | 0.816124 | 0.837160 | 0.838666 |
| `s_binding` | 0.792839 | 0.806192 | 0.832134 | 0.835273 |
| `s_both` | 0.788703 | 0.810521 | 0.832970 | 0.834621 |

Selected view by official-train scaffold CV: `s_binding`.

## Frozen parameters

```json
{
  "n_estimators": 597,
  "max_depth": 7,
  "learning_rate": 0.024442457492275964,
  "min_child_weight": 44.64307076086108,
  "subsample": 0.6607242789176415,
  "colsample_bytree": 0.9642764232041627,
  "reg_lambda": 23.956782625289467,
  "reg_alpha": 1.067692273019425,
  "gamma": 4.8873334070661265,
  "objective": "binary:logistic",
  "eval_metric": "auc",
  "tree_method": "hist",
  "random_state": 0,
  "n_jobs": 8,
  "max_bin": 256
}
```

No test score is reported by this protocol.
