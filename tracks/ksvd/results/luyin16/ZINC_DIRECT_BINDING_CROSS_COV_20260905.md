# luyin16-zinc-direct-structure-binding-cross-cov-xgb-v1

ZINC direct topology-role × attribute binding and cross-centre covariance; no WL, K-SVD, GNN, or attention.

## Protocol

- split: `PyG ZINC subset=True official train/val/test`; sizes: `{'train': 10000, 'valid': 1000, 'test': 1000}`
- one XGBoost/model seed: `0`; objective: `reg:absoluteerror`; metric: `MAE`
- tuning: `12` trials on `3` official-train folds only
- valid: PCA fit on official train, then model fit on official train
- test: PCA and model refit on official train+valid after parameters are frozen

## Direct patch definition

- patch: every atom centre, induced radius-3 ego patch
- `T`: node `(shell, induced degree, cycle)` and edge `(unordered shell pair, cycle)` roles; topology-only
- `A`: centre atom one-hot, patch atom histogram, incident-bond histogram, patch bond histogram
- node binding: `1120D`; edge binding: `80D`
- cross covariance: `Cov_v(T_v,A_v)` with raw width `6400D`
- interaction PCA: separate train-only `16D` projection for cross_cov and binding
- invariance audit: **True**, maximum drift `2.384e-07`

## Results

| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |
|---|---:|---:|---:|---:|
| `s` | 62 | 0.563881 | 0.559099 | 0.592362 |
| `s_marginal` | 395 | 0.470485 | 0.460278 | 0.445707 |
| `s_cross_cov` | 411 | 0.453161 | 0.441877 | 0.435351 |
| `s_binding` | 411 | 0.449619 | 0.440913 | 0.413400 |
| `s_both` | 427 | 0.444800 | 0.448394 | 0.422868 |

Negative deltas are improvements in MAE.
- `s_both - s_marginal`: valid `-0.011884`; test `-0.022839`
- train-CV selected view: `s_both`

## PCA scope metadata

```json
{
  "official_train_cv": [
    {
      "fold": 0,
      "n_train": 6666,
      "n_valid": 3334,
      "projection": {
        "binding": {
          "explained_variance_ratio_sum": 0.7482636570930481,
          "fit_rows": 6666,
          "n_components": 16,
          "raw_width": 2400
        },
        "cross_cov": {
          "explained_variance_ratio_sum": 0.8239893913269043,
          "fit_rows": 6666,
          "n_components": 16,
          "raw_width": 6400
        }
      }
    },
    {
      "fold": 1,
      "n_train": 6667,
      "n_valid": 3333,
      "projection": {
        "binding": {
          "explained_variance_ratio_sum": 0.7489708662033081,
          "fit_rows": 6667,
          "n_components": 16,
          "raw_width": 2400
        },
        "cross_cov": {
          "explained_variance_ratio_sum": 0.8246638178825378,
          "fit_rows": 6667,
          "n_components": 16,
          "raw_width": 6400
        }
      }
    },
    {
      "fold": 2,
      "n_train": 6667,
      "n_valid": 3333,
      "projection": {
        "binding": {
          "explained_variance_ratio_sum": 0.7493823766708374,
          "fit_rows": 6667,
          "n_components": 16,
          "raw_width": 2400
        },
        "cross_cov": {
          "explained_variance_ratio_sum": 0.8245113492012024,
          "fit_rows": 6667,
          "n_components": 16,
          "raw_width": 6400
        }
      }
    }
  ],
  "official_train_to_valid": {
    "binding": {
      "explained_variance_ratio_sum": 0.7485624551773071,
      "fit_rows": 10000,
      "n_components": 16,
      "raw_width": 2400
    },
    "cross_cov": {
      "explained_variance_ratio_sum": 0.8241434097290039,
      "fit_rows": 10000,
      "n_components": 16,
      "raw_width": 6400
    }
  },
  "official_train_valid_to_test": {
    "binding": {
      "explained_variance_ratio_sum": 0.7484773397445679,
      "fit_rows": 11000,
      "n_components": 16,
      "raw_width": 2400
    },
    "cross_cov": {
      "explained_variance_ratio_sum": 0.824007511138916,
      "fit_rows": 11000,
      "n_components": 16,
      "raw_width": 6400
    }
  }
}
```

Runtime: `563.8s`; feature cache hit: `False`.
