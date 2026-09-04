# ZINC decomposable conditional structure--attribute statistics

## Protocol

- split: `PyG ZINC subset=True official train/val/test`; sizes: `{'test': 1000, 'train': 10000, 'valid': 1000}`
- objective: `reg:absoluteerror`; model seed: `0`
- Optuna: `20` trials, `3` train-only folds
- condition block: reusable root-WL-role × aligned attribute cross moments; no exact joint IDs

## Results

| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |
|---|---:|---:|---:|---:|
| `s_factorized` | 526 | 0.512845 | 0.501143 | 0.511459 |
| `s_conditional_r3` | 4622 | 0.498049 | 0.484097 | 0.502942 |
| `s_conditional_multiscale` | 16910 | 0.472972 | 0.448777 | 0.471260 |

- CV-selected view: `s_conditional_multiscale`
- feature cache hit: `False`

## Representation

```json
{
  "attribute_definition": "centre atom + incident bond histogram + local atom/bond histograms",
  "centres": "every atom",
  "conditional_definition": "E[1{root role=r} * aligned attribute vector] per selected WL level",
  "edge_role_bins": 32,
  "feature_metadata": {
    "test": {
      "condition_multiscale": {
        "levels": [
          0,
          1,
          2,
          3
        ],
        "mean_nnz_per_graph": 357.046,
        "nnz": 357046,
        "width": 16384
      },
      "condition_r3": {
        "levels": [
          3
        ],
        "mean_nnz_per_graph": 124.663,
        "nnz": 124663,
        "width": 4096
      },
      "mean_centres": 23.117,
      "n_graphs": 1000,
      "observed_roles_by_level": [
        4,
        36,
        64,
        64
      ]
    },
    "train": {
      "condition_multiscale": {
        "levels": [
          0,
          1,
          2,
          3
        ],
        "mean_nnz_per_graph": 357.8028,
        "nnz": 3578028,
        "width": 16384
      },
      "condition_r3": {
        "levels": [
          3
        ],
        "mean_nnz_per_graph": 125.038,
        "nnz": 1250380,
        "width": 4096
      },
      "mean_centres": 23.1664,
      "n_graphs": 10000,
      "observed_roles_by_level": [
        4,
        38,
        64,
        64
      ]
    },
    "valid": {
      "condition_multiscale": {
        "levels": [
          0,
          1,
          2,
          3
        ],
        "mean_nnz_per_graph": 355.272,
        "nnz": 355272,
        "width": 16384
      },
      "condition_r3": {
        "levels": [
          3
        ],
        "mean_nnz_per_graph": 124.664,
        "nnz": 124664,
        "width": 4096
      },
      "mean_centres": 23.083,
      "n_graphs": 1000,
      "observed_roles_by_level": [
        4,
        35,
        64,
        64
      ]
    }
  },
  "node_role_bins": 64,
  "radius": 3,
  "structure_definition": "topology-only rooted-WL final root role + role/edge histograms + shell/size/cycle statistics",
  "wl_levels": [
    0,
    1,
    2,
    3
  ]
}
```

Runtime: `3830.1s`; script SHA-256: `d080b824aa8a61e3a21f95e549b77a7c0d67a944424a9c0ce3a85242fea79058`.
