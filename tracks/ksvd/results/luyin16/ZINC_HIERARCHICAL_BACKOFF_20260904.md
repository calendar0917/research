# ZINC hierarchical exact-token backoff

Train-only exact typed-WL vocabularies with same-centre lower-order fallback.

## Protocol

- split: `PyG ZINC subset=True official train/val/test`; sizes: `{'train': 10000, 'valid': 1000, 'test': 1000}`
- objective: `reg:absoluteerror`; model seed: `0`
- top-K per WL round: `2048`
- valid vocabulary: official train only; test vocabulary: train+valid only
- Optuna: `20` trials, `3` train-only folds

## Results

| view | dimension | valid MAE | test MAE after train+valid refit |
|---|---:|---:|---:|
| `s_wl_count` | 16454 | 0.372258 | 0.375721 |
| `s_hierarchical_backoff` | 41048 | 0.354605 | 0.345460 |

## Increment (positive means lower MAE)

- valid: `+0.017652`
- test: `+0.030261`

## Backoff coverage

### train

```json
{
  "backoff_blocks": [
    {
      "resolved_occurrences": 0.0,
      "source_round": 1,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 70143.0,
      "source_round": 2,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 0.0,
      "source_round": 2,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 60315.0,
      "source_round": 3,
      "target_round": 2,
      "width": 4098
    },
    {
      "resolved_occurrences": 69837.0,
      "source_round": 3,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 0.0,
      "source_round": 3,
      "target_round": 0,
      "width": 4098
    }
  ],
  "exact_width": 16392,
  "source_rounds": {
    "0": {
      "exact_known": 231664.0,
      "exact_unknown": 0.0,
      "occurrences": 231664.0,
      "unresolved": 0.0
    },
    "1": {
      "exact_known": 231664.0,
      "exact_unknown": 0.0,
      "occurrences": 231664.0,
      "unresolved": 0.0
    },
    "2": {
      "exact_known": 161521.0,
      "exact_unknown": 70143.0,
      "occurrences": 231664.0,
      "unresolved": 0.0
    },
    "3": {
      "exact_known": 101512.0,
      "exact_unknown": 130152.0,
      "occurrences": 231664.0,
      "unresolved": 0.0
    }
  },
  "total_width": 40986,
  "unresolved_width": 6
}
```

### valid

```json
{
  "backoff_blocks": [
    {
      "resolved_occurrences": 39.0,
      "source_round": 1,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 6962.0,
      "source_round": 2,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 39.0,
      "source_round": 2,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 6166.0,
      "source_round": 3,
      "target_round": 2,
      "width": 4098
    },
    {
      "resolved_occurrences": 6933.0,
      "source_round": 3,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 39.0,
      "source_round": 3,
      "target_round": 0,
      "width": 4098
    }
  ],
  "exact_width": 16392,
  "source_rounds": {
    "0": {
      "exact_known": 23083.0,
      "exact_unknown": 0.0,
      "occurrences": 23083.0,
      "unresolved": 0.0
    },
    "1": {
      "exact_known": 23044.0,
      "exact_unknown": 39.0,
      "occurrences": 23083.0,
      "unresolved": 0.0
    },
    "2": {
      "exact_known": 16082.0,
      "exact_unknown": 7001.0,
      "occurrences": 23083.0,
      "unresolved": 0.0
    },
    "3": {
      "exact_known": 9945.0,
      "exact_unknown": 13138.0,
      "occurrences": 23083.0,
      "unresolved": 0.0
    }
  },
  "total_width": 40986,
  "unresolved_width": 6
}
```

### combined_train_valid

```json
{
  "backoff_blocks": [
    {
      "resolved_occurrences": 0.0,
      "source_round": 1,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 77053.0,
      "source_round": 2,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 0.0,
      "source_round": 2,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 66425.0,
      "source_round": 3,
      "target_round": 2,
      "width": 4098
    },
    {
      "resolved_occurrences": 76772.0,
      "source_round": 3,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 0.0,
      "source_round": 3,
      "target_round": 0,
      "width": 4098
    }
  ],
  "exact_width": 16392,
  "source_rounds": {
    "0": {
      "exact_known": 254747.0,
      "exact_unknown": 0.0,
      "occurrences": 254747.0,
      "unresolved": 0.0
    },
    "1": {
      "exact_known": 254747.0,
      "exact_unknown": 0.0,
      "occurrences": 254747.0,
      "unresolved": 0.0
    },
    "2": {
      "exact_known": 177694.0,
      "exact_unknown": 77053.0,
      "occurrences": 254747.0,
      "unresolved": 0.0
    },
    "3": {
      "exact_known": 111550.0,
      "exact_unknown": 143197.0,
      "occurrences": 254747.0,
      "unresolved": 0.0
    }
  },
  "total_width": 40986,
  "unresolved_width": 6
}
```

### test

```json
{
  "backoff_blocks": [
    {
      "resolved_occurrences": 31.0,
      "source_round": 1,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 7108.0,
      "source_round": 2,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 31.0,
      "source_round": 2,
      "target_round": 0,
      "width": 4098
    },
    {
      "resolved_occurrences": 5994.0,
      "source_round": 3,
      "target_round": 2,
      "width": 4098
    },
    {
      "resolved_occurrences": 7091.0,
      "source_round": 3,
      "target_round": 1,
      "width": 4098
    },
    {
      "resolved_occurrences": 31.0,
      "source_round": 3,
      "target_round": 0,
      "width": 4098
    }
  ],
  "exact_width": 16392,
  "source_rounds": {
    "0": {
      "exact_known": 23117.0,
      "exact_unknown": 0.0,
      "occurrences": 23117.0,
      "unresolved": 0.0
    },
    "1": {
      "exact_known": 23086.0,
      "exact_unknown": 31.0,
      "occurrences": 23117.0,
      "unresolved": 0.0
    },
    "2": {
      "exact_known": 15978.0,
      "exact_unknown": 7139.0,
      "occurrences": 23117.0,
      "unresolved": 0.0
    },
    "3": {
      "exact_known": 10001.0,
      "exact_unknown": 13116.0,
      "occurrences": 23117.0,
      "unresolved": 0.0
    }
  },
  "total_width": 40986,
  "unresolved_width": 6
}
```

Runtime: `547.8s`; script SHA-256: `e9060e7daa0c219cb6de8e5a44353518263dd897cc45d38a55096de49bada440`.
