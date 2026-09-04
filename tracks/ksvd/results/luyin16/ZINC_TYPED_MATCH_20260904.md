# luyin16-zinc-typed-match-xgb-v1

Auditable candidate for typed matching: train-only frequent multi-level typed-WL prototypes, soft prefix response readout, and one XGBoost regressor.

> This is a candidate implementation, not an exact claim about the mentor's unavailable internal typed-match code.

## Protocol

- split: `PyG ZINC subset=True official train/val/test`; sizes: `{'train': 10000, 'valid': 1000, 'test': 1000}`
- objective: `reg:absoluteerror`; metric: `MAE`; model seed: `0`
- Optuna: `20` trials, `3` shuffled folds inside official train
- prototype bank: fold-local for CV; official train only for valid; official train+valid only for test
- K-SVD: disabled

## Representation

- typed-WL levels: `0/1/2/3`; prototypes: `1024` requested / `1024` selected
- level weights: `[0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645]`
- feature dimension: `2143` (`S`=`62`, typed-match=`2081`)
- match readout: per-prototype weighted prefix-match mean/max, response summaries, best-centre score and residual summaries

## Results

| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |
|---|---:|---:|---:|---:|
| `S + typed_match` | 2143 | 0.431489 | 0.410045 | 0.395111 |
| `S` capacity-matched reference | 62 | — | 0.562754 | 0.606813 |

## Coverage audit

```json
{
  "combined_bank_train_valid": {
    "bank": {
      "fit_centres": 254747,
      "fit_graphs": 11000,
      "requested_prototypes": 1024,
      "selected_occurrence_fraction": 0.3532524426195402,
      "selected_occurrences": 89990,
      "selected_prototypes": 1024,
      "unique_complete_signatures": 71102
    },
    "combined_train_valid": {
      "deepest_match_fraction": [
        0.006641883908348283,
        0.20074426784221208,
        0.30922837167856737,
        0.1301330339513321,
        0.3532524426195402
      ],
      "dimension": 2081,
      "full_level_match_fraction": 0.3532524426195402,
      "level_weights": [
        0.10000000149011612,
        0.20000000298023224,
        0.30000001192092896,
        0.4000000059604645
      ],
      "prototype_count": 1024,
      "readout": "per-prototype weighted prefix-match mean/max; response summaries; best-centre match and residual summaries"
    },
    "test": {
      "deepest_match_fraction": [
        0.006964571527447333,
        0.20305402950209803,
        0.3101613531167539,
        0.13051001427520872,
        0.349310031578492
      ],
      "dimension": 2081,
      "full_level_match_fraction": 0.349310031578492,
      "level_weights": [
        0.10000000149011612,
        0.20000000298023224,
        0.30000001192092896,
        0.4000000059604645
      ],
      "prototype_count": 1024,
      "readout": "per-prototype weighted prefix-match mean/max; response summaries; best-centre match and residual summaries"
    }
  },
  "train_bank_train": {
    "bank": {
      "fit_centres": 231664,
      "fit_graphs": 10000,
      "requested_prototypes": 1024,
      "selected_occurrence_fraction": 0.3535896816078458,
      "selected_occurrences": 81914,
      "selected_prototypes": 1024,
      "unique_complete_signatures": 67003
    },
    "test": {
      "deepest_match_fraction": [
        0.006964571527447333,
        0.20404896829173336,
        0.3086905740364234,
        0.12994765756802354,
        0.3503482285763724
      ],
      "dimension": 2081,
      "full_level_match_fraction": 0.3503482285763724,
      "level_weights": [
        0.10000000149011612,
        0.20000000298023224,
        0.30000001192092896,
        0.4000000059604645
      ],
      "prototype_count": 1024,
      "readout": "per-prototype weighted prefix-match mean/max; response summaries; best-centre match and residual summaries"
    },
    "train": {
      "deepest_match_fraction": [
        0.006625975550797707,
        0.20245700669935768,
        0.3091071551902756,
        0.1282201809517232,
        0.3535896816078458
      ],
      "dimension": 2081,
      "full_level_match_fraction": 0.3535896816078458,
      "level_weights": [
        0.10000000149011612,
        0.20000000298023224,
        0.30000001192092896,
        0.4000000059604645
      ],
      "prototype_count": 1024,
      "readout": "per-prototype weighted prefix-match mean/max; response summaries; best-centre match and residual summaries"
    },
    "valid": {
      "deepest_match_fraction": [
        0.006801542260538058,
        0.2027032881341247,
        0.3052896070701382,
        0.1379370099207209,
        0.3472685526144782
      ],
      "dimension": 2081,
      "full_level_match_fraction": 0.3472685526144782,
      "level_weights": [
        0.10000000149011612,
        0.20000000298023224,
        0.30000001192092896,
        0.4000000059604645
      ],
      "prototype_count": 1024,
      "readout": "per-prototype weighted prefix-match mean/max; response summaries; best-centre match and residual summaries"
    }
  }
}
```

## Selected parameters

```json
{
  "colsample_bytree": 0.12158581325112827,
  "gamma": 7.34527773987479,
  "learning_rate": 0.10343963270953856,
  "max_depth": 4,
  "min_child_weight": 4.030666772181392,
  "n_estimators": 1009,
  "reg_alpha": 0.02156101206194706,
  "reg_lambda": 2.123374892698157,
  "subsample": 0.7658043742611061
}
```

Runtime: `1358.7s`; token cache hit: `True`; script SHA-256: `e8ef8f6ee6e0ecb55cd054f481bc259993d87172da62314f83127599c77d9eb0`.
