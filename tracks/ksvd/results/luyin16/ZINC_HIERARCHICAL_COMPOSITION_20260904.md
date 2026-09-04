# ZINC hierarchical motif composition

Hierarchical exact-token backoff plus cross-level, cross-centre, rarity and molecule-composition features.

## Protocol

- split: `PyG ZINC subset=True official train/val/test`; sizes: `{'train': 10000, 'valid': 1000, 'test': 1000}`
- objective: `reg:absoluteerror`; model seed: `0`
- top-K per feature vocabulary: `2048`
- Optuna: `20` trials, `3` train-only folds
- all vocabulary/frequency fitting is train-only; test uses train+valid refit

## Results

| view | dimension | valid MAE | test MAE after train+valid refit |
|---|---:|---:|---:|
| `s_hierarchical_backoff` | 41048 | 0.354605 | 0.345460 |
| `s_hierarchical_composition` | 61592 | 0.378619 | 0.402539 |

## Increment over hierarchical backoff

- valid MAE reduction: `-0.024014`
- test MAE reduction: `-0.057079`

## Feature blocks

```json
{
  "baseline_hierarchical_combined": {
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
  },
  "baseline_hierarchical_test": {
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
  },
  "baseline_hierarchical_train": {
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
  },
  "baseline_hierarchical_valid": {
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
  },
  "combined_train_valid": {
    "cross_centre": {
      "known_fraction": 0.12847436963072112,
      "known_pairs": 376831.0,
      "total_pairs": 2933122.0,
      "unknown_fraction": 0.8715256303692789,
      "unknown_pairs": 2556291.0,
      "width": 4098
    },
    "cross_level": {
      "schemas": {
        "r0_r1_r2_r3": {
          "known": 111550.0,
          "known_fraction": 0.4378854314280443,
          "occurrences": 254747.0,
          "unknown": 143197.0,
          "unknown_fraction": 0.5621145685719557
        },
        "r1_r2": {
          "known": 177694.0,
          "known_fraction": 0.6975312761288651,
          "occurrences": 254747.0,
          "unknown": 77053.0,
          "unknown_fraction": 0.3024687238711349
        },
        "r2_r3": {
          "known": 111550.0,
          "known_fraction": 0.4378854314280443,
          "occurrences": 254747.0,
          "unknown": 143197.0,
          "unknown_fraction": 0.5621145685719557
        }
      },
      "width": 12294,
      "width_per_schema": 4098
    },
    "dimension": 61592,
    "frequency_fit_graphs": 11000,
    "hierarchical": {
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
    },
    "molecule_composition": {
      "levels": {
        "r2_parent_bag": {
          "graphs": 11000.0,
          "known": 2052.0,
          "known_fraction": 0.18654545454545454,
          "unknown": 8948.0,
          "unknown_fraction": 0.8134545454545454
        },
        "r3_parent_bag": {
          "graphs": 11000.0,
          "known": 2052.0,
          "known_fraction": 0.18654545454545454,
          "unknown": 8948.0,
          "unknown_fraction": 0.8134545454545454
        }
      },
      "width": 4098
    },
    "rarity": {
      "rounds": {
        "0": {
          "backoff": 0.0,
          "backoff_fraction": 0.0,
          "backoff_previous": 0.0,
          "backoff_previous_fraction": 0.0,
          "exact_known": 254747.0,
          "exact_known_fraction": 1.0,
          "exact_unknown": 0.0,
          "exact_unknown_fraction": 0.0,
          "occurrences": 254747.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "1": {
          "backoff": 0.0,
          "backoff_fraction": 0.0,
          "backoff_previous": 0.0,
          "backoff_previous_fraction": 0.0,
          "exact_known": 254747.0,
          "exact_known_fraction": 1.0,
          "exact_unknown": 0.0,
          "exact_unknown_fraction": 0.0,
          "occurrences": 254747.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "2": {
          "backoff": 77053.0,
          "backoff_fraction": 0.3024687238711349,
          "backoff_previous": 77053.0,
          "backoff_previous_fraction": 0.3024687238711349,
          "exact_known": 177694.0,
          "exact_known_fraction": 0.6975312761288651,
          "exact_unknown": 77053.0,
          "exact_unknown_fraction": 0.3024687238711349,
          "occurrences": 254747.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "3": {
          "backoff": 143197.0,
          "backoff_fraction": 0.5621145685719557,
          "backoff_previous": 66425.0,
          "backoff_previous_fraction": 0.26074889988890937,
          "exact_known": 111550.0,
          "exact_known_fraction": 0.4378854314280443,
          "exact_unknown": 143197.0,
          "exact_unknown_fraction": 0.5621145685719557,
          "occurrences": 254747.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        }
      },
      "width": 54
    },
    "vocabulary_fit_graphs": 11000
  },
  "official_test": {
    "cross_centre": {
      "known_fraction": 0.1273757454519426,
      "known_pairs": 33811.0,
      "total_pairs": 265443.0,
      "unknown_fraction": 0.8726242545480574,
      "unknown_pairs": 231632.0,
      "width": 4098
    },
    "cross_level": {
      "schemas": {
        "r0_r1_r2_r3": {
          "known": 9989.0,
          "known_fraction": 0.43210624215944976,
          "occurrences": 23117.0,
          "unknown": 13128.0,
          "unknown_fraction": 0.5678937578405503
        },
        "r1_r2": {
          "known": 15969.0,
          "known_fraction": 0.6907903274646364,
          "occurrences": 23117.0,
          "unknown": 7148.0,
          "unknown_fraction": 0.3092096725353636
        },
        "r2_r3": {
          "known": 9981.0,
          "known_fraction": 0.43176017649348963,
          "occurrences": 23117.0,
          "unknown": 13136.0,
          "unknown_fraction": 0.5682398235065104
        }
      },
      "width": 12294,
      "width_per_schema": 4098
    },
    "dimension": 61592,
    "frequency_fit_graphs": 11000,
    "hierarchical": {
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
    },
    "molecule_composition": {
      "levels": {
        "r2_parent_bag": {
          "graphs": 1000.0,
          "known": 0.0,
          "known_fraction": 0.0,
          "unknown": 1000.0,
          "unknown_fraction": 1.0
        },
        "r3_parent_bag": {
          "graphs": 1000.0,
          "known": 0.0,
          "known_fraction": 0.0,
          "unknown": 1000.0,
          "unknown_fraction": 1.0
        }
      },
      "width": 4098
    },
    "rarity": {
      "rounds": {
        "0": {
          "backoff": 0.0,
          "backoff_fraction": 0.0,
          "backoff_previous": 0.0,
          "backoff_previous_fraction": 0.0,
          "exact_known": 23117.0,
          "exact_known_fraction": 1.0,
          "exact_unknown": 0.0,
          "exact_unknown_fraction": 0.0,
          "occurrences": 23117.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "1": {
          "backoff": 31.0,
          "backoff_fraction": 0.0013410044555954492,
          "backoff_previous": 31.0,
          "backoff_previous_fraction": 0.0013410044555954492,
          "exact_known": 23086.0,
          "exact_known_fraction": 0.9986589955444045,
          "exact_unknown": 31.0,
          "exact_unknown_fraction": 0.0013410044555954492,
          "occurrences": 23117.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "2": {
          "backoff": 7139.0,
          "backoff_fraction": 0.30882034866115843,
          "backoff_previous": 7108.0,
          "backoff_previous_fraction": 0.307479344205563,
          "exact_known": 15978.0,
          "exact_known_fraction": 0.6911796513388415,
          "exact_unknown": 7139.0,
          "exact_unknown_fraction": 0.30882034866115843,
          "occurrences": 23117.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "3": {
          "backoff": 13116.0,
          "backoff_fraction": 0.56737465934161,
          "backoff_previous": 5994.0,
          "backoff_previous_fraction": 0.25928970022061687,
          "exact_known": 10001.0,
          "exact_known_fraction": 0.4326253406583899,
          "exact_unknown": 13116.0,
          "exact_unknown_fraction": 0.56737465934161,
          "occurrences": 23117.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        }
      },
      "width": 54
    },
    "vocabulary_fit_graphs": 11000
  },
  "official_train": {
    "cross_centre": {
      "known_fraction": 0.12865385523466596,
      "known_pairs": 343293.0,
      "total_pairs": 2668346.0,
      "unknown_fraction": 0.871346144765334,
      "unknown_pairs": 2325053.0,
      "width": 4098
    },
    "cross_level": {
      "schemas": {
        "r0_r1_r2_r3": {
          "known": 101512.0,
          "known_fraction": 0.43818633883555497,
          "occurrences": 231664.0,
          "unknown": 130152.0,
          "unknown_fraction": 0.5618136611644451
        },
        "r1_r2": {
          "known": 161521.0,
          "known_fraction": 0.69722097520547,
          "occurrences": 231664.0,
          "unknown": 70143.0,
          "unknown_fraction": 0.30277902479453
        },
        "r2_r3": {
          "known": 101512.0,
          "known_fraction": 0.43818633883555497,
          "occurrences": 231664.0,
          "unknown": 130152.0,
          "unknown_fraction": 0.5618136611644451
        }
      },
      "width": 12294,
      "width_per_schema": 4098
    },
    "dimension": 61592,
    "frequency_fit_graphs": 10000,
    "hierarchical": {
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
    },
    "molecule_composition": {
      "levels": {
        "r2_parent_bag": {
          "graphs": 10000.0,
          "known": 2051.0,
          "known_fraction": 0.2051,
          "unknown": 7949.0,
          "unknown_fraction": 0.7949
        },
        "r3_parent_bag": {
          "graphs": 10000.0,
          "known": 2051.0,
          "known_fraction": 0.2051,
          "unknown": 7949.0,
          "unknown_fraction": 0.7949
        }
      },
      "width": 4098
    },
    "rarity": {
      "rounds": {
        "0": {
          "backoff": 0.0,
          "backoff_fraction": 0.0,
          "backoff_previous": 0.0,
          "backoff_previous_fraction": 0.0,
          "exact_known": 231664.0,
          "exact_known_fraction": 1.0,
          "exact_unknown": 0.0,
          "exact_unknown_fraction": 0.0,
          "occurrences": 231664.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "1": {
          "backoff": 0.0,
          "backoff_fraction": 0.0,
          "backoff_previous": 0.0,
          "backoff_previous_fraction": 0.0,
          "exact_known": 231664.0,
          "exact_known_fraction": 1.0,
          "exact_unknown": 0.0,
          "exact_unknown_fraction": 0.0,
          "occurrences": 231664.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "2": {
          "backoff": 70143.0,
          "backoff_fraction": 0.30277902479453,
          "backoff_previous": 70143.0,
          "backoff_previous_fraction": 0.30277902479453,
          "exact_known": 161521.0,
          "exact_known_fraction": 0.69722097520547,
          "exact_unknown": 70143.0,
          "exact_unknown_fraction": 0.30277902479453,
          "occurrences": 231664.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "3": {
          "backoff": 130152.0,
          "backoff_fraction": 0.5618136611644451,
          "backoff_previous": 60315.0,
          "backoff_previous_fraction": 0.2603555148836246,
          "exact_known": 101512.0,
          "exact_known_fraction": 0.43818633883555497,
          "exact_unknown": 130152.0,
          "exact_unknown_fraction": 0.5618136611644451,
          "occurrences": 231664.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        }
      },
      "width": 54
    },
    "vocabulary_fit_graphs": 10000
  },
  "official_valid": {
    "cross_centre": {
      "known_fraction": 0.12414644831857873,
      "known_pairs": 32871.0,
      "total_pairs": 264776.0,
      "unknown_fraction": 0.8758535516814213,
      "unknown_pairs": 231905.0,
      "width": 4098
    },
    "cross_level": {
      "schemas": {
        "r0_r1_r2_r3": {
          "known": 9950.0,
          "known_fraction": 0.43105315600225275,
          "occurrences": 23083.0,
          "unknown": 13133.0,
          "unknown_fraction": 0.5689468439977473
        },
        "r1_r2": {
          "known": 16075.0,
          "known_fraction": 0.6963999480136898,
          "occurrences": 23083.0,
          "unknown": 7008.0,
          "unknown_fraction": 0.3036000519863103
        },
        "r2_r3": {
          "known": 9938.0,
          "known_fraction": 0.43053329289953646,
          "occurrences": 23083.0,
          "unknown": 13145.0,
          "unknown_fraction": 0.5694667071004635
        }
      },
      "width": 12294,
      "width_per_schema": 4098
    },
    "dimension": 61592,
    "frequency_fit_graphs": 10000,
    "hierarchical": {
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
    },
    "molecule_composition": {
      "levels": {
        "r2_parent_bag": {
          "graphs": 1000.0,
          "known": 1.0,
          "known_fraction": 0.001,
          "unknown": 999.0,
          "unknown_fraction": 0.999
        },
        "r3_parent_bag": {
          "graphs": 1000.0,
          "known": 1.0,
          "known_fraction": 0.001,
          "unknown": 999.0,
          "unknown_fraction": 0.999
        }
      },
      "width": 4098
    },
    "rarity": {
      "rounds": {
        "0": {
          "backoff": 0.0,
          "backoff_fraction": 0.0,
          "backoff_previous": 0.0,
          "backoff_previous_fraction": 0.0,
          "exact_known": 23083.0,
          "exact_known_fraction": 1.0,
          "exact_unknown": 0.0,
          "exact_unknown_fraction": 0.0,
          "occurrences": 23083.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "1": {
          "backoff": 39.0,
          "backoff_fraction": 0.0016895550838279252,
          "backoff_previous": 39.0,
          "backoff_previous_fraction": 0.0016895550838279252,
          "exact_known": 23044.0,
          "exact_known_fraction": 0.998310444916172,
          "exact_unknown": 39.0,
          "exact_unknown_fraction": 0.0016895550838279252,
          "occurrences": 23083.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "2": {
          "backoff": 7001.0,
          "backoff_fraction": 0.3032967985097258,
          "backoff_previous": 6962.0,
          "backoff_previous_fraction": 0.30160724342589784,
          "exact_known": 16082.0,
          "exact_known_fraction": 0.6967032014902742,
          "exact_unknown": 7001.0,
          "exact_unknown_fraction": 0.3032967985097258,
          "occurrences": 23083.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        },
        "3": {
          "backoff": 13138.0,
          "backoff_fraction": 0.569163453623879,
          "backoff_previous": 6166.0,
          "backoff_previous_fraction": 0.2671229909457176,
          "exact_known": 9945.0,
          "exact_known_fraction": 0.43083654637612095,
          "exact_unknown": 13138.0,
          "exact_unknown_fraction": 0.569163453623879,
          "occurrences": 23083.0,
          "unresolved": 0.0,
          "unresolved_fraction": 0.0
        }
      },
      "width": 54
    },
    "vocabulary_fit_graphs": 10000
  }
}
```

Runtime: `1453.0s`; script SHA-256: `c13f86dcbd1e2e26b5ddc63b2a163bbe67ce7252e732927527a036d76661087e`.
