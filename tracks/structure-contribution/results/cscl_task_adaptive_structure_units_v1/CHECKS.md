# cscl-task-adaptive-structure-units-v1 CPU feasibility gates

- git commit: `fdbd0b2d2c67a978739a4f06b57839de8b20cc84`
- overall: **PASS**

| check | result |
| --- | --- |
| structural_correctness | PASS |
| init_parity_all_arms | PASS |
| additive_decomposition | PASS |
| fixed_arm_zero_gates | PASS |
| scorer_init_cuts | PASS |
| invariance | PASS |
| scorer_receives_gradient | PASS |
| parameters_exact | PASS |
| no_forbidden_modules | PASS |
| opaque_capacity_matched | PASS |
| synthetic_s1 | PASS |
| synthetic_s2 | PASS |
| smoke_reproducible | PASS |

## structural correctness

```json
{
 "n_molecules": 200,
 "bridge_set_mismatches": 0,
 "ring_bridge_overlap": 0,
 "tree_edge_ok": 200,
 "block_partition_ok": 200,
 "path_product_ok": 200,
 "all_ok": true
}
```

## synthetic S1 (explicit boundary supervision)

```json
{
 "epochs": 120,
 "final_train": {
  "epoch": 120,
  "bce": 0.007467863962675134,
  "bridge_accuracy": 0.9965259683863124,
  "exact_partition_accuracy": 0.9866666666666667,
  "mean_units": 3.6206666666666667
 },
 "test_stats": {
  "bridge_accuracy": 0.9136125654450262,
  "exact_partition_accuracy": 0.732,
  "mean_units": 3.536
 },
 "test_bridge_accuracy_min": 0.85,
 "test_exact_partition_min": 0.5,
 "passed": true
}
```

## synthetic S2 (y-only, known truth)

```json
{
 "F-A": {
  "per_seed": {
   "0": {
    "test_mae": 2.3528159538682374,
    "train_mae_final": 1.276170682589213,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   },
   "1": {
    "test_mae": 2.084046837262598,
    "train_mae_final": 1.4262601874669394,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   },
   "2": {
    "test_mae": 2.084195522070906,
    "train_mae_final": 1.3635943851470946,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   }
  },
  "test_mae_seed_mean": 2.17368610440058,
  "test_mae_seed_std": 0.12666394581601118,
  "mean_units_seed_mean": 4.864
 },
 "L-A": {
  "per_seed": {
   "0": {
    "test_mae": 2.1261730439798443,
    "train_mae_final": 1.7536303523381551,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.3027950310559006,
     "exact_partition_accuracy": 0.0,
     "mean_units": 1.0
    }
   },
   "1": {
    "test_mae": 2.245112064982438,
    "train_mae_final": 2.053718915939331,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.3027950310559006,
     "exact_partition_accuracy": 0.0,
     "mean_units": 1.0
    }
   },
   "2": {
    "test_mae": 1.8345706847629166,
    "train_mae_final": 1.7474898980458577,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6780538302277432,
     "exact_partition_accuracy": 0.038,
     "mean_units": 4.654
    }
   }
  },
  "test_mae_seed_mean": 2.0686185979083995,
  "test_mae_seed_std": 0.17247307406293985,
  "mean_units_seed_mean": 2.218
 },
 "M-A": {
  "per_seed": {
   "0": {
    "test_mae": 1.9017993978179315,
    "train_mae_final": 1.483360079129537,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   },
   "1": {
    "test_mae": 2.117121526747106,
    "train_mae_final": 1.6654729525248209,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   },
   "2": {
    "test_mae": 1.9753457756337012,
    "train_mae_final": 1.5864909648895265,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   }
  },
  "test_mae_seed_mean": 1.9980889000662463,
  "test_mae_seed_std": 0.08936383341286593,
  "mean_units_seed_mean": 4.864
 },
 "O-A": {
  "per_seed": {
   "0": {
    "test_mae": 1.5467550502136664,
    "train_mae_final": 1.369782192548116,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   },
   "1": {
    "test_mae": 1.6013202004437275,
    "train_mae_final": 1.3438506708145141,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   },
   "2": {
    "test_mae": 1.533590732358399,
    "train_mae_final": 1.3348122574488321,
    "scorer_partition_stats": {
     "bridge_accuracy": 0.6972049689440993,
     "exact_partition_accuracy": 0.0,
     "mean_units": 4.864
    }
   }
  },
  "test_mae_seed_mean": 1.560555327671931,
  "test_mae_seed_std": 0.029321845726627258,
  "mean_units_seed_mean": 4.864
 },
 "seeds": [
  0,
  1,
  2
 ],
 "oracle_units_mean": 3.6453333333333333,
 "oracle_merge_fraction": 0.3313111111111111,
 "grouping_helps": true,
 "learned_uses_grouping": true,
 "pass_criterion": "3-seed mean test MAE: O-A < F-A (task rewards grouping) and L-A < F-A (learned uses grouping)",
 "passed": true,
 "oracle_headroom": 0.6131307767286491,
 "learned_vs_all_merge": 0.07052969784215324,
 "learned_overtake_of_all_merge": false,
 "learned_partition_degenerate_seeds": [
  0,
  1
 ],
 "recovery_oracle_partition": {
  "partition": "oracle",
  "n_units": 1847,
  "n_relations": 1347,
  "alpha_vs_oracle_unit_term": {
   "pearson": 0.6359553616475679,
   "spearman": 0.8071868632522758,
   "n": 1847
  },
  "gamma_vs_oracle_relation_term": {
   "pearson": 0.531736309952948,
   "spearman": 0.2615243955364891,
   "n": 1347
  }
 },
 "recovery_learned_partition": {
  "partition": "learned",
  "n_units": 1847,
  "n_relations": 1347,
  "alpha_vs_oracle_unit_term": {
   "pearson": 0.35758575551904626,
   "spearman": 0.14973793107157454,
   "n": 1847
  },
  "gamma_vs_oracle_relation_term": {
   "pearson": null,
   "spearman": null,
   "n": 1347
  }
 }
}
```
