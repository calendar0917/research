# cscl-task-adaptive-structure-units-v1 — task-adaptive molecular structure units

- git commit: `09cbfb9a1cd4a6b9282a8d53ca15dffe3994e6ba`
- decision: **STRUCTURE_UNITS_READOUT_BOTTLENECK**
- Monitor labels read exactly once, after prediction freeze; locked side untouched.

## Reading

- The learned partition is **real and non-degenerate** (5.69 units/molecule, 46% of
  bridges merged, 0.7% all-merge / 1.0% all-cut on monitor), but it **hurts** raw-y
  MAE under both readouts: L-A +0.0102 vs F-A and L-O +0.0089 vs F-O (both 95% CIs
  exclude 0).  Task loss alone does not find boundaries that help this target.
- The **additive decomposition is the dominant bottleneck**: capacity-matched opaque
  readers improve by 0.042-0.043 MAE at the same partition (F-O 0.1232 vs F-A 0.1649;
  L-O 0.1320 vs L-A 0.1751).  Explicit additive structure costs ~0.04 MAE here.
- Even the best arm (F-O 0.1232) is worse than the **BJOC-Null** reference (0.1043),
  so this route does not beat the existing backbone on split D.
- Unit reuse is high (78.3% chemical type reuse, 231 types across molecules, all
  3,750 top-type pairs exact-isomorphic) and H'/alpha are exactly context-free under
  identical model inputs (strict-feature max diff 7e-7).  Chemically identical units
  differ only through context-bearing input features (mean repr diff 0.91, 19% within
  1e-4): context-free unit identity is an input-feature question, not a pooling bug.

## Execution regime and budget

- Host `res` (node a100-2, 2x A100-40GB, driver 550.163.01, torch 2.5.1+cu124);
  training commit `2d2d8e7`, audit-fix commit `09cbfb9` (post-freeze amendment
  records that only label-blind audit functions changed).
- 8 training runs (4 arms x 2 seeds), 7,191 s wall total (2.00 GPU-h; frozen cap
  2.5 h).
- Monitor read exactly once (lock file); locked split untouched; official valid and
  test never loaded.

## Answers to the five protocol questions

1. **Can boundaries be learned from task loss?** CPU S1 direct supervision reaches 0.914 held-out bridge accuracy; S2 y-only: L-A 2.069 vs F-A 2.174 (oracle 1.561, all-merge 1.998). Monitor: L-A 0.175123 vs F-A 0.164934.
2. **Connected / non-duplicated / reusable?** Units are merged base blocks of the block tree, so connectedness is structural; partition degeneracy on monitor: all-merge=0.007, all-cut=0.010; type reuse=0.783 over 1233 chemical types (2631 strict-feature types).
3. **Does learned partition beat fixed?** H1 gate FAIL; L-A-F-A delta +0.010189 (threshold -0.01).
4. **Cost of explicit additive decomposition vs opaque readout?** L-O 0.132030 vs L-A 0.175123; F-O 0.123183 vs F-A 0.164934; capacity matched within 0.04%.
5. **Worth pursuing vs BJOC-Null?** BJOC-Null reference monitor MAE 0.104301; best arm here 0.123183.

## Monitor arms (raw-y MAE, seeds 0, 1)

| arm | seed 0 | seed 1 | seed mean |
| --- | --- | --- | --- |
| F-A | 0.159624 | 0.170243 | 0.164934 |
| L-A | 0.173509 | 0.176737 | 0.175123 |
| F-O | 0.121789 | 0.124576 | 0.123183 |
| L-O | 0.132393 | 0.131668 | 0.132030 |

## Label-blind audits before monitor read

```json
{
 "partition_learned_degenerate": false,
 "partition_monitor": {
  "F-A": {
   "n_molecules": 1000,
   "mean_units": 10.728,
   "mean_bridges": 9.728,
   "mean_merge_fraction": 0.0,
   "all_merge_fraction": 0.0,
   "all_cut_fraction": 1.0,
   "unit_size_histogram": {
    "1": 8411,
    "3": 59,
    "4": 14,
    "5": 576,
    "6": 1268,
    "7": 29,
    "8": 16,
    "9": 192,
    "10": 114,
    "11": 12,
    "12": 5,
    "13": 22,
    "14": 5,
    "15": 3,
    "16": 1,
    "22": 1
   },
   "examples_all_merge": [],
   "examples_all_cut": [
    "",
    "",
    "",
    "",
    ""
   ]
  },
  "L-A": {
   "n_molecules": 1000,
   "mean_units": 5.687,
   "mean_bridges": 9.728,
   "mean_merge_fraction": 0.46234686000274233,
   "all_merge_fraction": 0.007,
   "all_cut_fraction": 0.01,
   "unit_size_histogram": {
    "1": 2681,
    "2": 654,
    "3": 232,
    "4": 104,
    "5": 175,
    "6": 474,
    "7": 324,
    "8": 213,
    "9": 193,
    "10": 152,
    "11": 116,
    "12": 78,
    "13": 52,
    "14": 60,
    "15": 44,
    "16": 30,
    "17": 24,
    "18": 23,
    "19": 22,
    "20": 10,
    "21": 9,
    "22": 3,
    "23": 3,
    "24": 3,
    "25": 4,
    "26": 1,
    "27": 2,
    "29": 1
   },
   "examples_all_merge": [
    "",
    "",
    "",
    "",
    ""
   ],
   "examples_all_cut": [
    "",
    "",
    "",
    "",
    ""
   ]
  }
 },
 "isomorphism_consistency": {
  "strict_feature_identity": {
   "mode": "strict",
   "n_units": 5687,
   "n_wl_buckets": 2631,
   "n_wl_candidate_pairs": 3056,
   "n_exact_iso_pairs": 3056,
   "max_repr_abs_diff": 7.152557373046875e-07,
   "mean_repr_abs_diff": 7.228721030719617e-09,
   "p95_repr_abs_diff": 0.0,
   "frac_repr_diff_le_1e-4": 1.0,
   "max_alpha_abs_diff": 1.7881393432617188e-07,
   "within_float_tolerance": true
  },
  "chemical_isomorphism": {
   "mode": "chemical",
   "n_units": 5687,
   "n_wl_buckets": 1233,
   "n_wl_candidate_pairs": 4454,
   "n_exact_iso_pairs": 4454,
   "max_repr_abs_diff": 8.455907821655273,
   "mean_repr_abs_diff": 0.9146748196692145,
   "p95_repr_abs_diff": 2.3135403990745544,
   "frac_repr_diff_le_1e-4": 0.19375841939829366,
   "max_alpha_abs_diff": 1.2781941294670105,
   "within_float_tolerance": false
  }
 },
 "sharing_monitor": {
  "n_units": 5687,
  "chemical": {
   "n_types": 1233,
   "reuse_rate": 0.7831897309653596,
   "types_reused_across_molecules": 231,
   "max_type_count": 1275,
   "mean_type_count": 4.612327656123276,
   "top_types": [
    {
     "hash": "b1e544fb299c88b69dd7ad6544be908a",
     "count": 1275,
     "molecules": 648,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "12e0ad3d1154942e3d4ad6df4886f5c0",
     "count": 923,
     "molecules": 642,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "8ce19713a1042e9bf53dad5b18d4f718",
     "count": 258,
     "molecules": 240,
     "unit_atoms": 6,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "bfd086bc793f9578740f58415e2f8a88",
     "count": 228,
     "molecules": 195,
     "unit_atoms": 2,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "d7daa36237354813e49a8ca622bee93d",
     "count": 194,
     "molecules": 168,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "9a609b4b04af4cde02442d2fd7e1c294",
     "count": 166,
     "molecules": 160,
     "unit_atoms": 2,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "b3dd8b6d00b9dbe31f49177d3e3ab492",
     "count": 90,
     "molecules": 52,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "908de23b1d2ed2ffafbff7c8736ac317",
     "count": 86,
     "molecules": 84,
     "unit_atoms": 2,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "f155b07dd38940986230babd9b72f52c",
     "count": 75,
     "molecules": 73,
     "unit_atoms": 2,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "f1fac53528b2cef83134ad4c2a70b4dd",
     "count": 68,
     "molecules": 66,
     "unit_atoms": 7,
     "examples": [
      "",
      "",
      ""
     ]
    }
   ],
   "top_type_exact_iso_checked": 3750,
   "top_type_exact_iso_ok": 3750,
   "alpha_within_type_mean_std": 0.14219453473194446
  },
  "strict_feature": {
   "n_types": 2631,
   "reuse_rate": 0.5373659222788816,
   "types_reused_across_molecules": 472,
   "max_type_count": 588,
   "mean_type_count": 2.16153553781832,
   "top_types": [
    {
     "hash": "2c2cf7633c0d0d812cce744c9850b834",
     "count": 588,
     "molecules": 467,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "b980c7b186878cd6d490485b6d6ae7e5",
     "count": 132,
     "molecules": 106,
     "unit_atoms": 2,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "70f8fcbf95f4840912324cd7941ebc5e",
     "count": 85,
     "molecules": 84,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "897ed5a60cb66bcaaca4213faeb1883f",
     "count": 68,
     "molecules": 46,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "b27ad75d5304b2f0b1a6ce225fe665df",
     "count": 63,
     "molecules": 53,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "a3d31c0380112c3ce76fcbbea452a1d5",
     "count": 53,
     "molecules": 21,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "e03433a0871e0bdcef558f0da63f674d",
     "count": 49,
     "molecules": 20,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "76bff57b1d196262e72d3308818e2196",
     "count": 44,
     "molecules": 41,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "9fce92c757745cf26df13ba8a41b1c63",
     "count": 42,
     "molecules": 40,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    },
    {
     "hash": "9926bd78871c21ab7b1867fa92128e1c",
     "count": 42,
     "molecules": 40,
     "unit_atoms": 1,
     "examples": [
      "",
      "",
      ""
     ]
    }
   ],
   "top_type_exact_iso_checked": 1479,
   "top_type_exact_iso_ok": 1479,
   "alpha_within_type_mean_std": 4.963520430386701e-10
  }
 },
 "decomposition_identity": {
  "F-A": {
   "decomposition_max_abs_err": 1.0523945093154907e-06,
   "decomposition_ok": true,
   "sum_abs_alpha": 2937.0084838867188,
   "sum_abs_gamma": 945.5065383911133,
   "relation_absolute_share": 0.24352939601413165
  },
  "L-A": {
   "decomposition_max_abs_err": 5.373731255531311e-07,
   "decomposition_ok": true,
   "sum_abs_alpha": 2479.435821533203,
   "sum_abs_gamma": 704.8761367797852,
   "relation_absolute_share": 0.2213590081648974
  }
 }
}
```

## Synthetic known-truth evidence

```json
{
 "s1": {
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
 },
 "s2": {
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
}
```

## ZINC block applicability

```json
{
 "protocol": "cscl-task-adaptive-structure-units-v1",
 "git_commit": "fdbd0b2d2c67a978739a4f06b57839de8b20cc84",
 "zones": {
  "fit": {
   "n_molecules": 8000,
   "n_blocks": {
    "mean": 10.850875,
    "median": 11.0,
    "p90": 14.0,
    "max": 20
   },
   "n_bridges": {
    "mean": 9.850875,
    "median": 10.0,
    "p90": 13.0,
    "max": 19,
    "total": 78807
   },
   "single_block_fraction": 0.000125,
   "frac_molecules_with_boundary": 0.999875,
   "frac_molecules_with_2plus_boundaries": 0.999,
   "block_size_histogram": {
    "1": 68169,
    "2": 0,
    "3": 496,
    "4": 103,
    "5": 4563,
    "6": 10303,
    "7": 166,
    "8": 135,
    "9": 1488,
    "10": 976,
    "11": 84,
    "12": 51,
    "13": 170,
    "14": 39,
    "15": 21,
    "16": 15,
    "17": 15,
    "18": 8,
    "19": 2,
    "20": 1,
    "21": 1,
    "22": 0,
    "23": 0,
    "24": 0,
    "25": 0,
    "26": 1
   },
   "frac_singleton_blocks": 0.7852938127109564,
   "frac_ring_blocks": 0.2147061872890435,
   "max_path_length": 13,
   "mean_pair_entries_per_molecule": 398.59975,
   "max_pair_entries_per_molecule": 2016,
   "ring_bridge_overlap": 0,
   "bridge_not_raw_bond": 0,
   "ring_bond_fraction": 0.6050407954613797
  },
  "monitor": {
   "n_molecules": 1000,
   "n_blocks": {
    "mean": 10.728,
    "median": 11.0,
    "p90": 14.0,
    "max": 18
   },
   "n_bridges": {
    "mean": 9.728,
    "median": 10.0,
    "p90": 13.0,
    "max": 17,
    "total": 9728
   },
   "single_block_fraction": 0.0,
   "frac_molecules_with_boundary": 1.0,
   "frac_molecules_with_2plus_boundaries": 1.0,
   "block_size_histogram": {
    "1": 8411,
    "2": 0,
    "3": 59,
    "4": 14,
    "5": 576,
    "6": 1268,
    "7": 29,
    "8": 16,
    "9": 192,
    "10": 114,
    "11": 12,
    "12": 5,
    "13": 22,
    "14": 5,
    "15": 3,
    "16": 1,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 0,
    "21": 0,
    "22": 1
   },
   "frac_singleton_blocks": 0.7840231170768084,
   "frac_ring_blocks": 0.21597688292319164,
   "max_path_length": 11,
   "mean_pair_entries_per_molecule": 387.04,
   "max_pair_entries_per_molecule": 1508,
   "ring_bridge_overlap": 0,
   "bridge_not_raw_bond": 0,
   "ring_bond_fraction": 0.6061857339486681
  }
 },
 "applicability_ok": true,
 "criterion": "fit zone: >=50% molecules have a bridge candidate and no ring bond is a bridge"
}
```
