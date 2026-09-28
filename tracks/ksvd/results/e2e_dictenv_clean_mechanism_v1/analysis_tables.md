## Stage A — BASE vs C6 (soup valid MAE, CPU matched)

| seed | BASE | C6 | C6-BASE | C1 | C1-BASE |
|---|---|---|---|---|---|
| 0 | +0.143298 | +0.128499 | -0.014799 | +0.133530 | -0.009768 |
| 1 | +0.130622 | +0.130505 | -0.000116 | — | — |
| 2 | +0.131494 | +0.122265 | -0.009229 | — | — |

C6 gate: **STRONG_SUPPORT** (adopted=True); mean=-0.008048 median=-0.009229 std=+0.006052 range=+0.014683

CLEAN_BASE = **C6**

## Stage B — core probe deltas by arm/seed (distribution-preserving)

| arm_seed | grap:GS1 | grap:GS4 | fill:EG2 | read:PS1 | read:PS2 | read:PS3 | read:PS4 | read:PS5 | read:PS6 | rela:RS1 | rela:RS2 | rela:RS3 | rela:RS4 | rela:RS5 | zero:N1 | zero:N2 | zero:N3 | zero:N4 | zero:N6 | zero:EB2 | zero:EB3 | zero:T1 | fill:A2 | fill:A4 | fill:A5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE_seed0 | +0.0550 | +0.2583 | +0.0023 | +1.3377 | +0.3662 | -0.0000 | +0.2731 | +0.0980 | +0.0023 | +0.0617 | +0.0471 | +0.3022 | +0.0016 | +0.1696 | +0.2253 | +0.2534 | +0.0273 | +0.0867 | +0.3238 | +0.0292 | +0.7339 | +0.1609 | +0.5808 | +0.4251 | +0.1819 |
| BASE_seed1 | +0.0788 | +0.2583 | +0.0058 | +1.2549 | +0.2662 | -0.0002 | +0.1683 | +0.0979 | +0.0004 | +0.0595 | +0.0695 | +0.1444 | +0.0014 | +0.1736 | +0.2688 | +0.2414 | +0.0106 | +0.0654 | +0.2721 | +0.0228 | +0.5379 | +0.1510 | +0.6115 | +0.4574 | +0.1771 |
| BASE_seed2 | +0.0778 | +0.2397 | +0.0029 | +1.2967 | +0.4192 | -0.0000 | +0.1749 | +0.1018 | +0.0036 | +0.0997 | +0.1204 | +0.0877 | +0.0236 | +0.1847 | +0.1298 | +0.3624 | +0.0034 | +0.0544 | +0.4176 | +0.0530 | +1.2450 | +0.1257 | +0.5457 | +0.3686 | +0.2417 |
| C6_seed0 | +0.0000 | +0.2674 | +0.0000 | +1.4517 | +0.3791 | +0.0000 | +0.2136 | +0.1148 | +0.0000 | +0.1190 | +0.1249 | +0.6614 | +0.0000 | +0.2359 | +0.1952 | +0.2464 | +0.0154 | +0.1052 | +0.2678 | +0.0547 | +0.4499 | +0.1691 | +0.6440 | +0.4848 | +0.2130 |
| C6_seed1 | +0.0000 | +0.2524 | +0.0000 | +1.4316 | +0.3512 | +0.0000 | +0.1992 | +0.1291 | +0.0000 | +0.0756 | +0.0722 | +0.1118 | +0.0000 | +0.1266 | +0.2349 | +0.2496 | +0.0174 | +0.0693 | +0.2646 | +0.0393 | +0.4568 | +0.1992 | +0.7530 | +0.5282 | +0.2349 |
| C6_seed2 | +0.0000 | +0.2489 | +0.0000 | +1.3604 | +0.4295 | +0.0000 | +0.3049 | +0.1228 | +0.0000 | +0.1501 | +0.2429 | +0.0867 | +0.0000 | +0.2777 | +0.1457 | +0.2891 | +0.0215 | +0.0637 | +0.3835 | +0.0273 | +1.0979 | +0.1338 | +0.5727 | +0.3528 | +0.2897 |

* BASE_seed0: spec={'coding': 'sparse', 'edge_binding': 'paired', 'mask_kind': 'BASE', 'node_binding': 'paired', 'tag': 'BASE'} dictionary: active=27/32 effective=14.12 nnz_per_row=8.00 top5_share=4.68 rec=+0.000 movement=5.465
* BASE_seed1: spec={'coding': 'sparse', 'edge_binding': 'paired', 'mask_kind': 'BASE', 'node_binding': 'paired', 'tag': 'BASE'} dictionary: active=28/32 effective=13.68 nnz_per_row=8.00 top5_share=4.98 rec=+0.000 movement=5.531
* BASE_seed2: spec={'coding': 'sparse', 'edge_binding': 'paired', 'mask_kind': 'BASE', 'node_binding': 'paired', 'tag': 'BASE'} dictionary: active=28/32 effective=14.66 nnz_per_row=8.00 top5_share=4.45 rec=+0.000 movement=5.743
* C6_seed0: spec={'coding': 'sparse', 'edge_binding': 'paired', 'mask_kind': 'C6', 'node_binding': 'paired', 'tag': 'C6'} dictionary: active=27/32 effective=14.52 nnz_per_row=8.00 top5_share=4.80 rec=+0.000 movement=5.596
* C6_seed1: spec={'coding': 'sparse', 'edge_binding': 'paired', 'mask_kind': 'C6', 'node_binding': 'paired', 'tag': 'C6'} dictionary: active=27/32 effective=14.80 nnz_per_row=8.00 top5_share=4.56 rec=+0.000 movement=5.418
* C6_seed2: spec={'coding': 'sparse', 'edge_binding': 'paired', 'mask_kind': 'C6', 'node_binding': 'paired', 'tag': 'C6'} dictionary: active=28/32 effective=15.35 nnz_per_row=8.00 top5_share=4.41 rec=+0.000 movement=5.571

## Stage C1 — frozen binding-replacement diagnostics

| checkpoint | baseline | NODE_INDEP Δ | EDGE_INDEP Δ | BOTH Δ | corr(node) | corr(edge) |
|---|---|---|---|---|---|---|
| BASE_seed0 | 0.143298 | +0.012425 | +0.036520 | +0.049033 | 0.9993 | 0.9979 |
| BASE_seed1 | 0.130622 | +0.004738 | +0.040793 | +0.041375 | 0.9997 | 0.9979 |
| BASE_seed2 | 0.131494 | +0.002028 | +0.026314 | +0.028077 | 0.9999 | 0.9988 |
| C6_seed0 | 0.128499 | +0.006770 | +0.057069 | +0.064908 | 0.9997 | 0.9967 |
| C6_seed1 | 0.130505 | +0.009250 | +0.030772 | +0.037205 | 0.9996 | 0.9983 |
| C6_seed2 | 0.122265 | +0.011962 | +0.032379 | +0.049795 | 0.9996 | 0.9984 |

### per-shell node residual (ratio = ||R||/||U_pair||)

| checkpoint | shell | n_slots | n_multi | ratio_mean | ratio_median | p90 | p95 | zero_frac |
|---|---|---|---|---|---|---|---|---|
| BASE_seed0 | shell0 | 23083 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed0 | shell1 | 23083 | 19012 | 0.1229 | 0.0000 | 0.4040 | 0.5607 | 0.1910 |
| BASE_seed0 | shell2 | 23083 | 22079 | 0.2215 | 0.1382 | 0.6097 | 0.7540 | 0.0455 |
| BASE_seed1 | shell0 | 23083 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed1 | shell1 | 23083 | 19012 | 0.1065 | 0.0000 | 0.3626 | 0.5089 | 0.1908 |
| BASE_seed1 | shell2 | 23083 | 22079 | 0.1928 | 0.1214 | 0.5328 | 0.6518 | 0.0462 |
| BASE_seed2 | shell0 | 23083 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed2 | shell1 | 23083 | 19012 | 0.0751 | 0.0000 | 0.2500 | 0.3587 | 0.1913 |
| BASE_seed2 | shell2 | 23083 | 22079 | 0.1407 | 0.0767 | 0.4224 | 0.5294 | 0.0465 |
| C6_seed0 | shell0 | 23083 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed0 | shell1 | 23083 | 19012 | 0.1078 | 0.0000 | 0.3873 | 0.4900 | 0.1911 |
| C6_seed0 | shell2 | 23083 | 22079 | 0.1635 | 0.0996 | 0.4525 | 0.5732 | 0.0461 |
| C6_seed1 | shell0 | 23083 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed1 | shell1 | 23083 | 19012 | 0.1280 | 0.0000 | 0.4201 | 0.6070 | 0.1894 |
| C6_seed1 | shell2 | 23083 | 22079 | 0.2223 | 0.1509 | 0.5991 | 0.7341 | 0.0458 |
| C6_seed2 | shell0 | 23083 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed2 | shell1 | 23083 | 19012 | 0.1238 | 0.0000 | 0.4017 | 0.5353 | 0.2020 |
| C6_seed2 | shell2 | 23083 | 22079 | 0.2091 | 0.1796 | 0.5349 | 0.6653 | 0.0456 |

### per-shellpair edge residual

| checkpoint | shellpair | n_multi | ratio_mean | ratio_median | p90 | p95 | zero_frac |
|---|---|---|---|---|---|---|---|
| BASE_seed0 | shellpair0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed0 | shellpair1 | 19012 | 0.1799 | 0.0000 | 0.2916 | 1.4811 | 0.1852 |
| BASE_seed0 | shellpair2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed0 | shellpair3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed0 | shellpair4 | 22082 | 0.2380 | 0.0642 | 0.5263 | 1.4568 | 0.0460 |
| BASE_seed0 | shellpair5 | 70 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.9971 |
| BASE_seed1 | shellpair0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed1 | shellpair1 | 19012 | 0.1527 | 0.0000 | 0.2618 | 1.3563 | 0.1846 |
| BASE_seed1 | shellpair2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed1 | shellpair3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed1 | shellpair4 | 22082 | 0.2194 | 0.0733 | 0.5321 | 1.3706 | 0.0464 |
| BASE_seed1 | shellpair5 | 70 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.9971 |
| BASE_seed2 | shellpair0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed2 | shellpair1 | 19012 | 0.1577 | 0.0000 | 0.4795 | 1.0795 | 0.1847 |
| BASE_seed2 | shellpair2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed2 | shellpair3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| BASE_seed2 | shellpair4 | 22082 | 0.2709 | 0.1453 | 0.6837 | 1.1915 | 0.0488 |
| BASE_seed2 | shellpair5 | 70 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.9971 |
| C6_seed0 | shellpair0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed0 | shellpair1 | 19012 | 0.1759 | 0.0000 | 0.3445 | 1.1102 | 0.1906 |
| C6_seed0 | shellpair2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed0 | shellpair3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed0 | shellpair4 | 22082 | 0.2808 | 0.1252 | 0.6130 | 1.5819 | 0.0468 |
| C6_seed0 | shellpair5 | 70 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.9971 |
| C6_seed1 | shellpair0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed1 | shellpair1 | 19012 | 0.1847 | 0.0000 | 0.2733 | 1.1010 | 0.1856 |
| C6_seed1 | shellpair2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed1 | shellpair3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed1 | shellpair4 | 22082 | 0.2468 | 0.0835 | 0.5165 | 1.5191 | 0.0571 |
| C6_seed1 | shellpair5 | 70 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9971 |
| C6_seed2 | shellpair0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed2 | shellpair1 | 19012 | 0.1003 | 0.0000 | 0.2706 | 0.4888 | 0.1863 |
| C6_seed2 | shellpair2 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed2 | shellpair3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| C6_seed2 | shellpair4 | 22082 | 0.1739 | 0.0886 | 0.4764 | 0.7900 | 0.0515 |
| C6_seed2 | shellpair5 | 70 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.9971 |

## Stage C2 — 20-epoch warm-start adaptation (vs matched continuation)

* continuation control (C6 mask): +0.128782
* node: soup=+0.131394 delta_vs_control=+0.002612 best_epoch=18
* edge: soup=+0.137107 delta_vs_control=+0.008326 best_epoch=16

## Stage C2 — from-scratch independence gates

* node: verdict=EXTEND_SEEDS per_seed={'0': 8.1e-05, '1': 0.003684, '2': 0.009251} adaptation=0.002612
* edge: verdict=EXTEND_SEEDS per_seed={'0': 0.009005, '1': 0.004711, '2': 0.008946} adaptation=0.008326

## Stage D — relation simplification

* provenance all_passed=True
* 20-epoch screen control=+0.128782
  * REL-DIST: soup=+0.143664 delta_vs_control=+0.014882 best_epoch=11
  * REL-DIST-BOUNDARY: soup=+0.133965 delta_vs_control=+0.005183 best_epoch=15
* candidate: **REL-DIST** (REL-DIST survived the 20-epoch screen (delta=+0.01488))
* REL-DIST: gate=INCONCLUSIVE_KEEP_FULL raw={'delta': 0.005660657159052784, 'thresholds': {'extend_max': 0.005, 'stop_min': 0.015}, 'verdict': 'INCONCLUSIVE_KEEP_FULL'}
* REL-DIST-BOUNDARY: gate=NO_DATA raw={'verdict': 'NO_DATA'}

## Stage F — dictionary specificity

* adopted: {'clean_base': 'C6', 'edge_indep': False, 'node_indep': False, 'relation': None}
* final spec: {'coding': 'sparse', 'edge_binding': 'paired', 'mask_kind': 'C6', 'node_binding': 'paired', 'tag': 'FINAL-CLEAN'}
* sparse reference: {'json': 'tracks/ksvd/results/e2e_dictenv_h1_clarity_audit/matched_cpu/C6_e320.json', 'soup_valid_mae': 0.12849851670576026, 'source': 'reused_clarity_audit'}
  * seed 0: sparse=+0.128499 dense=+0.125563 G_dict=-0.002936
* gate: {'delta_dense_minus_sparse': -0.002935989641584452, 'threshold': 0.003, 'verdict': 'SPECIFICITY_NOT_ESTABLISHED'}

