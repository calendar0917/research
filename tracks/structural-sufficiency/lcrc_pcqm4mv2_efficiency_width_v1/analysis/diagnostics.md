# Read-only checkpoint diagnostics

## W1

checkpoint `tracks/structural-sufficiency/lcrc_pcqm4mv2_efficiency_width_v1/runs/PCQM_W1_1M/checkpoint_best_valid.pt` (SHA256 `d33a6a6b032405e3…`), epoch 36, 259,831 parameters, full Validation MAE 0.143313988 eV (official 0.143313988, NumPy 0.143313988)

**Official-Train exact-SMILES overlap:** 5,375 / 73,545 (7.308%). MAE duplicate 0.138238 eV (n=5,375), non-duplicate 0.143714 eV (n=68,170).

Bootstrap MAE difference (duplicate − non-duplicate) -0.005477 eV, 95% CI [-0.009751, -0.000404].

1M-subset overlap 1,901 (2.585%); MAE dup 0.138346 vs non-dup 0.143446 eV.

| atom-count bin | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| 0–20 | 70,269 | 0.140172 | 0.041058 |
| 21–25 | 2,593 | 0.199767 | 0.058649 |
| 26–30 | 553 | 0.233419 | 0.080802 |
| 31–35 | 92 | 0.315195 | 0.137282 |
| 36–40 | 14 | 0.395396 | -0.160959 |
| 41–50 | 23 | 0.358679 | -0.191206 |
| 51–+ | 1 | 0.389088 | -0.389088 |

| true-gap bin (eV) | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| [0.0, 4.5) | 16,066 | 0.176820 | 0.084447 |
| [4.5, 5.0) | 12,311 | 0.129700 | 0.032394 |
| [5.0, 5.5) | 13,354 | 0.123858 | 0.028366 |
| [5.5, 6.0) | 11,761 | 0.119336 | 0.021608 |
| [6.0, 100.0) | 20,053 | 0.151847 | 0.034857 |

UNK-containing molecules: 230 (0.313%); MAE with UNK 0.736478 eV, without 0.141453 eV.

## W2

checkpoint `tracks/structural-sufficiency/lcrc_pcqm4mv2_transfer_pilot_v1/runs/PCQM_W2_1M/checkpoint_best_valid.pt` (SHA256 `e45316c29d9152ad…`), epoch 40, 939,079 parameters, full Validation MAE 0.132310906 eV (official 0.132310906, NumPy 0.132310906)

**Official-Train exact-SMILES overlap:** 5,375 / 73,545 (7.308%). MAE duplicate 0.124882 eV (n=5,375), non-duplicate 0.132897 eV (n=68,170).

Bootstrap MAE difference (duplicate − non-duplicate) -0.008014 eV, 95% CI [-0.012730, -0.003101].

1M-subset overlap 1,901 (2.585%); MAE dup 0.119866 vs non-dup 0.132641 eV.

| atom-count bin | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| 0–20 | 70,269 | 0.128266 | 0.033526 |
| 21–25 | 2,593 | 0.198150 | 0.083075 |
| 26–30 | 553 | 0.270314 | 0.180194 |
| 31–35 | 92 | 0.420775 | 0.372496 |
| 36–40 | 14 | 0.503994 | 0.316672 |
| 41–50 | 23 | 0.355594 | 0.247982 |
| 51–+ | 1 | 0.421446 | 0.421446 |

| true-gap bin (eV) | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| [0.0, 4.5) | 16,066 | 0.161779 | 0.087442 |
| [4.5, 5.0) | 12,311 | 0.117667 | 0.030708 |
| [5.0, 5.5) | 13,354 | 0.113221 | 0.025292 |
| [5.5, 6.0) | 11,761 | 0.109716 | 0.017330 |
| [6.0, 100.0) | 20,053 | 0.143657 | 0.019513 |

UNK-containing molecules: 230 (0.313%); MAE with UNK 0.651691 eV, without 0.130682 eV.

## W3

checkpoint `tracks/structural-sufficiency/lcrc_pcqm4mv2_efficiency_width_v1/runs/PCQM_W3_1M/checkpoint_best_valid.pt` (SHA256 `f4175f141702b9cc…`), epoch 39, 2,038,167 parameters, full Validation MAE 0.127809174 eV (official 0.127809174, NumPy 0.127809174)

**Official-Train exact-SMILES overlap:** 5,375 / 73,545 (7.308%). MAE duplicate 0.120822 eV (n=5,375), non-duplicate 0.128360 eV (n=68,170).

Bootstrap MAE difference (duplicate − non-duplicate) -0.007538 eV, 95% CI [-0.011671, -0.002762].

1M-subset overlap 1,901 (2.585%); MAE dup 0.115713 vs non-dup 0.128130 eV.

| atom-count bin | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| 0–20 | 70,269 | 0.123943 | 0.023437 |
| 21–25 | 2,593 | 0.196309 | 0.061184 |
| 26–30 | 553 | 0.244280 | 0.125574 |
| 31–35 | 92 | 0.335687 | 0.252079 |
| 36–40 | 14 | 0.472251 | 0.213905 |
| 41–50 | 23 | 0.348276 | 0.113228 |
| 51–+ | 1 | 0.731850 | 0.731850 |

| true-gap bin (eV) | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| [0.0, 4.5) | 16,066 | 0.150972 | 0.056256 |
| [4.5, 5.0) | 12,311 | 0.113103 | 0.016941 |
| [5.0, 5.5) | 13,354 | 0.110139 | 0.018739 |
| [5.5, 6.0) | 11,761 | 0.107245 | 0.013691 |
| [6.0, 100.0) | 20,053 | 0.142108 | 0.018993 |

UNK-containing molecules: 230 (0.313%); MAE with UNK 0.604699 eV, without 0.126313 eV.

