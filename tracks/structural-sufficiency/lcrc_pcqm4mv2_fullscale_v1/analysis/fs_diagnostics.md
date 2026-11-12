# Read-only checkpoint diagnostics

## W1

not available

## W2

checkpoint `tracks/structural-sufficiency/lcrc_pcqm4mv2_fullscale_v1/runs/PCQM_W2_FULL/checkpoint_best_valid.pt` (SHA256 `000f2b61edc3b586…`), epoch 18, 939,079 parameters, full Validation MAE 0.128821302 eV (official 0.128821302, NumPy 0.128821302)

**Official-Train exact-SMILES overlap:** 5,375 / 73,545 (7.308%). MAE duplicate 0.125832 eV (n=5,375), non-duplicate 0.129057 eV (n=68,170).

Bootstrap MAE difference (duplicate − non-duplicate) -0.003225 eV, 95% CI [-0.007474, +0.001238].

1M-subset overlap 1,901 (2.585%); MAE dup 0.129024 vs non-dup 0.128816 eV.

| atom-count bin | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| 0–20 | 70,269 | 0.125734 | 0.039129 |
| 21–25 | 2,593 | 0.184913 | 0.023103 |
| 26–30 | 553 | 0.216837 | 0.000939 |
| 31–35 | 92 | 0.244428 | -0.028017 |
| 36–40 | 14 | 0.397411 | -0.175355 |
| 41–50 | 23 | 0.458419 | -0.404379 |
| 51–+ | 1 | 0.981940 | -0.981940 |

| true-gap bin (eV) | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| [0.0, 4.5) | 16,066 | 0.156527 | 0.090123 |
| [4.5, 5.0) | 12,311 | 0.118643 | 0.044125 |
| [5.0, 5.5) | 13,354 | 0.111801 | 0.033246 |
| [5.5, 6.0) | 11,761 | 0.106347 | 0.011108 |
| [6.0, 100.0) | 20,053 | 0.137388 | 0.011417 |

UNK-containing molecules: 230 (0.313%); MAE with UNK 0.592918 eV, without 0.127365 eV.

## W3

checkpoint `tracks/structural-sufficiency/lcrc_pcqm4mv2_fullscale_v1/runs/PCQM_W3_FULL/checkpoint_best_valid.pt` (SHA256 `786f3a5d745599b7…`), epoch 35, 2,038,167 parameters, full Validation MAE 0.118775495 eV (official 0.118775495, NumPy 0.118775495)

**Official-Train exact-SMILES overlap:** 5,375 / 73,545 (7.308%). MAE duplicate 0.111947 eV (n=5,375), non-duplicate 0.119314 eV (n=68,170).

Bootstrap MAE difference (duplicate − non-duplicate) -0.007367 eV, 95% CI [-0.011537, -0.002721].

1M-subset overlap 1,901 (2.585%); MAE dup 0.114653 vs non-dup 0.118885 eV.

| atom-count bin | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| 0–20 | 70,269 | 0.114492 | 0.039125 |
| 21–25 | 2,593 | 0.186428 | 0.042647 |
| 26–30 | 553 | 0.267794 | 0.048726 |
| 31–35 | 92 | 0.352277 | 0.212332 |
| 36–40 | 14 | 0.875174 | 0.379779 |
| 41–50 | 23 | 0.480851 | 0.276252 |
| 51–+ | 1 | 2.914943 | 2.914943 |

| true-gap bin (eV) | n | MAE (eV) | bias (eV) |
|---|---:|---:|---:|
| [0.0, 4.5) | 16,066 | 0.138067 | 0.080413 |
| [4.5, 5.0) | 12,311 | 0.104517 | 0.037585 |
| [5.0, 5.5) | 13,354 | 0.101223 | 0.032424 |
| [5.5, 6.0) | 11,761 | 0.099787 | 0.021145 |
| [6.0, 100.0) | 20,053 | 0.134899 | 0.024166 |

UNK-containing molecules: 230 (0.313%); MAE with UNK 0.445766 eV, without 0.117750 eV.

