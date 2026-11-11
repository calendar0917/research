# Read-only checkpoint diagnostics

## W1

not available

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

not available

