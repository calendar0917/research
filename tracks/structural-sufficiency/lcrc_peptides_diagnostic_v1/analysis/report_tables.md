### T1 — label distribution (official Train/Validation)

| label | Train pos | Train n | Train pos % | Valid pos | Valid n | Valid pos % (= constant-AP baseline) |
|---|---|---|---|---|---|---|
| antifungal | 966 | 10873 | 8.88 | 207 | 2331 | 8.88 |
| cell_cell_communication | 385 | 10873 | 3.54 | 79 | 2331 | 3.39 |
| anticancer | 698 | 10873 | 6.42 | 150 | 2331 | 6.44 |
| drug_delivery_vehicle | 677 | 10873 | 6.23 | 146 | 2331 | 6.26 |
| antimicrobial | 6820 | 10873 | 62.72 | 1464 | 2331 | 62.81 |
| antiviral | 2148 | 10873 | 19.76 | 461 | 2331 | 19.78 |
| antihypertensive | 1162 | 10873 | 10.69 | 250 | 2331 | 10.73 |
| antibacterial | 2002 | 10873 | 18.41 | 433 | 2331 | 18.58 |
| antiparasitic | 217 | 10873 | 2.00 | 43 | 2331 | 1.84 |
| toxic | 2825 | 10873 | 25.98 | 609 | 2331 | 26.13 |

### T2 — learning-curve snapshots (Fast v1)

| epoch | seed0 loss | seed1 loss | seed2 loss | seed0 Valid AP % | seed1 Valid AP % | seed2 Valid AP % |
|---|---|---|---|---|---|---|
| 1 | 0.67495 | 0.56186 | 0.62722 | 31.35 | 30.57 | 30.68 |
| 20 | 0.27361 | 0.27564 | 0.27532 | 44.70 | 45.22 | 46.13 |
| 40 | 0.25912 | 0.25777 | 0.25672 | 47.40 | 48.26 | 49.54 |
| 60 | 0.24715 | 0.24844 | 0.24380 | 50.55 | 51.36 | 51.58 |
| 80 | 0.23957 | 0.23963 | 0.23466 | 52.42 | 51.86 | 53.31 |
| 100 | 0.23278 | 0.23471 | 0.22952 | 53.34 | 53.52 | 54.11 |
| 120 | 0.23034 | 0.23230 | 0.22755 | 53.57 | 53.79 | 54.76 |

### T3 — learning-curve summary (Fast v1)

| seed | best epoch | epochs run | best Valid AP % | Valid AP @100 % | Valid AP @last % | Δ Valid AP last 20 (pp) | new best in last 20 | train loss @100 | train loss @last | Δ train loss last 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 116 | 120 | 53.73 | 53.34 | 53.57 | +0.10 | True | 0.23278 | 0.23034 | -0.00229 |
| 1 | 119 | 120 | 53.94 | 53.52 | 53.79 | +0.14 | True | 0.23471 | 0.23230 | -0.00183 |
| 2 | 113 | 120 | 54.79 | 54.11 | 54.76 | +0.47 | True | 0.22952 | 0.22755 | -0.00219 |

### T4 — per-label Train/Validation AP of the frozen Fast v1 checkpoints

| label | Valid pos % (baseline) | seed0 Train AP % | seed1 Train AP % | seed2 Train AP % | seed0 Valid AP % | seed1 Valid AP % | seed2 Valid AP % | Valid AP mean % | mean Valid AP − pos rate (pp) |
|---|---|---|---|---|---|---|---|---|---|
| antifungal | 8.88 | 33.32 | 33.44 | 33.55 | 34.30 | 34.00 | 33.45 | 33.92 | +25.03 |
| cell_cell_communication | 3.39 | 39.53 | 39.45 | 39.80 | 40.73 | 42.48 | 40.64 | 41.29 | +37.90 |
| anticancer | 6.44 | 44.62 | 43.97 | 46.01 | 37.91 | 39.18 | 39.75 | 38.95 | +32.51 |
| drug_delivery_vehicle | 6.26 | 39.55 | 37.91 | 39.94 | 38.72 | 36.67 | 41.18 | 38.86 | +32.59 |
| antimicrobial | 62.81 | 92.84 | 92.66 | 92.98 | 91.36 | 91.39 | 91.81 | 91.52 | +28.71 |
| antiviral | 19.78 | 68.63 | 65.99 | 70.14 | 65.11 | 63.11 | 66.86 | 65.03 | +45.25 |
| antihypertensive | 10.73 | 90.84 | 90.69 | 91.07 | 88.86 | 89.20 | 89.36 | 89.14 | +78.41 |
| antibacterial | 18.58 | 53.11 | 53.40 | 53.41 | 48.28 | 50.57 | 49.51 | 49.45 | +30.88 |
| antiparasitic | 1.84 | 26.22 | 27.60 | 28.98 | 15.17 | 16.30 | 18.09 | 16.52 | +14.68 |
| toxic | 26.13 | 75.94 | 75.29 | 76.30 | 76.91 | 76.51 | 77.33 | 76.92 | +50.79 |

### T5 — ECFP4 + LightGBM per-label Validation AP

| label | Train pos % | Valid pos % (baseline) | Valid AP % | AP − baseline (pp) |
|---|---|---|---|---|
| antifungal | 8.88 | 8.88 | 50.30 | +41.42 |
| cell_cell_communication | 3.54 | 3.39 | 73.51 | +70.12 |
| anticancer | 6.42 | 6.44 | 66.17 | +59.74 |
| drug_delivery_vehicle | 6.23 | 6.26 | 69.23 | +62.97 |
| antimicrobial | 62.72 | 62.81 | 97.23 | +34.43 |
| antiviral | 19.76 | 19.78 | 86.71 | +66.93 |
| antihypertensive | 10.69 | 10.73 | 92.04 | +81.32 |
| antibacterial | 18.41 | 18.58 | 68.59 | +50.02 |
| antiparasitic | 2.00 | 1.84 | 35.32 | +33.47 |
| toxic | 25.98 | 26.13 | 90.66 | +64.53 |

### T6 — Validation comparison (this round)

| method | seed | cosine horizon | best epoch | epochs run | Valid macro AP % | note |
|---|---|---|---|---|---|---|
| LCRC-C1 (Fast v1) | 0 | 120 | 116 | 120 | 53.73 | historical frozen run; copied curve + checkpoint hash verified |
| LCRC-C1 (Fast v1) | 1 | 120 | 119 | 120 | 53.94 | historical frozen run; copied curve + checkpoint hash verified |
| LCRC-C1 (Fast v1) | 2 | 120 | 113 | 120 | 54.79 | historical frozen run; copied curve + checkpoint hash verified |
| LCRC-C1 (Fast v1) mean±std | 0,1,2 | 120 | — | 120 | 54.15 | sample std 0.56 pp over 3 seeds |
| Count ECFP4 + LightGBM | 0 | — | — | 500 | 72.98 | Train-only fit, Validation evaluation; no Test access |
| constant-score baseline | — | — | — | — | 16.48 | mean Validation positive rate over the 10 labels |
| LCRC-C1 long run (phase 1–120) | 0 | 240 | 120 | 240 | 57.72 | cosine horizon 240 (not 120) |
| LCRC-C1 long run (phase 121–240) | 0 | 240 | 231 | 240 | 60.62 | cosine horizon 240 |
| LCRC-C1 long run (overall) | 0 | 240 | 231 | 240 | 60.62 | cosine horizon 240; best Validation checkpoint |

### T7 — per-label Validation AP comparison

| label | Valid pos % (baseline) | Fast v1 seed0 % | Fast v1 seed1 % | Fast v1 seed2 % | ECFP4+LGBM % | long run % |
|---|---|---|---|---|---|---|
| antifungal | 8.88 | 34.30 | 34.00 | 33.45 | 50.30 | 36.97 |
| cell_cell_communication | 3.39 | 40.73 | 42.48 | 40.64 | 73.51 | 50.63 |
| anticancer | 6.44 | 37.91 | 39.18 | 39.75 | 66.17 | 46.97 |
| drug_delivery_vehicle | 6.26 | 38.72 | 36.67 | 41.18 | 69.23 | 47.68 |
| antimicrobial | 62.81 | 91.36 | 91.39 | 91.81 | 97.23 | 93.57 |
| antiviral | 19.78 | 65.11 | 63.11 | 66.86 | 86.71 | 74.11 |
| antihypertensive | 10.73 | 88.86 | 89.20 | 89.36 | 92.04 | 90.09 |
| antibacterial | 18.58 | 48.28 | 50.57 | 49.51 | 68.59 | 52.94 |
| antiparasitic | 1.84 | 15.17 | 16.30 | 18.09 | 35.32 | 32.09 |
| toxic | 26.13 | 76.91 | 76.51 | 77.33 | 90.66 | 81.23 |
