# Analysis tables — `e2e_dictenv_capacity_localization_v2`

Preregistration sha256 `c5004e3d2aee…`; commit `cc1165438ff5`; CAP-BASE = CSSD-q1; official test never loaded.

## 1. Parameter budget and architecture fingerprints

CAP-BASE 97727 params; added-parameter ratio 1.1634 (max 1.5); v2 counts match v1: True.

| candidate | added | total | matches v1 | shape fingerprint |
|---|---|---|---|---|
| M0 | 0 | 97727 | - | `4d4d8cebe82f…` |
| F | 29568 | 127295 | True | `e00737abb3b9…` |
| R | 34400 | 132127 | True | `326cf787c72a…` |
| G | 30336 | 128063 | True | `f0606a211603…` |

## 2. Initialization audit (step 0, before any training)

| candidate | step0 MAE | Δ vs CAP-BASE | mean shift | median shift | max shift | residual-zero identity | grads > 0 | hard ≤ 0.002 | preferred ≤ 0.001 |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.130028 | 0.000000 | 0.00000 | 0.00000 | 0.00000 | True | True | True | True |
| F | 0.130073 | 0.000045 | 0.00080 | 0.00078 | 0.00389 | True | True | True | True |
| R | 0.129986 | -0.000042 | 0.00059 | 0.00050 | 0.00378 | True | True | True | True |
| G | 0.129941 | -0.000087 | 0.00113 | 0.00114 | 0.00300 | True | True | True | False |

## 3. Phase A — M0 calibration (20 epochs, fresh Adam lr 1e-4)

| metric | value | threshold | pass |
|---|---|---|---|
| soup (1–20) | 0.128984 | ≤ 0.132 | True |
| last-5 mean (16–20) | 0.130331 | ≤ 0.135 | True |
| epoch 20 | 0.129672 | ≤ 0.137 | True |
| late slope (16–20) | -1.548e-04 | ≤ +5.0e-04 | True |

best 0.129672 (Δ -0.000356), soup Δ -0.001044, last-5 Δ 0.000303; verdict **WARM_ADAPTATION_PROTOCOL_VALIDATED**.

## 4. Phase B — repaired differential-LR screen (40 epochs, base 1e-4 / new 1e-3)

| candidate | params | step0 MAE | best valid | soup (21–40) | Δsoup vs M0 | Δ vs start | last-10 | Δlast10 vs M0 | branch grad | S1 | S2 | S3 | S4 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 97727 | 0.130028 | 0.128668 | 0.128723 | 0.000000 | -0.001305 | 0.130721 | 0.000000 | 0.000 | control | control | control | control | CONTROL |
| F | 127295 | 0.130073 | 0.129047 | 0.128613 | -0.000110 | -0.001415 | 0.130971 | 0.000250 | 0.191 | False | False | False | True | NO_CAPACITY_SIGNAL |
| R | 132127 | 0.129986 | 0.129948 | 0.129422 | 0.000699 | -0.000606 | 0.133506 | 0.002785 | 7.395 | False | False | False | True | NO_CAPACITY_SIGNAL |
| G | 128063 | 0.129941 | 0.128998 | 0.128879 | 0.000156 | -0.001149 | 0.130604 | -0.000116 | 0.054 | False | False | False | True | NO_CAPACITY_SIGNAL |

Winner: **None** (LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED); passing: []; frozen tie order F > R > G.

## 5. Phase C — full winner run (only if bought)

Not run: LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED.

## 6. Mechanism audit (frozen winner checkpoint)

Not run (no full winner).

