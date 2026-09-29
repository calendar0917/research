# Analysis tables — `e2e_dictenv_capacity_localization_v1`

Preregistration sha256 `f5c4a8d2e489…`; commit `becc6f6bb699`; CAP-BASE = CSSD-q1; official test never loaded.

## 1. Parameter budget

CAP-BASE 97727 params; added-parameter ratio 1.1634 (max 1.5).

| candidate | added | total | relative | preferred | hard ceiling |
|---|---|---|---|---|---|
| M0 | 0 | 97727 | 0.00 % | False | True |
| F | 29568 | 127295 | 30.26 % | True | True |
| R | 34400 | 132127 | 35.20 % | True | True |
| G | 30336 | 128063 | 31.04 % | True | True |

## 2. Initialization audit (step 0, before any training)

| candidate | step0 MAE | Δ vs CAP-BASE | mean shift | max shift | grads > 0 | bit-identical |
|---|---|---|---|---|---|---|
| M0 | 0.130028 | 0.000000 | 0.00000 | 0.00000 | True | True |
| F | 0.131102 | 0.001074 | 0.01003 | 0.05211 | True | True |
| R | 0.130860 | 0.000832 | 0.01741 | 0.12942 | True | True |
| G | 0.129941 | -0.000087 | 0.00113 | 0.00300 | True | True |

## 3. Screening (40 epochs warm from the CAP-BASE soup, fresh Adam)

| candidate | params | step0 MAE | best valid | best epoch | soup (21–40) | last-10 mean | Δsoup vs M0 | Δlast10 vs M0 | branch grad | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 97727 | 0.130028 | 0.135621 | 21 | 0.129662 | 0.154314 | 0.000000 | 0.000000 | 0.000 | CONTROL |
| F | 127295 | 0.131102 | 0.136613 | 37 | 0.129464 | 0.150048 | -0.000198 | -0.004266 | 0.239 | NO_CAPACITY_SIGNAL |
| R | 132127 | 0.130860 | 0.138940 | 28 | 0.129168 | 0.149848 | -0.000494 | -0.004466 | 2.513 | NO_CAPACITY_SIGNAL |
| G | 128063 | 0.129941 | 0.135855 | 27 | 0.129663 | 0.147582 | 0.000002 | -0.006732 | 0.001 | NO_CAPACITY_SIGNAL |

Winner: **None** (NO_CLEAR_CAPACITY_LOCALIZATION); passing: []; frozen tie order F > R > G.

## 4. Full winner run (only if bought)

Not run: no candidate passed the preregistered capacity gate.

## 5. Mechanism audit (frozen winner checkpoint)

Not run (no full winner).

