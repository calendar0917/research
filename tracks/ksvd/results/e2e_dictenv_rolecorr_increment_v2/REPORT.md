# E2E-DictEnv-RoleCorr-Increment-v2 - report

- protocol: `e2e_dictenv_rolecorr_increment_v2` (study `zinc-context-gap`, seed 0)
- git commit: `8244602b2e5341d38a1dbe21b3dda6465e1debf0`
- device: `cpu`
- official test loaded: `false` (never instantiated)
- verdict: **INCREMENT_NO_MATERIAL_GAIN**

## Primary result (official valid, Top-5 soup, width 49)

| arm | third block | best valid MAE | soup valid MAE |
|---|---|---:|---:|
| A `EXTRA-STRUCT` | frozen K16/s4 structural residual | 0.131769 | 0.126393 |
| B `CORR-ADD` | frozen K16/s4 correspondence (RoleCorr D_C) | 0.128747 | 0.125662 |
| C `CORR-PCA-ADD` | train-fitted PCA16 of the same object | 0.130171 | 0.123292 |

`M_A - M_B = +0.000731` (frozen absolute gate `>= 0.003`, fired: `False`).

## Mechanism probes (soup states)

- block zero B: G = +0.017293
- block shuffle B: G = +0.018822
- object shuffle B: G = +0.021111
- block zero/shuffle A: +0.055292 / +0.076695
- block zero/shuffle C: +0.011696 / +0.024157

## Curves

| arm | first | last | min | min epoch | last-20 mean |
|---|---:|---:|---:|---:|---:|
| EXTRA-STRUCT | 0.646532 | 0.141928 | 0.131769 | 288 | 0.143776 |
| CORR-ADD | 0.588374 | 0.141438 | 0.128747 | 317 | 0.138201 |
| CORR-PCA-ADD | 0.643078 | 0.132216 | 0.130171 | 311 | 0.139365 |

## Paired per-molecule differences (official valid)

- A_vs_B: mean delta = -0.000731, median = +0.000621, fraction B better = 0.4930
- A_vs_C: mean delta = -0.003101, median = -0.001380, fraction B better = 0.5120
- C_vs_B: mean delta = +0.002369, median = +0.001416, fraction B better = 0.4850

## Scope notes

- All three arms keep the **full** frozen K32/s8 structural budget and share a
  bit-identical readout initialisation; only the appended 16-wide block differs.
- Single seed; this is a screening round, not a significance claim.
- Route 2 (joint 709-D encoding) is only authorised after a route-1 gate miss.
