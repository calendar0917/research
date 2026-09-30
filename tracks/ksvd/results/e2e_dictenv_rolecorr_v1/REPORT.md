# E2E-DictEnv-RoleCorr-v1 - report

- protocol: `e2e_dictenv_rolecorr_v1` (study `zinc-context-gap`, seed 0, CPU)
- git commit: `af0293ae5b7f2d026d16e643be182daff0761319`
- official test loaded: `false` (never instantiated)
- verdict: **ROLE_CORR_NO_MATERIAL_GAIN**

## Primary result (official valid, Top-5 soup)

| arm | coordinate | best valid MAE | soup valid MAE |
|---|---|---:|---:|
| A `TOPO` | frozen SDB K32/s8 + common1 (33) | 0.128810 | 0.123254 |
| B `CORR` | D_S16 + D_C16 + common1 (33) | 0.132879 | 0.126447 |

`relative improvement = (M_A - M_B) / M_A = -2.5910%` (frozen screening gate `>= 2%`, fired: `False`).

## Curves

| arm | first | last | min | min epoch | last-20 mean |
|---|---:|---:|---:|---:|---:|
| TOPO | 0.719359 | 0.135955 | 0.128810 | 309 | 0.139343 |
| CORR | 0.632212 | 0.140212 | 0.132879 | 313 | 0.139150 |

## Frozen inference probes (soup states)

- `M_A` = 0.123254, `M_B` = 0.126447
- `M_A0` (A, alpha32 -> 0) = 0.171613 (G = +0.048359)
- `M_C0` (B, alpha_C -> 0) = 0.161967 (G_C0 = +0.035520)
- `M_S0` (B, alpha_S -> 0) = 0.165276 (G_S0 = +0.038829, diagnostic)
- `M_Cshuf` (B, within-group attribute permutation, 5 seeds) mean G = +0.037430 (range +0.015202 .. +0.060752)

## Per-molecule paired difference (official valid)

- mean |err| A = 0.123254, B = 0.126447
- fraction of molecules where B is better = 0.4820
- mean / median (|err_B| - |err_A|) = +0.003194 / +0.002643

## Object and code usage

- `alpha_S` active atoms 16/16, effective 13.83, top1 share 0.5306
- `alpha_C` active atoms 16/16, effective 7.89, top1 share 0.5307

## Scope notes

- Baseline A is a **frozen-dictionary** re-run of the Sem108 route, not a historical
  end-to-end number; A and B share the readout initialisation bit-for-bit.
- Single seed; this is a screening round, not a significance claim.
- Controls C/D were only trained if the 2% screening gate fired; otherwise they are absent.
