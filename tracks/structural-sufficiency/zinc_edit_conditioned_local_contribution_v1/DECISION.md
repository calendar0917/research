# DECISION — ECLC-v1

**Decision label:** `STOP_NO_COMPOSITION_GAIN`

**Scope:** zinc-edit-conditioned-local-contribution-v1, official PyG ZINC subset 10k/1k, frozen LCRC-A backbone, seed 0 only.

## Why

The seed-0 promotion rule required all five conditions. Three failed:

1. REAL full valid `0.077558` > `0.075`.
2. REAL − NULL full valid `−0.001324` > `−0.003`.
3. REAL − NULL `r2_unseen_ordinary` `−0.002518` > `−0.005`.

Conditions 4 (not worse on ordinary and `minus_row172`) and 5 (windows and base-path activity) passed:

- ordinary delta `+0.000090` ≤ `+0.001`; `minus_row172` delta `−0.001561`;
- REAL W20 `0.078498` < NULL `0.079184`; REAL T100 `0.078688` < NULL `0.079226`;
- zeroing `f_base` at the best REAL checkpoint changes valid MAE from `0.077558` to `0.476070` (|Δ| `0.3985`), base per-molecule std `0.5698`.

The REAL direction is positive on full valid (`−0.00132`) and unseen radius-2 ordinary (`−0.00252`), but both effects are below the pre-registered thresholds, both 95% CIs include zero, and the per-molecule win rates are below 0.5 (`0.491` full, `0.467` unseen ordinary). Ordinary and seen molecules are slightly worse for REAL. The full-valid level is also above the absolute `0.075` gate; both arms are about `0.003` above the historical LCRC-A 4-seed valid `0.075990`, near the known `~0.0026` drift band.

## What was not done

- Seed 1 was **not** launched; the protocol permits it only after full seed-0 promotion.
- No architecture, width, parameter count, pair count, loss coefficient, epoch, data-order or slice change was made.
- No replacement experiment, grid search, extra seed or post-hoc rescue was started.
- The official test split was never constructed, loaded, predicted or scored.

## Correct interpretation

Under this frozen pilot, forcing `f_base` to atom type + rooted radius-1 typed adjacency did **not** produce a REAL-vs-NULL gain of the pre-registered size on unseen radius-2 environments. This is a negative pilot, not proof that radius-1 information is useless: the parameter-identical NULL was not worse, and REAL has 253 dead `f_base` input columns, which biases the comparison toward NULL.

No claim of composition transfer, edit-class identity, or a new relative-learning paradigm is made.

## Provenance and budget

| item | value |
|---|---|
| execution commit | `2b0208c088d097726e184ffea21051bb0c29cde2` |
| REAL run | `eclc-real-s0-20261029-225105-822a37b8` |
| NULL run | `eclc-null-s0-20261029-225028-3910e6d9` |
| REAL best checkpoint | `84ac7bec6547cd0a8830239cd8ec5f226f22465da001378573debe40aebddcb1` |
| NULL best checkpoint | `b37042f872c2077f5a02ad0fad4758a5177843a6409da28bcdbc7de055980756` |
| parameters / updates | 248,368 / 47,400 per arm |
| GPU·h seed 0 | 1.459 (REAL 0.775 + NULL 0.683 + smoke 0.002) |
| seed-0 cap | 2.0 A100·h (respected) |
| seed 1 / cumulative | 0 GPU·h / 1.459 of 4.0 A100·h |
| official test loaded | false |

Detailed valid slices, CIs and checks are in `REPORT.md`, `analysis/promotion.json`, `analysis/main_table.json`, `analysis/slices.json`, and `analysis/cpu_checks.json`.
