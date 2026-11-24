# ERRATA — structure-route-resolution-8h-v1

**Appended:** 2026-11-24 (audit round `r2_internal_binding_gap_audit_v0`)
**Status:** summary-field / report-text correction only. No training run is affected.
No historical file is rewritten; this file is the append-only correction record.

## E1 — D1 identity in `REPORT.md` §1

The line

> D1 identity/fit/eval: `60ac13c0…/b9f74030…/342c1de0…`

records the **D0** identity prefix for D1. The correct frozen D1 identity is

`c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee`

so the line should read

> D1 identity/fit/eval: `c94e2d4b…/b9f74030…/342c1de0…`.

The D1 Fit-row hash (`b9f74030…ebbc`) and Dev-row hash (`342c1de0…80a6`) in that
line were already correct.

## E2 — D1 identity in `analysis/queue_lock.json`

`d1_data.identity_sha256` records

`60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e`

which is the D0 identity. The correct value is

`c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee`.

The same queue-lock entry already records the correct D1 Fit rows
(`b9f74030…ebbc`), D1 eval rows (`342c1de0…80a6`), and D1 target scaler
(`mean 2.151278011472791`, `std 1.2159778136886414`, `n 2856`).

## What was verified

The audit (`tracks/structural-sufficiency/r2_internal_binding_gap_audit_v0/analysis/identity_audit.json`)
recomputed the D1 split with the frozen loader and checked **every** D1 run manifest
in `results/` (nine runs: the seven locked D1 runs plus the two D1/s1 conflict-package
runs). For every D1 run:

- manifest split = `D1`;
- `data.identity_sha256` / `split.identity_sha256` = `c94e2d4b…6bee`;
- Fit rows = `b9f74030…ebbc`, Dev rows = `342c1de0…80a6`;
- target scaler = `mean 2.151278011472791`, `std 1.2159778136886414`, `n 2856`;
- stored Dev prediction row ids equal the frozen D1 Dev rows and stored labels equal
  the frozen D1 labels;
- best checkpoint epoch equals the manifest `best_epoch` and the checkpoint split
  identity is `c94e2d4b…6bee`.

The R2 and EGO2 D1/s0 best checkpoints were also replayed on D1 Dev rows; stored
predictions reproduce with maximum absolute differences `1.41e-6` (R2) and
`1.38e-6` (EGO2), well inside the audit's `< 1e-5` criterion
(`r2_internal_binding_gap_audit_v0/analysis/replay.json`). No run-level evidence
of a wrong row set was found.

## Consequence

- No retraining, no checkpoint replacement, no result overwrite.
- D1 comparisons may proceed; the D1 numbers in the parent report are not changed by
  this erratum.
- The recorded D1 identity in this audit is `c94e2d4b…6bee`; the D0 identity remains
  `60ac13c0…559e`.