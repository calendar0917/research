# PROTOCOL — `zinc_pooling_scale_count_seed0_v1`

**Frozen before the two formal trajectories.** Round start 2026-10-04 ~22:39 CST. Exactly two
seed-0 trajectories are trained: `C` (sum/count control) and `N` (mean/count candidate). The
fresh-fold `M` soup (`zinc_local_tuple_fresh_fold_replication_seed0_v1`, commit `a5400de`) is a
read-only working reference; its `B` soup is historical context only and is not part of any gate.

## 1. Question

With the same information, the same parameter count and the same training recipe, does replacing
the unary/pair pooling *sum* moments by *count-normalised mean* moments improve the internal
chemical component `g = y - c`? Secondary: does merely exposing the six existing count
coordinates (unary `log1p(n)` + five pair-bucket `log1p(n)`) already improve it?

This round only studies the scale/accessibility of the existing six count input coordinates. It
does not add information (the C/N moments are recoverable from each other), does not add
parameters, and does not change the skeleton (`M` skeleton, 297,499 parameters).

## 2. Fixed inputs

* Official-train only (10000 rows); official-valid and official-test are never loaded.
* Fold: the frozen fresh fold, fit 8000 (`2a21cb8771f6…`) / dev 2000 (`270ab4126b0f…`),
  seed0; fit/dev/targets/prep/payload copied byte-identically from the source round and verified
  against their frozen SHA256.
* Targets: `g = y - c` with the source fit-only constants; body standardizers, tuple phi scaler
  and kappa are the frozen source values. No dev refit or selection.

## 3. Arms

| arm | unary and each pair bucket | everything else |
|---|---|---|
| `C` | `[Σz, Σz², log1p(n)]` | identical |
| `N` | `[Σz/max(n,1), Σz²/max(n,1), log1p(n)]` | identical |

`n` is the actual number of environment rows (unary) or pair rows in the bucket (pair) entering
the pool, per graph; empty pair buckets output two zero moments and count 0 with denominator 1.
Both arms restore the six existing count coordinates (global atom/bond histogram and relation
`path_count` stay masked as in the source C6 mask). Construction of both arms is byte-identical
to the source `M` init (297,499 parameters, `W_loc ≡ 0`, `A_raw = D_loc_init.T`, same RNG state).

## 4. Training recipe (frozen)

Seed 0, 240 epochs, 8000 fit, batch 128, 15,120 optimizer steps; Adam `lr=1e-3`, coupled
`weight_decay=1e-5`, `grad_clip=5.0`; FP32, no AMP/DDP; soup = mean of epoch-end states
236..240; no warm start from any trained state. The schedule and RNG streams are the source
recipe (`zw.build_schedule(8000, 240, 101)`; frozen schedule SHA `7b11a529…`; source `gid`
stream expected `69187f13…`). Diagnostics run in `eval`/`no_grad` and consume no training RNG.

## 5. Evaluation

* One fit-median bias per arm from its own raw fit predictions: `b = median(g_fit − pred_fit)`,
  `pred_cal = pred_raw + b`; folded once. Dev bias is never fitted.
* Endpoints: fit overall cal; dev overall raw/cal; dev G0 (k=0) raw/cal. `gain = MAE(control) −
  MAE(candidate)` (positive = candidate better).
* Groups: k=0, −1, −2, ≤−3 with `n`, MAE, contribution `Σ|err|/N_dev`; group contributions sum
  to overall.
* Paired bootstrap: 1000 draws, seed `20261006`, shared dev-row indices across arms; G0
  resamples within G0 rows, overall within all rows. Witnesses: identical predictions → zero
  gain/CI; swapped arms → mirrored sign.
* Sensitivity: drop the single dev row with the largest mean calibrated error per comparison
  (descriptive only).
* Scale diagnostic: fixed fit node-count quartile bins; fit/dev `n`, MAE and signed dev error
  per bin; not causal.

## 6. Gates and classification

Five conditions per comparison (control → candidate): G0 cal gain ≥ 0.003; overall cal gain ≥
0.003; G0 cal paired 95% CI lower > 0; G0 raw gain > 0; overall raw gain > 0.

Classification order (first match): 1) `N` passes vs `C` and vs `M` →
`NORMALIZATION_PERFORMANCE_CANDIDATE`; 2) `N` passes vs `M`, not vs `C` →
`INTERFACE_CANDIDATE_ATTRIBUTION_UNRESOLVED`; 3) `N` fails vs `M`, `C` passes vs `M` →
`COUNT_ACCESS_CANDIDATE`; 4) both `N`/`C` fail vs `M`, `N` passes vs `C` →
`NORMALIZATION_SIGNAL_NO_PRACTICAL_GAIN`; 5) otherwise `NO_CANDIDATE`.

## 7. Scope / policy

`g`-MAE is an internal chemical-component diagnostic, not an official `y`-MAE; no claim of
official-valid or deployable `y` performance is made even if `g`-MAE < 0.09. This dev has been
used for several prior decisions; one seed and one soup per arm cannot establish training
robustness. `test_policy: terminal`; no official-valid/test read.

## 8. Budget

Wall ≤120 min; no new computation started after minute 90; ≤0.8 allocation-GPU-hours; ≤2
concurrent GPUs; local CPU ≤8 threads. Remote `res-2`, pool `res2-cu124`, A100, torch
2.5.1+cu124, FP32.
