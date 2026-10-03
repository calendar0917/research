# REPORT — zinc-frozen-chemistry-learned-cycle-v1

Round type: fixed-config diagnostic.  The frozen `O` branch (`h_s(x)`, the
g-predictor from `zinc_full_cycle_target_decomposition_v1`) is **kept and used at
real inference**; a small cycle head `q_s(T)` is added on top.  This replaces the
old `DECISION.md` suggestion of a training-time auxiliary head that is discarded
at inference.

## Answer to the one question

How much of the frozen `O` oracle gain can the **existing Full topology25 input**
already learn, and therefore be retained without reading the true label-derived
`c`?

**A genuine, positive part: on the reused 2000-row diagnostic dev the deployed
`P = h_raw + q(T) + b_P` retains `0.5197` (seed 0) and `0.4320` (seed 1) of the
oracle gain, equal-weight mean `0.4758` (about half), and it never hurts `G0`.
The rare extreme tail is only partly recovered.**  The pre-registered inference
gate and the secondary `G0` marker both pass.

## Data and identity (all train-only)

* Fixed 8000 fit / 2000 dev split; `fit_idx` / `dev_idx` SHA-256 equal
  `165e87ef…` / `fb8b7806…`; dev strata `k=0: 1926`, `k=-1: 65`, `k<=-2: 9`.
* `c`/`g` reproduce the frozen definition; `median_fit(c) = 0.0004627120960246`.
* `T` is the **actual frozen Full topology25 model input**: built with the
  frozen runner's train-only `load_train_only` + `apply_prep_train_only`
  (8000-fit standardization; the same field the `O` arm consumed), all 10000
  rows, `sha256 = dc2e1516…`.  Columns/order unchanged; no SMILES, atom/bond,
  label, group or molecule id enters the head.
* `h_raw` is the cached `fit_raw`/`dev_raw` of `O_seed{0,1}_predictions.npz`
  (not the last state, not `h+c`).  Re-forwarding the exported raw soup on a
  fixed 32-row dev batch reproduces the cache within `9.5e-7` / `1.4e-6`
  (`<= 2e-6`); the frozen O parameters are unchanged
  (`e759f906…` / `cbdb6bd2…`) and match the old manifest.  `h_raw + b_O + c`
  reproduces the released dev MAE exactly
  (`0.1016261613` / `0.0989988937`).
* Frozen `O`/`Y` artifacts are byte-identical to the old manifest
  (`manifest.json → frozen_artifacts_unchanged = true`).

## Fixed head, smoke, and fit

* Architecture `25→64→32→1`, **3,777** params; first two layers default init from
  head seed 0/1; last layer `weight=0`, `bias=median_fit(c)`.  300 epochs, Adam
  (coupled L2), `lr=1e-3`, `wd=1e-5`, batch 128, clip 5, mean `L1(q(T), c)`,
  FP32, no scheduler/AMP/DDP.  Final model = mean of epoch-end states 296–300.
  Two fresh heads (seed 0/1), 18,900 steps each, ~10.0 s each.  Train inputs are
  only `T_fit` and `c_fit`; dev labels were not used for fitting/early stop/
  calibration.
* Smoke (4 steps, discarded): the last layer updates; layers 1–3 have zero
  gradient on step 1 by design (last-layer `W=0`) and update afterwards; the
  frozen O hash is unchanged.  `smoke_checks.json ok=true`.

## A. What the head learned (fit vs dev, `q` vs `c`, y units)

| split | group | n | `q` vs `c` MAE (s0/s1) | constant vs `c` MAE |
|---|---|---:|---:|---:|
| fit | overall | 8000 | 0.01371 / 0.01449 | 0.15261 |
| fit | k=0 | 7702 | 0.00087 / 0.00096 | 0.00000 |
| fit | k=-1 | 260 | 0.0460 / 0.0641 | 3.468 |
| fit | k=-2 | 33 | 0.603 / 0.625 | 6.937 |
| fit | k<=-3 | 5 | 14.22 / 14.25 | 18.04 |
| fit | k<=-2 | 38 | 2.395 / 2.418 | 8.397 |
| dev | overall | 2000 | 0.01263 / 0.01403 | 0.16648 |
| dev | k=0 | 1926 | 0.00048 / 0.00060 | 0.00000 |
| dev | k=-1 | 65 | 0.0119 / 0.0376 | 3.468 |
| dev | k<=-2 | 9 | 2.617 / 2.717 | 11.947 |

`q` output std/range: fit `std 0.764/0.759` over `[-17.26, 0.042]`; dev
`std 1.151/1.143` over `[-40.44, 0.026]`.

The head **clearly reads the discrete cycle levels** from topology25:
`k=-1` (`c=-3.468`) and `k=-2` (`c=-6.936`) are learned to high accuracy
(`~0.05` / `~0.6` fit MAE, dev `k=-1` `~0.01`).  The fixed descriptive criteria
are **overall drop `90.5–91.0 % >= 50 %` (met)** but **fit severe MAE
`2.395/2.418 > 1.0` (not met)**, because the 5 `k<=-3` fit rows (true `c` down to
`-20.8`) dominate the 38-row aggregate.  So this is *not* a near-constant head;
it is a head that completes the bulk/discrete readout but not the rare extreme
tail.  Per the frozen note this diagnostic is a reference, not a purchase gate;
it does not license "T25 has no information".

### Severe 9 dev rows (seed 0; `c` is real, `q` is pure prediction)

| global | k | y | c | q | c-q | Y err | P err | O err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2210 | -12 | -42.04 | -41.62 | -40.44 | -1.18 | 19.21 | 2.30 | 1.12 |
| 5050 | -5 | -15.33 | -17.34 | -5.64 | -11.70 | 13.15 | 12.19 | 0.49 |
| 1238 | -2 | -5.44 | -6.94 | -7.02 | +0.08 | -1.00 | -0.46 | -0.38 |
| 2052 | -2 | -7.93 | -6.94 | -2.62 | -4.32 | 5.66 | 4.48 | 0.16 |
| 4485 | -2 | -5.58 | -6.94 | -6.96 | +0.02 | -0.27 | -0.03 | -0.01 |
| 4344 | -2 | -7.18 | -6.94 | -7.02 | +0.08 | 0.32 | 0.05 | 0.13 |
| 7659 | -2 | -5.10 | -6.94 | -6.91 | -0.03 | -0.01 | 0.04 | 0.02 |
| 5093 | -2 | -7.99 | -6.94 | -7.02 | +0.08 | -0.39 | 0.02 | 0.10 |
| 7507 | -2 | -9.05 | -6.94 | -0.87 | -6.06 | 4.79 | 6.12 | 0.06 |

Most `k=-2` rows are recovered to `~0.02–0.08`; the two misses (`2052`, `7507`)
and the rare extremes dominate the severe MAE.  Seed 1 is analogous.

## B. Full prediction (dev MAE vs `y`, y units)

| seed | arm | b | dev raw | dev cal | fit cal | dev G0 | dev k=-1 | dev k<=-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Y | -0.02482 | 0.12794 | 0.12568 | 0.03666 | 0.10213 | 0.15146 | 4.979 |
| 0 | **P** | +0.00521 | 0.11361 | **0.11318** | 0.04516 | 0.10029 | 0.11541 | 2.856 |
| 0 | O | +0.00554 | 0.10210 | 0.10163 | 0.03234 | 0.10027 | 0.11796 | 0.275 |
| 0 | K | +0.00365 | 0.26407 | 0.26389 | 0.18366 | 0.11429 | 0.25025 | 2.914 |
| 1 | Y | -0.02080 | 0.12203 | 0.12034 | 0.03273 | 0.10228 | 0.17805 | 3.568 |
| 1 | **P** | -0.01235 | 0.11176 | **0.11112** | 0.04333 | 0.09692 | 0.12671 | 3.036 |
| 1 | O | -0.01203 | 0.09960 | 0.09900 | 0.03034 | 0.09695 | 0.12795 | 0.329 |
| 1 | K | -0.01367 | 0.26110 | 0.25997 | 0.18148 | 0.11241 | 0.24077 | 3.040 |

`K = h_raw + median_fit(c)` is the de-cycled branch with a constant offset; it
carries no cycle information and is worst on the tail (`~0.26`), so `P` beating
`K` by `~0.15` is genuine learned-cycle content, not a constant shift.

**Gains (positive = improvement; point from the full dev, not the bootstrap):**

| quantity | seed 0 | seed 1 | mean |
|---|---:|---:|---:|
| `MAE(Y_cal)-MAE(P_cal)` | +0.012500 | +0.009220 | **+0.010860** |
| same, raw | +0.014330 | +0.010270 | +0.012300 |
| `MAE(K_cal)-MAE(P_cal)` | +0.150706 | +0.148853 | +0.149779 |
| `MAE(P_cal)-MAE(O_cal)` (oracle loss) | +0.011554 | +0.012121 | +0.011838 |
| oracle retention `gain(Y→P)/gain(Y→O)` | 0.5197 | 0.4320 | 0.4758 |

Raw and calibrated gains agree in sign and magnitude for both seeds.

**Group contributions** (`Σ|error|/2000`) and gain:

| seed | group | n | Y | P | O | gain Y→P | oracle loss |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | k=0 | 1926 | 0.098350 | 0.096579 | 0.096556 | +0.001771 | +0.000023 |
| 0 | k=-1 | 65 | 0.004922 | 0.003751 | 0.003834 | +0.001172 | -0.000083 |
| 0 | k<=-2 | 9 | 0.022407 | 0.012850 | 0.001236 | +0.009557 | +0.011614 |
| 1 | k=0 | 1926 | 0.098496 | 0.093338 | 0.093358 | +0.005158 | -0.000021 |
| 1 | k=-1 | 65 | 0.005787 | 0.004118 | 0.004158 | +0.001669 | -0.000040 |
| 1 | k<=-2 | 9 | 0.016057 | 0.013664 | 0.001482 | +0.002393 | +0.012182 |

Identities hold exactly (group contributions sum to the overall MAE; group gains
sum to the total gain).  The retained gain is split between the bulk `k=0`
(seed 0 `+0.00177`, seed 1 `+0.00516`) and the cycle groups; the oracle loss is
almost entirely the `k<=-2` tail.

**Per-row identity** (`paired_dev_predictions.csv`):
`e_g = g - h_raw`, `e_c = c - q`, `e_P = e_g + e_c - b_P` verified to
`< 1e-9` for every dev row.  Component errors cancel on some rows (e.g.
`global 2052`, seed 0: `e_g=-1.87`, `e_c=-4.32`, `e_P=+4.48`) and add on others
(`global 5050`: `e_g=+1.69`, `e_c=-11.70`, `e_P=+12.19`), so the `c`-MAE cannot
be read as the overall gain directly.

**Sensitivity (descriptive):** dropping the control's single worst row
(seed 0 `global 2210`, seed 1 `global 5050`) the gain is `+0.00405 / +0.00856`;
excluding all severe rows it is `+0.00296 / +0.00686` (still positive).

**`G0` protection:** `P` improves `G0` relative to `Y` by `-0.001839` (s0) /
`-0.005356` (s1) MAE (negative = better).  `G0 MAE(P) - MAE(O)` is
`+2.4e-5` / `-2.1e-5`, both `<= 0.001`: the original O bulk gain is fully
preserved.

**Fixed bootstrap** (1000 draws, seed 20261003, canonical groups, same index
across arms and seeds):

* mean `gain(Y→P) = 0.010717 [0.002923, 0.022493]`
  (seed 0 `0.012172 [0.000132, 0.031501]`, seed 1 `0.009263 [0.003707, 0.015314]`);
* mean `gain(K→P) = 0.148566 [0.107975, 0.199572]`;
* mean `G0 gain(Y→P) = 0.003594 [0.000741, 0.006584]` (seed 0 interval includes 0);
* mean `G0 marker P-O = 2.3e-6 [-4.4e-5, 4.6e-5]`.

These intervals describe only this reused diagnostic dev and these two paired
seeds; they are not a cross-seed population CI.

## Deployability / no-true-c evidence

* `deploy_wrapper.py` builds `p = frozen_h(x) + q(T) + b_P`; `forward(batch,
  topo25)` takes no label table.  Wrapper vs cached `P` differs by `8.5e-7` /
  `1.4e-6`; repeated calls are bit-identical; permuting `c`/`g`/`k`/`y` dev
  labels changes the prediction by exactly `0`.
* The exported head soup, reloaded and replayed over all 10000 `T25` rows,
  reproduces the saved `q` with `max_abs = 0`.
* Exactly one fit-only calibration `b_P` per seed; `q` is never separately
  calibrated; no dev offset/gating is fitted.

## Pre-registered inference gate

| condition | value | met |
|---|---|---|
| both seeds `gain(Y→P)_cal > 0` | +0.012500 / +0.009220 | yes |
| mean `gain(Y→P)_cal >= 0.003` | +0.010860 | yes |
| both seeds `gain(K→P)_cal > 0` | +0.150706 / +0.148853 | yes |
| mean `gain(K→P)_cal >= 0.003` | +0.149779 | yes |
| each seed `G0 MAE` worsen vs Y `<= 0.001` | -0.001839 / -0.005356 | yes |
| identity / no-true-c / single-cal / replay | all pass | yes |
| **secondary** `G0 MAE(P)-MAE(O) <= 0.001` | +2.4e-5 / -2.1e-5 | **yes** |

**Gate met; secondary marker met.**  Boundary note: the fit-severe reference
`<= 1.0` is not met (the rare `k<=-3` tail dominates), so the extreme-tail
readout is not established even though the gate is met.

## Boundaries / caveats

* `c` is label-derived and its cycle basis is node-order dependent; `T25` cannot
  in principle recover the exact order-dependent value.  The head recovers the
  discrete level, not the exact tail.
* Fit-severe `2.395` is dominated by 5 rare rows; the head is far from constant
  (overall `90–91 %` drop, `k=-1`/`k=-2` levels learned).
* 2000-row dev is a reused diagnostic set, not a new confirmation set; no
  official-valid conversion.
* A positive result here is "preliminary real inference gain", not proof of
  dictionary specificity and not a claim that this head reaches dev `0.09`.
  If `q` were exactly `c`, `P` would reproduce this round's `O ~= 0.1003`; so a
  remaining chemistry-branch gap exists.