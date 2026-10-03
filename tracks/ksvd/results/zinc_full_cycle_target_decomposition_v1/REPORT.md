# REPORT — zinc-full-cycle-target-decomposition-v1

Round type: pre-registered target-decomposition training diagnostic of the
canonical `Full` (408,651 parameters).  Four paired trajectories: seeds
`[0, 1]` × arms `Y` (control, task `L1(f(x), y)`) and `O` (oracle diagnostic,
task `L1(h(x), g)`, `g = y - c`, evaluated as `h(x) + c`).

**`O` is an oracle diagnostic, not a deployable model.**  The true `c` it adds
back is a label-derived term (`c = (snapped - mu_cycle)/sigma_cycle`): the
audit documents that the label's cycle basis is node-order dependent, so `c` is
not a permutation-invariant feature that the model could read from `x`.

## Data and identity

Fixed 8000 fit / 2000 dev split, train-only.  `fit_idx`/`dev_idx` SHA-256 equal
the frozen `165e87ef…`/`fb8b7806…`; dev strata `k=0: 1926`, `k=-1: 65`,
`k<=-2: 9`; no canonical group crosses the boundary.  `y` from
`encoded_train.pt` equals the decomposition exactly (`max_abs = 0`);
`y = g + c` and the `c` formula hold to `1.8e-15`; the all-train X175 hash is
reproduced.  Official ZINC test was never instantiated and the official
validation split was never loaded.

Seed-0 `Y`/`O` initial states are hash-identical (`beee36a2…`); seed-1 is
hash-identical within its pair (`cabe2fb0…`) and differs from seed 0.  All four
runs did exactly `15,120` optimizer steps (240 epochs × 63 batches), fixed
soup `236–240`, `cm.H1_LAMBDA = 33.95873017865987`.  The exported raw soup
state reloads and replays the released dev predictions within
`7.2e-7 – 1.4e-6` (tolerance `2e-6`).  Calibration is external (`b` stored, never
folded into the released state); for `O`, `median_fit(y-(h_raw+c))` equals
`median_fit(g-h_raw)` to machine precision.

## Main comparison (dev, calibrated, y units)

| seed | arm | b | dev raw MAE | dev cal MAE | eval fit cal MAE | fit/dev gap |
|---|---|---|---:|---:|---:|---:|
| 0 | Y | −0.02482 | 0.127943 | 0.125680 | 0.036655 | 0.089024 |
| 0 | O | +0.005535 | 0.102096 | 0.101626 | 0.032343 | 0.069284 |
| 1 | Y | −0.020798 | 0.122033 | 0.120339 | 0.032732 | 0.087608 |
| 1 | O | −0.012035 | 0.099596 | 0.098999 | 0.030337 | 0.068662 |

Primary gain `MAE(Y_cal) - MAE(O_cal)`: **+0.024053 (seed 0)**, **+0.021341
(seed 1)**, equal-weight mean **+0.022697** (positive = improvement).  Raw gain
is the same direction and slightly larger: +0.025846 / +0.022437, mean
+0.024142.  The fitted biases are small (`|b| <= 0.025`) and the raw/cal gains
agree in direction and magnitude, so the gain is not a calibration artefact.

For `O`, the calibrated dev error on `y` equals its error on `g` exactly (the
oracle offset cancels `c`), so the "true `g`-MAE" is `0.101626 / 0.098999`
(dev) and `0.032343 / 0.030337` (fit).

## Group decomposition (contribution = `sum|error| / 2000`)

| seed | group | n | Y MAE | O MAE | MAE gain | Y contrib | O contrib | contrib gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | k=0 | 1926 | 0.102129 | 0.100266 | +0.001863 | 0.098350 | 0.096556 | +0.001794 |
| 0 | k=-1 | 65 | 0.151461 | 0.117965 | +0.033496 | 0.004922 | 0.003834 | +0.001088 |
| 0 | k<=-2 | 9 | 4.979330 | 0.274684 | +4.704646 | 0.022407 | 0.001236 | +0.021171 |
| 0 | overall | 2000 | 0.125680 | 0.101626 | +0.024053 | 0.125680 | 0.101626 | +0.024053 |
| 1 | k=0 | 1926 | 0.102280 | 0.096945 | +0.005335 | 0.098496 | 0.093358 | +0.005138 |
| 1 | k=-1 | 65 | 0.178055 | 0.127952 | +0.050103 | 0.005787 | 0.004158 | +0.001629 |
| 1 | k<=-2 | 9 | 3.568223 | 0.329415 | +3.238808 | 0.016057 | 0.001482 | +0.014575 |
| 1 | overall | 2000 | 0.120339 | 0.098999 | +0.021341 | 0.120339 | 0.098999 | +0.021341 |

Identities hold to `1e-16`: per-group contributions sum to the overall MAE and
per-group contribution gains sum to the total gain, for both seeds.

Mean contribution gains: `k=0 +0.003466`, `k=-1 +0.001358`, `k<=-2 +0.017873`
(total `+0.022697`).  The severe tail is **78.7 %** of the equal-weight mean
gain; the bulk `k=0` moves but accounts for only `15.3 %`.

## Severe 9 rows and sensitivity

Eight of the nine severe rows improve substantially under `O`; the ninth
(`dev 1535`, `k=-2`) is already within `0.015` for the control and degrades by
`0.001` under `O`.  For the remaining rows the `O` calibrated prediction tracks
`y` to `0.006–1.12`, while `Y` misses the cycle-sized penalty.  For the extreme
`k=-12` row (`global_index 2210`, `y=-42.04`, `c=-41.62`) the control lands at
`-22.8` (`|err|=19.2`) while `O` lands at `-40.9` (`|err|=1.12`).

| seed | gain dropping the control's max-error row | gain excluding all severe rows |
|---|---:|---:|
| 0 | +0.015017 | +0.002895 |
| 1 | +0.014692 | +0.006796 |
| mean | +0.014855 | +0.004846 |

The control max-error row is chosen from each seed's `Y_cal` (seed 0:
`global_index 2210`, `k=-12`, `|err|=19.2`; seed 1: `global_index 5050`,
`k=-5`, `|err|=14.0`).  Removing the single worst control row leaves a large
gain; removing all 9 severe rows leaves a smaller but still positive gain
(seed 0 just below the 0.003 threshold, seed 1 clearly above; mean 0.0048).
No training sample was deleted.

## Fixed bootstrap

1000 paired draws, seed `20261003`, canonical groups resampled jointly across
arms and seeds (same index):

* mean cal gain `+0.02216` `[0.00654, 0.04323]`;
* mean `G0` gain `+0.00360` `[0.00074, 0.00658]` (seed 0 `[−0.00244, 0.00606]`,
  seed 1 `[0.00174, 0.00952]`).

The `G0` interval is positive on average but only marginally excludes zero, and
does not exclude zero for seed 0.  These intervals describe only this diagnostic
dev set and these two seeds; they are not a cross-seed population CI.

## Pre-registered route signal

| condition | value | met |
|---|---|---|
| both seeds cal gain > 0 | +0.024053 / +0.021341 | yes |
| mean cal gain ≥ 0.003 | +0.022697 | yes |
| each seed `G0` MAE worsen ≤ 0.001 | −0.001863 / −0.005335 | yes |

**Route signal met.**  Frozen branch, top-down: **B1 — "signal met and mean
`G0` gain ≥ 0.002 with both seeds improving"** (mean `G0` MAE gain `+0.003599`,
both seeds positive).  Per the frozen table this supports *de-cycle training
also changes bulk generalisation; decomposition supervision deserves further
testing — not proof that the tail causes all trouble*.

## Interpretation

1. **Can the remaining chemical target generalise well enough?**  `O` reaches a
   true `g`-MAE of `0.1016 / 0.0990` (dev) and `0.0323 / 0.0303` (fit),
   `G0` `g`-MAE `0.1003 / 0.0969`.  It is better than the control on both fit
   and dev (`Y`: dev `0.1257 / 0.1203`, fit `0.0367 / 0.0327`) and its fit/dev
   gap is smaller (`~0.069` vs `~0.088`).  But a gap of `~0.069` and a `G0`
   MAE of `~0.10` remain: the remaining chemical target is **not** solved, and
   this dev set is a reused diagnostic, not a new confirmation set nor a
   converted official-valid number.
2. **Where does the de-cycling gain come from?**  The equal-weight mean gain is
   `+0.02270`; the severe group contributes `+0.01787` (78.7 %), the `k=-1`
   group `+0.00136`, and the bulk `k=0` `+0.00347`.  The severe-row gain is
   oracle: it comes from adding back the true `c`, which the model cannot
   compute.  The only strictly non-oracle component is the bulk: `G0` improves
   by `+0.0036` mean (where `c ≈ +0.00046` is a constant), and excluding all
   severe rows the gain is still `+0.0048` mean.  This is a small but
   consistently positive effect, with a `G0` bootstrap interval that barely
   excludes zero for seed 0.
3. **Does this support a formal decomposition-supervision route?**  Only the
   weak form: training without the rare cycle target slightly improves the bulk
   and clearly removes the severe-tail error *under oracle `c`*.  The oracle /
   deployable gap is the whole severe gain plus most of the `k=-1` gain.  What
   is missing before any deployable claim is whether the original model inputs
   (in particular `topology25 → 8`) can predict the cycle term by themselves,
   and what a *training-side-only* use of `c` would buy.  `c`'s label-derived,
   order-dependent provenance means `O` is **not** a realisable upper bound.
4. **Single next action** (design only, not run): see `DECISION.md`.

## Boundaries / caveats

* Fixed `λ` means changing the task target changes the task/reconstruction
  gradient balance; the result therefore supports "reachable performance /
  training influence after removing this learning task", not a unique
  attribution to tail interference or to one optimisation mechanism.
* 2000-row dev is a fixed diagnostic set already used by earlier rounds; no
  official-valid conversion.  `O` is never a SOTA or deployable claim.
* No 10000-row confirmation, no official-valid, no cycle head, no extra seed,
  no `lambda`/WD/loss/optimizer search, no node rescue, no coord perturbation.
* `O`'s DONE/log line reports `dev_cal` without the oracle `c`; the released
  per-row predictions and `main_table.csv`/`group_table.csv` include it.  This
  is a cosmetic log issue only — the oracle offset is applied externally and the
  artifacts are correct.