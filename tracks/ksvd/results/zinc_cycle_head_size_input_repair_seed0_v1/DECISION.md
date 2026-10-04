# DECISION — zinc-cycle-head-size-input-repair-seed0-v1

## Final class: `INPUT_REPAIRED_NOT_FIT`

The pre-registered structure gate passed, so the two small heads were bought.
The paired result does not show a material fit improvement on the original
conflict/extreme tail, and the pinned performance gate is not met:

* structure gate: `global_min_l1 0.0060697235 → 0.0021677584` (−0.003902, pass)
  and `train:1424` splits to span 0 (pass); 4 conflict classes → 1, 10 → 2 rows;
* fit marker: original 5 `k<=-3` rows q-MAE `14.2209 → 14.1539` (−0.47 %,
  required ≥ 25 %) → **not met**;
* dev overall cal gain `+0.0001351`, CI `[-0.0007847, +0.0011728]`
  (must be ≥ 0.003 with CI lower > 0) → **not met**; raw gain `+0.0001266 > 0`
  and G0 worsening `+4.3e-5 ≤ 0.001` are the only positive gate clauses.

Per the frozen table this is `INPUT_REPAIRED_NOT_FIT` — "the structure gate
passed, but the original conflict/extreme-tail q fit shows no material
improvement; the current small-head training path is still limited; do not
conclude that every new representation is useless".  Exactly this.

## Why this is not `FIT_GAIN_NO_TRANSFER`

`FIT_GAIN_NO_TRANSFER` requires the pre-registered fit-improvement marker to be
met first (5 fit `k<=-3` rows q-MAE −25 %).  It moves −0.47 %, so the premise
is absent.  The structure repair is a *fit-side, full-key* fact; it is not the
"fit改善" marker defined for the heads.

## What the evidence localises

* Input identity: mostly repaired.  The only surviving conflict
  (`train:0593`/`train:9913`) has identical `T25`, identical `N=28` and
  `E=31`; N/E cannot split it by construction.
* Fitting path: the head's tail readout barely moves even where the input is
  now unique (`1424`, `1760`, `2347`, `3776`); the size branch is live
  (`W_S` norm 0.617) but the dev effect is a ~2e-3 reshuffle.
* Transfer/extrapolation: the two dev `k<=-3` rows have no exact fit match
  under either input; one gets worse (`2210`), one mildly better (`5050`).
* Still underdetermined: capacity vs optimisation vs target rarity with only 5
  fit / 2 dev extreme rows, one seed, one reused dev.

## Single next action (design only; not run here)

One pre-registered, fit-only diagnostic on the same frozen `X27` input and the
same head family that isolates the 5 extreme rows and asks whether they are
fittable at all under the same recipe.  If not, no new input should be bought
on this dev; if yes, the blocker is the joint whole-fit optimisation and the
candidate is one future frozen full-scale confirmation.  No feature/
hyper-parameter menu, no automatic start.

## Stop

All this-round compute is stopped.  No third arm, no extra seed, no Full
training, no official-valid/test read, no remote job.  Branch is isolated; no
push/merge.
