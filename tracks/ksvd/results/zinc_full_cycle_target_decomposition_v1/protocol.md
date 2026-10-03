# Protocol / pre-registration — zinc-full-cycle-target-decomposition-v1

Frozen before the formal runs.  Question, arms, recipe, gate and interpretation
branches are fixed here and are not changed after seeing dev results.

## Question

If the canonical `Full` (`tracks/ksvd/experiments/luyin16/e2e_dictenv_scale_v1.py`,
408,651 parameters) is **not** required to learn the rare cycle penalty while
learning the rest of the chemical objective, do the existing representation and
static composer generalise better on the remaining chemical targets?  This is a
target-decomposition training diagnostic, not a node-amplitude / WD / support-key
search and not a residual-readout change.

## Scope / authorisation

* Frozen recipe: model seeds `[0, 1]`, the canonical Full's own `scale_seed = 0`,
  240 epochs, Adam (coupled L2, not AdamW), `lr = 1e-3`, `wd = 1e-5`,
  batch 128, grad clip 5, no scheduler / AMP / DDP, FP32, fixed last-5-epoch
  soup (236–240) averaged.  No dev-based epoch / soup / seed / ensemble
  selection.
* Only the training task target changes between the paired arms:
  * `Y` (control): output `f(x)`, loss `L1(f(x), y)`, evaluation `p_Y = f(x)`.
  * `O` (oracle diagnostic): output `h(x)`, loss `L1(h(x), g)`, `g = y - c`,
    evaluation `p_O = h(x) + c`.
* The complete loss is always `task L1 + cm.H1_LAMBDA * structural
  reconstruction` with the frozen `cm.H1_LAMBDA = 33.95873017865987`; the task
  dictionary adds no reconstruction term.  Architecture, widths, dictionary
  init/normalisation, ISTA/IHT, C6 mask, Sem108, structural dictionary, task
  dictionary, structural–semantic binding, fusion, static unary/pair composer,
  `topology25 -> 8` and the reader are identical across arms and seeds.
* Model input `x` never contains `c`, `g`, `k`, `y` or `group id`.  `c` is used
  only to build the `O` supervision target and, at diagnostic evaluation time,
  added back externally.  Neither `c` nor `g` enters forward, routing, sampling,
  sample weighting or the reconstruction loss.
* `O` is always reported as **oracle diagnostic**.  Even if its dev MAE is
  `<= 0.09`, it is not a deployable model and not SOTA.

## Data — fixed 8000 fit / 2000 dev, train-only

* Reuse the `zinc_joint_dictionary_decision_v1` split exactly:
  `prep/fold_objects.npz` `fit_idx` / `dev_idx`, whose SHA-256 must equal
  `165e87ef…` / `fb8b7806…`.  No new fold, no 4000-row reduction.
* Expected dev strata: `penalty 0 = 1926`, `-1 = 65`, `<= -2 = 9` rows.
* Inputs: only `encoded_train.pt` + the train env cache (train-only).  The
  official ZINC **test** split is never instantiated; the official **valid**
  split is never loaded or read this round.  The 2000-row dev set is a fixed
  diagnostic set that has already been used by earlier rounds, not a new
  confirmation set.
* Fit objects (structural dictionary `D_fit`, common subspace, per-column
  standardisers, X175 normalisation) come exclusively from the 8000 fit rows;
  the 4000-row A/B blobs and the joint X175/D pre-training are not reused.
  Scalers are applied once; no re-inverse/refit on already transformed data.

## Target decomposition (`c`)

Per stable row (positional, `train:%04d`):

* `y` — the frozen handoff target (identical to `encoded_train.pt .y`).
* `k` — `round(label_effective_cycle_snapped)` from
  `zinc_long_cycle_audit/train_cycle_audit_label.csv`.
* `c = label_cycle_component = (label_effective_cycle_snapped - mu_cycle) / sigma_cycle`
  with `mu_cycle = -0.0001334074230764987`,
  `sigma_cycle = 0.28831626452532244` (`stage_refine.json`).
  This is a **label-derived oracle**: it is reverse-engineered from the label
  generator's additive cycle term, and the audit documents that the label's
  cycle basis is node-order dependent, so it is not a permutation-invariant
  structural feature the model can read.  `O` is therefore not a realisable
  performance upper bound.
* `g = y - c` (`= label y_without_cycle_label`).  The identity `y = g + c` is
  an implementation check, not a physical verification of `c`.
* `c` is already in `y` units and is never re-normalised.  `c` is not forced to
  zero for `k = 0`.

Failure to reproduce any of these inputs stops the round as INVALID/INCOMPLETE
(no fill-zero, no row deletion, no approximate cycle reconstruction).

## Calibration

Single calibration, fit-only, eval mode, on the **unfolded-bias** raw last-5
soup, with each arm's own 8000 fit predictions:

```
b_Y = median_fit(y - f_raw)
b_O = median_fit(g - h_raw)      # == median_fit(y - (h_raw + c)), verified
p_Y_raw = f_raw                  p_Y_cal = f_raw + b_Y
p_O_raw = h_raw + c              p_O_cal = h_raw + b_O + c
```

`b` is always added externally; the released soup state is never bias-folded.
The old `zinc_joint_dictionary_decision_v1.run_arm` double-calibration is not
copied.  The raw soup state is exported and replayed from disk; the released
dev predictions must reproduce within `2e-6`.

## Fixed analysis

* Primary comparison, per seed: raw/cal overall dev MAE, eval fit MAE,
  fit/dev gap, `b`; primary gain `MAE(Y_cal) - MAE(O_cal)` (positive = better).
  Primary summary is the equal-weight mean of the two seeds' gains; each seed is
  also reported.  No new prediction ensemble.
* Grouped by `k` (`0`, `-1`, `<= -2`): `n`, MAE, signed residual (`y - pred`),
  contribution `sum(abs(error)) / N` (`N = 2000`); group contributions sum to
  the overall MAE and group gains to the total gain.
* Per-row paired table (`paired_dev_predictions.csv`): per seed, per dev row,
  `g`, `c`, `k`, `y`, raw/cal predictions of both arms.
* Severe 9 rows: per-row changes.  Sensitivity: gain with the control's largest
  error row removed (chosen from each seed's `Y_cal`), and gain with all severe
  rows excluded.  These are descriptive only; no training sample is deleted and
  the main gate is unchanged.
* One fixed 1000-draw paired bootstrap, seed `20261003`, resampling **canonical
  groups** jointly across arms and seeds with the same index.  CIs describe only
  this diagnostic dev set and these two seeds.

## Pre-registered route signal

Positive route signal iff **both seeds' cal total gain `> 0`**, **mean gain
`>= 0.003`**, and **each seed's `G0` (`k=0`) MAE worsens by `<= 0.001`**.
Raw-direction is reported alongside; a raw/cal disagreement must be explained by
the bias.

Frozen interpretation branches (top-down, take the first that matches):

| observation | what this round can support | next design only |
|---|---|---|
| signal met and mean `G0` gain `>= 0.002` with both seeds improving | de-cycling training also moves bulk generalisation; decomposition supervision deserves further testing — not proof that the tail causes all trouble | supervise chemical / cycle branches separately under the same Full input interface; `c` only as a training-side auxiliary label, never at inference |
| signal met and `G0` preserved, gain mainly from severe | the severe-cycle prediction task is an important recoverable error pool; chemistry bulk is at least preserved under the oracle | test whether the original topology input can learn the cycle term independently; evaluate the component and overall `G0` protection; do not promise the oracle gain reproduces |
| `O` improves severe but clearly hurts `G0`, or seeds conflict | trade-off / instability, no reliable decomposition-route signal yet | stop the current purchase; report contributions and uncertainty; no `lambda` / WD / head-width rescue |
| `O` total gain tiny / negative and the de-cycled target still shows an obvious fit/dev gap | the current Full also has a generalisation problem on the remaining chemical target; the severe ring is not a sufficient explanation | only next round consider an equal-information structural–semantic fusion control; do not claim dictionary information is insufficient or regularisation is the only problem |

If no branch matches cleanly, record "no route choice possible" — do not force a
conclusion.  Near-threshold conditions are recorded as weak / ambiguous.

## Stop rule

After the four paired trajectories, all compute stops.  No 10000-row
confirmation, no official-valid evaluation, no cycle head, no D-wide, no dense
baseline, no fusion replacement, no node rescue, no coord perturbation, no
`lambda` / WD / loss / optimizer search, no third seed.  Both seeds are inside
the four trajectories; there is no seed-0 purchase gate.