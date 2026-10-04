# REPORT — zinc-chemistry-dictionary-vs-mlp-seed0-v1

Status: **executed**.  Single seed (seed 0), exactly two formal GPU arms
(`D_g`, `M_g`), frozen fit-only target decomposition, same new internal fold as
the paired y round, no official-valid/test access, no 10k confirmation, no
write-back.  Branch `task/zinc-chemistry-dictionary-vs-mlp-seed0-v1`; training
revision `7c0224a` (remote `7c0224a37141`); both arms used the same Slurm pool,
node and software regime.

## 0. Conclusion first

The pre-registered dictionary gate is **`INCONCLUSIVE`** and the independent
target-interference label is **`RELIEF_UNCONFIRMED`**.  After removing the ring
component from the supervision label, the current task dictionary `D_g` does
**not** confirm a generalisation advantage over the matched MLP `M_g`, and `M_g`
does **not** confirm an advantage either:

* Primary endpoint — new-dev G0 (k=0) calibrated `g`-MAE:
  `D_g 0.103614` vs `M_g 0.101875`; gain `MAE(M) - MAE(D) = -0.001739`
  (negative = `M` better), paired CI `[-0.005604, +0.002716]` — crosses zero
  and the point is below the `0.003` threshold.
* G0 **raw** gain is `+0.002262` in favour of `D_g` (CI `[-0.002065,
  +0.006455]`).  The raw/cal sign disagreement is not a small technicality: the
  two arms' fit-median biases differ (`D_g -0.011670`, `M_g -0.031378`).
* New-dev overall calibrated `g`-MAE: `D_g 0.105486` vs `M_g 0.103717`,
  gain `-0.001769`, CI `[-0.005804, +0.002341]`; overall raw gain `+0.002122`
  in favour of `D_g`, CI `[-0.001982, +0.006244]`.
* In-sample the MLP is clearly better: fit calibrated `g`-MAE
  `D_g 0.033291` vs `M_g 0.029280`, fit gain `-0.004011` with CI
  `[-0.004993, -0.003077]` (separated).  That in-sample advantage does **not**
  carry to the dev endpoint.
* Removing ring supervision did **not** produce a confirmed subject-level
  change for either bridge: `B_D` cal `+0.000686` CI `[-0.003722, +0.004978]`,
  `B_M` cal `+0.001430` CI `[-0.003186, +0.005689]`, interaction
  `I = -0.000744` CI `[-0.006229, +0.005203]`; all points below `0.003` and all
  CIs cross zero, with raw/cal sign flips for `B_M` and `I`.

The one robust descriptive fact is that the extreme-tail error that dominated
the previous y round collapses when `c` is removed from the target: on the two
new-dev `k<=-3` rows the calibrated `g`-MAE is `D_g 0.171` / `M_g 0.645`,
versus `D_y 3.96` / `M_y 1.15` in the previous round.  With 2 dev rows this is
descriptive, not a confirmed gain.

## 1. How `g/c` labels were formed (Q1)

`y = g + c` holds exactly (`max |y-(g+c)| = 0.0`).  The labels come from the
train-only audit table (`train_cycle_audit_label.csv` +
`formula_verification_per_molecule.csv`, both 10000 positional train rows) and
the frozen train handoff `y`.  The cycle component is
`c = (k - mu_cycle)/sigma_cycle` with
`k = clip(round(eff_norm*sigma_cycle + mu_cycle), -20, 0)` and
`eff_norm = y - [(logP - mu_logP)/sigma_logP + (SA - mu_SA)/sigma_SA]`.

The previous round reused constants that had been fitted (OLS + Nelder-Mead
snap, `stage_refine`) on the whole 12000-row audit table, which includes the
current dev rows.  This round **refitted every constant on the 8000 fit rows
only** and froze them in `fit_only_targets.npz`:

| constant | old (all 12000) | new (fit-only 8000) | change |
| --- | --- | --- | --- |
| `sigma_logP` | 1.4351122692 | 1.4359246977 | +0.000812 |
| `sigma_SA` | 0.8321043074 | 0.8308644903 | -0.001240 |
| `mu_SA` | -3.1922244722 | -3.1917448189 | +0.000480 |
| `sigma_cycle` | 0.2883162645 | 0.2885487757 | +0.000233 |
| `mu_cycle` | -0.0001334074 | -0.0001670863 | -0.0000337 |

Consequences (reported before training in `target_provenance.json`):

* `k` labels are identical to the old decomposition on **all 10000 rows**
  (0 differences); fit counts 7713/252/30/5 and dev counts 1915/73/10/2 are
  reproduced exactly.
* `c` differs only by the constant refit: `max |c_new - c_old| = 0.033654`;
  `g` differs by the same amount.  On G0, `c0 = 0.0005790572` (old
  `0.0004627`).
* No current-dev label was used in any fit.  The `mu_logP` constant is the
  fixed label-generation community value; no cycle-algorithm search was run.
* Dev `k/c` are used only for grouped scoring.  The prediction path reads no
  label; the input standardizers were refit on fit rows only and are identical
  to the paired y round.

Because G0 has a constant `c0`, the new G0 `g`-MAE equals the y-unit MAE of
`pred_g + c0`; that is the only place where old y arms and new g arms are
compared.  The overall y-MAE of the old arms is **not** comparable to the
overall g-MAE of the new arms and is marked `y(ref)` everywhere.

## 2. `D_g` vs `M_g` (Q2)

Pre-registered endpoint (`raw_soup`, epochs 236-240, no dev selection):

| endpoint | `D_g` raw | `M_g` raw | `D_g` cal | `M_g` cal | cal gain (M-D) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| dev overall (2000) | 0.106834 | 0.108956 | 0.105486 | 0.103717 | **-0.001769** | [-0.005804, +0.002341] |
| dev G0 (1915) | 0.104989 | 0.107251 | 0.103614 | 0.101875 | **-0.001739** | [-0.005604, +0.002716] |
| fit overall (8000) | 0.034667 | 0.040199 | 0.033291 | 0.029280 | -0.004011 | [-0.004993, -0.003077] |
| fit G0 (7713) | — | — | 0.033000 | 0.029019 | -0.003981 | [-0.005020, -0.002979] |

Raw gains on dev: overall `+0.002122` (D better), CI `[-0.001982, +0.006244]`;
G0 `+0.002262` (D better), CI `[-0.002065, +0.006455]`.  Calibration biases:
`b_D = -0.011670`, `b_M = -0.031378`.

The endpoint ranking therefore depends on the calibration convention, and no
clause of `D_CHEM_SUPPORT` or `M_CHEM_SUPPORT` is met.  `LOCAL_EQUIVALENCE`
is also not declared because the overall cal CI extends beyond `±0.003`.
`TRADEOFF` is not met (G0 cal point `< 0.003`).

Group breakdown (dev, calibrated, `g`; contribution sums to overall MAE):

| group | n | `D_g` MAE | `M_g` MAE | `D_g` contribution | `M_g` contribution |
| --- | --- | --- | --- | --- | --- |
| k=0 | 1915 | 0.103614 | 0.101875 | 0.099210 | 0.097545 |
| k=-1 | 73 | 0.150140 | 0.126388 | 0.005480 | 0.004613 |
| k=-2 | 10 | 0.125078 | 0.182842 | 0.000625 | 0.000914 |
| k<=-3 | 2 | 0.170609 | 0.644664 | 0.000171 | 0.000645 |
| k<=-2 | 12 | 0.132666 | 0.259812 | 0.000796 | 0.001559 |

`M_g` is better on the dominant frequent group and `k=-1`; `D_g` is better on
`k=-2` and `k<=-3`.  The old y round was the opposite on the extreme rows
(`D_y` underfit the rare penalty: `k<=-3` dev y-MAE 3.96 vs `M_y` 1.15), so the
tail reversal is a target-interference effect, not a new dictionary property
that is CI-confirmed.

Sensitivity (pre-registered): dropping the single new-dev row with the largest
`|err_D_g_cal| + |err_M_g_cal|` (`train:8052`, combined error 10.805) gives
overall cal gain `-0.001468`, G0 cal gain `-0.001424`, overall raw gain
`+0.002415`; the category does not change.

## 3. `D_y -> D_g`, `M_y -> M_g` and the interaction (Q3)

Four arms on the same G0 rows (`y = g + c0` there); gain convention
`err(control) - err(candidate)`, so all entries below are "negative supervision
removed minus y-supervision":

| quantity | cal point | cal CI | raw point | raw CI |
| --- | --- | --- | --- | --- |
| `G_y = M_y - D_y` | -0.000995 | [-0.006240, +0.003491] | +0.001025 | [-0.004064, +0.005583] |
| `G_g = M_g - D_g` | -0.001739 | [-0.005604, +0.002716] | +0.002262 | [-0.002065, +0.006455] |
| `B_D = D_y - D_g` | +0.000686 | [-0.003722, +0.004978] | +0.000461 | [-0.003866, +0.004762] |
| `B_M = M_y - M_g` | +0.001430 | [-0.003186, +0.005689] | -0.000776 | [-0.005499, +0.003536] |
| `I = B_D - B_M = G_g - G_y` | -0.000744 | [-0.006229, +0.005203] | +0.001237 | [-0.004564, +0.007013] |

`G_y` reproduces the previous round exactly (`-0.0009945985`).  The per-row
identity `I = B_D - B_M = G_g - G_y` holds at every bootstrap sample (max abs
difference 0.0).  Point estimates all lie below `0.003`, CIs all cross zero,
and `B_M`/`I` flip sign between raw and cal.  No relief class is declared:
`BULK_RELIEF_SUPPORTED`, `D_SPECIFIC_RELIEF` and `M_SPECIFIC_RELIEF` are all
false; the label is `RELIEF_UNCONFIRMED`.

Both y/g pairs (D_y vs D_g, M_y vs M_g) share the same code path, pre-hashed
schedule and dropout RNG stream (the bridges contain no stochastic op; only the
shared body has Dropout(0.05) at fixed call sites), and the training-path code
is unchanged between the y execution revision `77c40dd` and this round.  The
differences are therefore target-only within each bridge family; this is
recorded as a descriptive qualifier, not as a proof about every historical
condition.  The fast-arm training logs confirm the same schedule and data
stream hashes (`7b11a529…`).

## 4. Fit vs dev, mechanism (Q4)

* Eval fit/dev gap (calibrated): `D_g` `0.033291 -> 0.105486` (x3.17),
  `M_g` `0.029280 -> 0.103717` (x3.54).  `M_g` fits the frequent chemistry
  better in sample (fit gain CI-separated) but the dev gap is not closed.
* `last` state (not the registered endpoint): `D_g` dev cal `0.117740` vs
  `M_g 0.123786` — a point reversal versus the soup, confirming that no
  dev-based member selection is legitimate here.
* Row movement (dev cal, M minus D): M better on 1046 rows, worse on 954,
  mean -0.001769, max |delta| 0.972; the mean comes from many small moves, not
  a few rows.
* The task dictionary code is **dense**: at the soup state the code nonzero
  fraction is 0.923 and the average active atoms per row is 265.8/288
  (~92%).  It must not be described as a sparse inductive bias.
* Neither bridge has dead dimensions (0/144 for both arms on fit and dev);
  bridge output RMS is larger for `M_g` (0.517) than `D_g` (0.387) at the soup.
* Train cost is not matched: `D_g` needs `1.78 s` per 8 train steps vs
  `M_g 0.75 s` (ratio 2.37); parameters are matched, FLOP/time are not.
* Parameter drift: `D_L` 0.745 / `V_L` 0.569; `fc1` 0.765 / `fc2` 0.664.
  Mean clip fraction 0.462 (D) vs 0.605 (M).

What this supports: `M_g` has a real in-sample fit advantage on the frequent
chemistry; `D_g` is relatively better on rare rows; the huge old tail failure
of `D_y` disappears under `g` supervision.  What cannot be attributed: whether
the remaining dev gap is capacity, optimisation, target rarity, or the shared
representation.  With one seed and 2/10 rare dev rows, none of these is
identified.

## 5. What is supported about the task dictionary, and the luyin19 boundary
(Q5)

Supported (this round, this fold, this implementation):

* The current fused task-dictionary bridge is statistically competitive with
  the matched MLP on G0 and overall under `g` supervision (all point
  differences `< 0.003`, all CIs crossing zero) and is clearly not dominated.
* The previous round's negative tail verdict for `D_y` is target-coupled: when
  `c` leaves the supervised target, `D_g` has the lower rare-row `g` error of
  the two points, and both arms' extreme errors drop by roughly an order of
  magnitude (descriptive).
* The "shared task dictionary" in this compressed skeleton is a dense
  post-fusion code, not a sparse structural dictionary; a win here would not
  be about sparsity.

Not supported / still open (luyin19):

* luyin19's claim that a shared dictionary learns a transferable **structure-
  attribute relation** is **not tested** here: the compressed skeleton has no
  structural `D`, no `U`/IHT path and no slot binding, and the bridge sees
  only the fused 144-d interface.
* The luyin19 open issue — how to fuse structure and semantics well — remains
  unaddressed by this comparison; the round neither confirms nor refutes it.
* No new `y` baseline, no deployable ring branch, no official-valid statement.

## 6. Statistics and quality

* 1000 paired bootstrap resamples, seed 20261004; the four G0 arms share the
  same row resample; no independent-resample subtraction.
* Self-tests all pass: identical predictions -> gain 0 / CI [0,0]; swapped arms
  -> mirrored point and CI; constant `+0.001` shift -> exactly `+0.001` on
  point and CI (and bounded by the shift magnitude).
* Contract checks: 15,120/15,120 steps per arm, `completed`, schedule and
  data-stream hashes equal to the frozen `7b11a529…`, fold hashes equal to the
  frozen `7bf1cfb8…` / `a61c8010…`, parameter audit 184,667 + 82,944 =
  267,611 per arm, CPU/GPU replay `1.19e-06` (D) / `1.43e-06` (M) at training
  time and `9.54e-07` / `1.19e-06` on the independent reload
  (`replay_checks.json`, threshold `1e-5`).  No contract errors.
* All prediction vectors are explicitly 1-D length 8000/2000 and aligned to
  the frozen stable-ID order.

## 7. Budget and closure

GPU wall: `D_g 908.7 s` + `M_g 573.4 s` = `1482.1 s = 0.4117 GPU-h`, plus two
remote smoke attempts (`~0.02 GPU-h`) — well inside the 1.0 GPU-h cap; at most
2 GPUs at once.  No job remains running.  Official-valid and official-test were
never loaded.  The branch is isolated; no push, no merge.

Artifacts: `main_table.csv`, `group_table.csv`, `paired_g0_gains.csv`,
`bootstrap.json`, `interaction.json`, `mechanism_health.json`,
`sensitivity.json`, `gate.json`, `replay_checks.json`,
`historical_anchor_checks.json`, `target_provenance.json`,
`init_identity.json`, `smoke_checks.json`, `budget.json`, `manifest.json`,
two figures, both arms' states/curves/meta and per-row predictions.
