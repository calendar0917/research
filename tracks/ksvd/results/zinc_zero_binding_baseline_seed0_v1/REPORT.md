# REPORT — zinc-zero-binding-baseline-seed0-v1

Single seed (0), one new formal trajectory (`N0`), train-inner dev only;
official-valid/test were never instantiated or scored.  Source: branch
`task/zinc-structure-semantic-factorial-seed0-v1`, HEAD
`4083ee38d8785c5a44d1f9628a5baeed568d4c34` (source arms trained at
`118e2481362f`).

## 0. Clarification on the previous round's candidate bar (no old file modified)

The source round (`zinc-structure-semantic-factorial-seed0-v1`) measured
`S_M` dev G0 cal MAE `0.097347`, G0 gain vs `S_J` `+0.004981` with paired 95%
CI `[+0.000740, +0.009020]`, and overall cal gain `+0.002471`.  That is a
**numeric pass of the pre-registered single-seed performance candidate
condition**.  The source round did not promote it because its binding channel
had collapsed to exactly zero, so the pass could not be read as evidence that
the *marginal binding operator* is a better mechanism.

This round records the distinction explicitly: **channel collapse does not
cancel the performance signal; it limits the mechanistic interpretation.**
"`S_M` is a single-seed G0 candidate relative to `S_J`" and "the marginal
binding operator is better" are different claims; only the first is supported.
No old report, protocol or decision file was edited.

## 1. What was tested

* **Phase A (inference):** is the trained `S_M` final prediction actually
  independent of the structural binding path, and can an algebraically
  equivalent reduced model be exported?
* **Phase B (training):** one new 240-epoch trajectory `N0`, identical to
  `S_M` (sparse tied-IHT code, independent-pairing aggregation, same init
  RNG, same stream, same loss/optimizer/clip, same 408,651 parameters) except
  that both slot tensors are multiplied by zero after aggregation and before
  the slot encoders, from the first optimizer step.

The structural dictionary `D`/`U` (phi65 -> common coordinate + sparse
residual code) is the object under test.  The task dictionary `D_L/V_L`,
Sem108 (shell-conditioned semantic statistics), size2, topology25 and the
static relations remain everywhere.

## 2. Phase A result — S_M does not need the binding path

All 8000 fit + 2000 dev rows, eval mode, CPU float32 (`phaseA_precheck.json`):

* node slots and edge slots are **exactly zero** everywhere: absmax `0.0`,
  RMS `0.0`, within-batch row variation `0.0`; encoder outputs are row
  constants (`rowvar = 0.0`);
* replacing the structural code with zeros or with a fixed finite vector, and
  perturbing `D`, changes the prediction by `0.0`;
* the exported `S_M_deploy` model (fusion input `110 = [Sem108(108); size2(2)]`,
  no `D`/`U`/IHT/binding projections/slot encoders) reproduces the full-model
  raw prediction on fit and dev with `max|diff| = 1.907e-6 ≤ 1e-5`
  (`phaseA_export.json`, `replay_deploy.json`); label shuffle `0.0`,
  graph-order `2.4e-7`, independent re-load `0.0`;
* deployment size: **267,611** parameters (removed 141,040 = 114,912 fusion
  first-layer weights + 2,080 `D` + 5,856 + 4,944 binding + 9,328 + 3,920 slot
  encoders).  No speed factor is claimed (not measured).

The compressed model inherits `S_M`'s measured performance (dev raw MAE
`0.1201223`); it is **not** a new training result, and it still comes from the
original full training process, so it cannot be described as "trained without
the structural path".

## 3. N0 — the single new trajectory

Recipe identical to `S_M`; the only change is the two zero-slot multiplies.

Smoke (`smoke/smoke.json`): N0 init state item-for-item identical to the
source factory and to the released `S_M_init_state.pt` (including `kappa`);
switch off reproduces the `S_M` forward exactly; switch on gives slot absmax
`0.0`; task gradients on `W_A_S/W_A_C/W_E_S/W_E_C` are exactly zero, `D`
reconstruction gradient alive, all 408,651 parameters in the optimizer; label
shuffle unchanged.

Training (`N0.json`): 15,120 steps, 240 epochs, soup 236–240, wall 1189.2 s,
init hash `d90263758459…` (matches the source manifest), stream hash equals
the frozen schedule hash.  Health at epochs 1/40/120/240: slots entering the
encoders exactly zero and finite in every snapshot; by epoch 40 the *pre-zero*
slots are also exactly zero (the binding weights decayed to the denormal
floor); reconstruction loss stays alive (`2e-5…5e-5`); train MAE
`0.894 → 0.075`, mean pre-clip grad norm `≈4–10`, clip fraction `0.68 → 0.17–0.60`.

## 4. Main comparison (dev, 2000 rows; fit 8000 for bias/gap)

Positive gain = the first-named model is better.  `N0` bias is the single
fit-median calibration `-0.0118853`; `S_J`/`S_M` keep their original external
biases.

| arm | fit raw | fit cal | dev G0 raw | dev G0 cal | dev raw overall | dev cal overall | bias | cal fit→dev gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S_J | 0.040255 | 0.035538 | 0.102877 | 0.102329 | 0.123153 | 0.122601 | −0.021330 | +0.087063 |
| S_M | 0.037966 | 0.037964 | 0.097337 | 0.097347 | 0.120122 | 0.120130 | −0.000440 | +0.082167 |
| **N0** | 0.040562 | 0.039191 | 0.100102 | 0.099960 | 0.124952 | 0.124699 | −0.011885 | +0.085508 |

Paired bootstrap, 1000×, seed `20261003`, one shared index set across all
three models and raw/cal; G0 resamples the 1926 G0 rows, overall is stratified
by the fixed `1926/65/9` group counts; points are the full-dev values.

| contrast | endpoint | point | 95% CI | reading |
|---|---|---:|---|---|
| `gain_N0_vs_SM` | G0 cal | **−0.002613** | [−0.006588, +0.001113] | inconclusive (CI crosses 0) |
| `gain_N0_vs_SM` | G0 raw | −0.002765 | [−0.006676, +0.001009] | same direction |
| `gain_N0_vs_SM` | overall cal | **−0.004569** | [−0.009696, −0.000316] | N0 worse overall |
| `gain_N0_vs_SM` | overall raw | −0.004830 | [−0.009818, −0.000643] | N0 worse overall |
| `gain_N0_vs_SJ` | G0 cal | +0.002369 | [−0.001651, +0.006412] | numerically better, CI crosses 0 |
| `gain_N0_vs_SJ` | overall cal | −0.002098 | [−0.009001, +0.003580] | no S_J-level preservation |
| `gain_SM_vs_SJ` | G0 cal | +0.004981 | [+0.000740, +0.009020] | reproduces the source exactly |
| `gain_SM_vs_SJ` | overall cal | +0.002471 | [−0.006392, +0.009370] | reproduces the source |

**Pre-registered verdict for the primary endpoint (`gain_N0_vs_SM`, G0 cal):**
point `−0.002613` does not reach `−0.003`, the CI crosses zero, and the overall
calibration worsens by `+0.004569 > 0.001`.  This is the "other" row:
**inconclusive / calibration-robust-but-not-decisive**.  It is *not* the
negative-signal row (which requires point ≤ −0.003 and CI upper bound < 0), so
this seed/dev does **not** establish that the early binding path has a
required training-time role; it is also *not* the equivalence row, because the
CI is not contained in [−0.003, +0.003] and the overall endpoint worsens.
The raw endpoint points the same way as cal (no calibration-sensitivity flag).

Group table (`group_table.csv`; residual = y − pred, contribution =
Σ|error|/2000):

| group | n | S_J cal | S_M cal | N0 cal | N0 cal residual | N0 contribution |
|---|---:|---:|---:|---:|---:|---:|
| k=0 | 1926 | 0.102329 | 0.097347 | 0.099960 | −0.002230 | 0.096262 |
| k=−1 | 65 | 0.170088 | 0.171901 | 0.206481 | −0.061820 | 0.006711 |
| k≤−2 | 9 | 4.117891 | 4.621729 | 4.828242 | −4.543038 | 0.021727 |

The G0 point gap is small and CI-inconclusive; the overall worsening comes
mostly from `k=−1` (65 rows), where N0 is clearly worse than both controls.
Group contributions sum exactly to the overall cal MAE
(`contribution_identity`, `abs_diff ≤ 1.4e-17`).

Bootstrap witnesses (`summary.json`): identical predictions -> gain `0.0`;
swap -> sign flip; constant shift `0.5` -> gain `0.3999 ≤ 0.5`.

N0 deployment: same reduction as Phase A, dev `max|diff| = 1.43e-6 ≤ 1e-5`,
267,611 parameters, `N0_deploy_state.pt`; no further training.

## 5. What this round establishes and what it does not

Established (this seed, this reused dev):

* the trained `S_M` prediction function is exactly reproducible without the
  structural binding path; the reduced deployment model is valid and
  independent;
* from the first training step, fixing both slot tensors to zero yields
  G0 cal `0.099960` versus `S_M` `0.097347` (`−0.002613`, CI crossing 0) and
  overall cal `0.124699` versus `0.120130` (worse by `0.004569`, CI excluding
  the gain zero on the cal/raw overall endpoints);
* `S_M`'s original relative G0 advantage over `S_J` is reproduced exactly
  (`+0.004981`, CI `[+0.000740, +0.009020]`), so the matched-comparison
  pipeline is valid.

Not established / not claimed:

* that the early binding signal gives a generalisation benefit (the primary
  difference does not reach the pre-registered magnitude and its CI crosses
  zero);
* that N0 is equivalent to `S_M` (the overall endpoints worsen with CI
  excluding 0);
* that "all dictionaries are useless": task dictionary `D_L/V_L`, Sem108
  shell statistics, size2, topology25 and static relations remain in every
  model; only the explicit slot-binding path is removed;
* any multi-seed robustness claim, any official-valid/test claim, any
  attribution of the N0 gap to "dictionary knowledge transfer" specifically:
  removing the early task path also changes the `D` trajectory, the
  reconstruction/clip coupling and the constant-bias learning path, so those
  remain competing explanations.  The `k=−1` worsening is a 65-row cell and
  is not a decision basis.
