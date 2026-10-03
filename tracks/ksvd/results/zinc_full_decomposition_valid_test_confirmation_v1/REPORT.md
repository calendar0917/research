# REPORT — zinc-full-decomposition-valid-test-confirmation-v1

Round type: **full-scale confirmation with one frozen official-valid/test
evaluation**.  The fixed method from `c11342e` — a frozen dictionary chemistry
branch plus an *independent* learned cycle head — is retrained on **all 10,000
official-train rows** with two paired seeds, calibrated once on train, frozen,
and then evaluated exactly once on official-valid (1,000) and official-test
(1,000).

The deployed model is
`P_s = h_s(x) + q_s(T) + b_P,s` with **no true `c` at inference** (`T` is the
existing Full topology25 input).  `Y_s = f_s(x) + b_Y,s` is the matched control.
`h_s` is supervised on `g = y − c`; `q_s` on `c`.  `c` is the label-derived
cycle component; **this round therefore uses training-side decomposition labels**
and is not a single-scalar-`y` baseline.

## Answer to the four questions

### 1. What is the real test level, and the measured valid−test gap?

MAE in `y` units; positive `gain = MAE(Y_cal) − MAE(P_cal)` = improvement.

| split | arm | raw seed0 | raw seed1 | raw mean | raw ens | cal seed0 | cal seed1 | cal mean | cal ens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train (10k) | Y | 0.038638 | 0.053635 | 0.046136 | 0.038530 | 0.037172 | 0.041771 | 0.039471 | 0.031852 |
| train (10k) | P | 0.042608 | 0.044390 | 0.043499 | 0.036041 | 0.042448 | 0.042503 | 0.042475 | 0.035708 |
| official-valid | Y | 0.118618 | 0.117695 | 0.118156 | 0.109857 | 0.118392 | 0.114879 | 0.116635 | 0.108338 |
| official-valid | P | 0.120876 | 0.117719 | 0.119297 | 0.110915 | 0.120485 | 0.117521 | 0.119003 | 0.111033 |
| official-test | Y | 0.094589 | 0.096643 | 0.095616 | 0.085955 | 0.094549 | 0.092097 | 0.093323 | 0.084133 |
| official-test | P | 0.091429 | 0.087964 | 0.089697 | 0.080861 | 0.091286 | 0.087825 | **0.089556** | **0.081007** |

`gain(Y→P)`, same seed, positive = better:

| split | raw s0 | raw s1 | raw mean | raw ens | cal s0 | cal s1 | cal mean | cal ens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | −0.003970 | +0.009245 | +0.002638 | +0.002489 | −0.005276 | −0.000732 | −0.003004 | −0.003857 |
| official-valid | −0.002258 | −0.000024 | −0.001141 | −0.001057 | −0.002093 | −0.002642 | **−0.002367** | −0.002694 |
| official-test | +0.003160 | +0.008679 | +0.005920 | +0.005093 | +0.003263 | +0.004272 | **+0.003768** | +0.003126 |

**Measured `valid MAE − test MAE` (same model/seed):**

| | raw s0 | raw s1 | raw ens | cal s0 | cal s1 | cal ens |
|---|---:|---:|---:|---:|---:|---:|
| Y | +0.024029 | +0.021051 | +0.023903 | +0.023842 | +0.022782 | +0.024205 |
| P | +0.029447 | +0.029755 | +0.030053 | +0.029199 | +0.029696 | +0.030026 |

Test is easier than valid by ≈0.023 (Y) / ≈0.029 (P); the gap is **not** a fixed
0.02 and is larger for `P`.

**0.09 target, per split, honestly:**
* official-valid: `Y_cal` mean `0.1166`, `P_cal` mean `0.1190` — **neither
  reaches 0.09**.
* official-test: `Y_cal` mean `0.0933` (does not reach 0.09); `P_cal` mean
  `0.08956` (**just under 0.09**), per seed `0.09129 / 0.08783`, secondary
  `P_cal` ensemble `0.08101` and `Y_cal` ensemble `0.08413`.

So the 0.09 goal is **not** reached on the official-valid split, and is reached
only marginally on official-test, mostly by the secondary prediction ensemble.

### 2. Does the old +0.0109 inner-dev signal reproduce at full scale?

**No — not on official-valid; the fixed method is not validated there.**

* official-valid pre-registered gate: both-seed cal gain > 0 **fails**
  (seed0 `−0.002093`, seed1 `−0.002642`); mean `−0.002367 < 0.003` fails.
  `G0` (`k=0`) is preserved (`P−Y` worsen `−0.000853 / +0.000037`, both
  `≤ 0.001`), and identity / no-true-`c` / single-cal / freeze-replay all pass.
* official-test (reported unconditionally, **not** used to flip the gate): both
  seeds improve (`+0.003263 / +0.004272`; mean `+0.003768`; raw mean `+0.005920`)
  and `G0` improves (mean `+0.004059`).

The old `+0.010860` mean gain came from a *reused 2,000-row train-side dev*.
The full-scale, official held-out result is **valid: negative; test: positive**,
i.e. the signal did **not** stably transfer to the official-valid split.  The
gate is left as **FAIL** (see §8 rule "valid未过, test改善 → report both, do not
retro-change the gate").

### 3. Where do the gain / remaining gap come from?

Group contribution `Σ|error|/N_split` (cal, seed 0; contributions sum to the
overall MAE, contribution gains sum to the total gain):

| split | group | n | Y contrib | P contrib | contribution gain |
|---|---|---:|---:|---:|---:|
| valid | k=0 | 965 | 0.088559 | 0.087736 | **+0.000823** |
| valid | k=−1 | 30 | 0.005035 | 0.003858 | **+0.001177** |
| valid | k≤−2 | 5 | 0.024797 | 0.028890 | **−0.004093** |
| test | k=0 | 948 | 0.084503 | 0.082068 | **+0.002435** |
| test | k=−1 | 44 | 0.006858 | 0.005791 | **+0.001067** |
| test | k≤−2 | 8 | 0.003188 | 0.003427 | **−0.000239** |

* `Q` vs `c` (y units): train overall `0.0118 / 0.0116` vs constant `0.1554`;
  valid `0.0309 / 0.0301` vs `0.1526`; test `0.0067 / 0.0045` vs `0.2081`.
  Discrete levels are learned (`k=−1` valid `0.032/0.019`, test `0.058/0.036`);
  the rare extreme `k≤−3` is not (train n=7 `≈11.1`, valid n=1 `≈20.6`).
* **Valid failure is one extreme row.** The single `k≤−3` valid molecule is
  `valid:0172` (`k=−6`, `y=−20.3405`).  There the chemistry branch is *good*:
  `h = 0.4612` vs `g = 0.4695` (error `0.008`), but `q = −0.173` vs `c = −20.810`
  (error `20.64`), so `P_cal = 0.292` (error `20.63`) vs `Y_cal = −0.447`
  (error `19.89`).  The cycle head — not the chemistry branch — carries the loss.
  Removing this row, the valid group gains are positive.
* **Test has no `k≤−3` row**, so the tail loss is small and the bulk `k=0`
  (`+0.002435`) and `k=−1` (`+0.001067`) gains dominate.
* On the 10k train fit, `P_cal` (`0.042475`) is slightly *worse* than `Y_cal`
  (`0.039471`); the improvement, where it exists, is a held-out effect, not a
  better fit.  Per-row identity `y − P_cal = e_g + e_c − b_P` holds to `≤4e−15`
  on valid and test.

### 4. What route should be kept, and what discriminating evidence is missing?

Keep the fixed decomposition interface (frozen `h` + independent `q(T)` +
single train bias) as the working method, but **do not promote `P` to a new
baseline yet**: it is positive on official-test for both seeds and preserves
`G0`, yet it **fails the official-valid gate** and never reaches 0.09 on valid.
The missing discriminator is *whether the rare extreme cycle tail (`k≤−3`,
`c` down to `−41.6`) is learnable from this topology25 input at all*.  This round
shows it is not learned by the fixed head (`valid:0172` `q≈0` vs `c≈−20.8`) while
the chemistry branch predicts `g` there correctly.  That question is not settled
by this round and must **not** be bought with post-test tuning.  No new training
is authorised here.

## Data, supervision and identity

* Train: all 10,000 official-train rows, stable ids `train:%04d`.
* One all-10k **train-only** prep object (`all_train_prep.npz`, sha256
  `a7dbcc43…`): historical `build_fold_inputs` semantics with `fit_idx = all`;
  `D_fit` (K-SVD, sha256 `b0c5da98…`) and `U` refit on all 231,664 `phi` rows.
  Per-column standardisers equal the all-train standardisers (the only gap is on
  15 constant `anchor` columns, recorded in `prep_meta.json`).
* `c = (label_effective_cycle_snapped − mu_cycle)/sigma_cycle` with the frozen
  constants; `g = y − c`; `median_train(c) = 0.0004627120960`.  The supervision
  target `c` is **label-derived** (source `target_decomposition.npz`, sha256 in
  the manifest); it is never a model input.
* `T` is the actual canonical Full topology25 model input (10000×25,
  sha256 `2129ec2f…`), original column order, train-only standardisation.
* Four Full models are 408,651 parameters; each `Q` head is 3,777; a deployed
  `P` is 412,428.  `Y_s`/`H_s` share the same fresh init per seed (identical
  state hash), identical data stream and budget; only the task target differs.
* `official_valid_loaded`/`official_test_loaded` are `false` for every training
  artifact and `true` only after the frozen manifest; first reads are
  `2026-10-03T11:41:13Z` (valid) and `2026-10-03T11:41:26Z` (test).

## Frozen gate, bootstrap, deployment

Pre-registered **valid** gate: (1) both-seed cal gain > 0 and mean ≥ 0.003;
(2) each-seed `G0` worsen ≤ 0.001; (3) identity / train-only / no-true-`c` /
single-cal / freeze-replay.

| condition | value | met |
|---|---|---|
| both seeds valid cal gain > 0 | −0.002093 / −0.002642 | **no** |
| mean valid cal gain ≥ 0.003 | −0.002367 | **no** |
| each-seed `G0` worsen ≤ 0.001 | −0.000853 / +0.000037 | yes |
| identity / no-true-`c` / single-cal / replay | all pass | yes |
| **valid gate** | | **FAIL** |

Fixed paired bootstrap (1000 draws, seed 20261003, row resampling, joint across
arms/seeds), point from the full split:

| stat | valid point [95%] | test point [95%] |
|---|---:|---:|
| gain `Y→P` cal mean | −0.002367 [−0.016335, +0.007543] | +0.003768 [−0.001759, +0.009023] |
| gain `Y→P` raw mean | −0.001141 [−0.012511, +0.009144] | +0.005920 [+0.000668, +0.011117] |
| `G0` gain cal mean | +0.000408 [−0.003466, +0.004232] | +0.004059 [−0.000005, +0.008230] |

The test cal interval includes 0; the test **raw** interval excludes 0.  These
intervals describe only these held-out rows and these two seeds.

Deployment / no-leak evidence: `deploy_wrapper.py` returns
`f(x)+b_Y` (Y) or `h(x)+q(T)+b_P` (P) and takes no label table; wrapper vs cached
predictions differ by `≤1.7e−6`; permuting labels changes the output by exactly
`0`; raw-soup reload replay `≤1.5e−6`; the fresh-graph adapter reproduces
canonical `patch_cont`, `global_context`, `topology_features` and `phi` with
max-abs `0.0` on 128 train rows.

## ERRATA

See `ERRATA.md`.  In brief: the earlier "7/9 `k=−2` solved" is wrong (it is 5 of
7); `idx 2210` (`k=−12`) is substantially recovered (`q≈−40.4` vs `c≈−41.6`) and
is not a total failure; the `P`-vs-`O` residual increment is concentrated in
three old inner-dev rows (`2052, 5050, 7507`) that do not map onto the new
held-out splits; the `K` overall values match `group_table.csv`; and this round
uses one signed convention, `residual = y − prediction`.

## Boundaries

* This is a **label-supervised decomposition** result: `H` is trained on `g` and
  `Q` on `c`.  It is not a single-scalar-`y` baseline and not SOTA.
* Only the official-valid split was ever used for interpretation.  The
  official-test split was read once, after freezing; it is now exposed and can
  no longer be described as pristine for future rounds.
* No dictionary-vs-capacity-matched-MLP ablation was run; "dictionary
  specificity" is not claimed.
* The method is fixed; no width/λ/WD/loss/optimizer/seed search was performed,
  and none is authorised after seeing test.