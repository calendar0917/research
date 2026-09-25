# `results/e2e_dictenv_a2` — round status

**`E2E-DictEnv-A2: COMPUTE-BUDGET-TRUNCATED`**

This result directory belongs to the frozen round `E2E-DictEnv-A2`
(preregistration `tracks/ksvd/notes/e2e_dictenv_a2_preregistration.md`,
commit `1813f53`; implementation commit `24d5286`; remote physical GPU1 only).

The user requested a compute-budget contraction while the round was running.
The contraction is recorded in
`tracks/ksvd/notes/e2e_dictenv_a2_compute_amendment.md` (commit `457c276`),
which **does not edit** the frozen preregistration and changes none of its
gates, thresholds or decision rules.  The round was stopped at the next safe
arm boundary; the minimum paired question continues in the separately labelled
protocol `E2E-DictEnv-A2-Lite` (`../e2e_dictenv_a2_lite/`).

## Completed artifacts (valid, unchanged)

| artifact | sha256 | meaning |
|---|---|---|
| `artifact_identity.json` | `1b967746…` | blocking reuse gate: `all_passed = true`, 26/26 entries (3 dictionary sha pins, 10 cache sha pins, 2 scalers, 6 exact-OMP recomputations, matched-init no-op), 130 s |
| `correctness.json` | `95f6583c…` | frozen Gate-0 correctness, 19/19 |
| `assignment_semantics.json` | `070a068f…` | REAL joint max abs change `3.6243410110473633`, INDEP `0.0`, marginals preserved |
| `dictionary_health.json`, `dictionary_health_{topo,indep,real}.json` | `0b253298…`, `ea2ddd22…`, `df74e5b4…`, `6724a81e…` | holdout normalised reconstruction TOPO `1.41e-05` / INDEP `2.109e-02` / REAL `1.971e-01`; used atoms 32/32/31; usage Spearman 0.9989/0.9989/0.9993 |
| `parameter_accounting.json` | `e5fca4e0…` | parameter accounting, pass (TOPO 97 487 / INDEP 109 263 / REAL 109 263) |
| `qualification.json` | `8a347408…` | Gate-0 qualification summary, `passed = true` |
| `continuity_v2.json` | `4b00f80d…` | continuity-v2 diagnostic (pooled REAL x 0.2971 / code 0.2464; INDEP x 0.2750 / code 0.4014; TOPO x 0.1350 / code 0.1329) — diagnostic only, never a gate |
| `smoke/smoke.json` (+`smoke/`) | `0d5bd356…` | GPU1 plumbing smoke; all curves finite, OMP dictionaries frozen, IHT dictionary gradient `6.399`; cannot influence any number |
| `omp_screen_topo.json` / `omp_screen_T0.json` / `curves/omp_screen_T0_curve.csv` / `states/omp_screen_T0_*` | `fac0ccb9…` / `59572c9a…` | the **complete** frozen Stage-1 TOPO arm (`T0`, horizon 320, frozen dictionary): soup valid MAE `0.12681294702464949`, best `0.13136472144449363 @ epoch 314`, soup members `[284, 288, 307, 314, 315]`, wall `2255.3 s` |

Provenance of every artifact above: `git_commit 24d528635fab49c082115d90592cb7d5938eeb37`,
`preregistration_commit 1813f53`, `official_test_loaded = false`.

## Truncation boundary

* stop time **2026-09-25 09:44:51 CST (+08:00)**, immediately after the `T0`
  arm's artifacts were written and before the `INDEP` arm produced anything.
* **partial checkpoints: none.**  The frozen `train_arm` writes state, curve and
  run JSON only after the final epoch, so stopping the runner mid-arm leaves no
  artifact at all; the `omp_screen_I0` training that had just started (≈3 s) left
  nothing on disk.
* the runner was stopped with `SIGTERM` at this boundary; the round was never
  abandoned and no result was deleted or edited.

## NOT RUN — deferred by user compute-budget amendment

`omp-screen` arms `I0`/`R0` (real/indep), `omp-decision`, `compression`,
`compression-decision`, `coder`, `formal`, `mechanism`, `liveness`,
`specificity`, and this round's own `decision`/`report`.
All are `DEFERRED_PENDING_USER_AUTHORIZATION`, not cancelled.

## Continuation

`tracks/ksvd/results/e2e_dictenv_a2_lite/` (`E2E-DictEnv-A2-Lite`): a
160-epoch, two-arm (`ATTR-REAL-OMP` vs `ATTR-INDEP-OMP`) matched screen plus, if
and only if that screen is not strong, the train-only PCA32 diagnostic.  Its
numbers are screen numbers: they may never be reported as this round's frozen
320-epoch Stage 1.

Official ZINC test data is not loaded by any artifact in this directory.
