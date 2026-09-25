# E2E-DictEnv-A2-Confirm — REPORT

* protocol `e2e_dictenv_a2_confirm`; commit `76f65af3a7d0a58f12efca2751d25d0f583bc2ed`
* preregistration `tracks/ksvd/notes/e2e_dictenv_a2_confirm_preregistration.md` (commit `4b9e206`, rule commit `4b9e206`)
* parent round `E2E-DictEnv-A2` at `24d528635fab49c082115d90592cb7d5938eeb37` (prereg `1813f53`); predecessor `E2E-DictEnv-A2-Lite`
* seed 0; horizon 320; device policy: physical GPU1 only
* continuation regime: `fresh_matched_320`

## Q1 — exact resume or fresh matched 320?

`fresh_matched_320`.  exact continuation cannot be proven for every arm (missing: REAL:optimizer_state,scheduler_state,rng_states,loader_order_state,soup_member_states_for_full_trajectory; INDEP:optimizer_state,scheduler_state,rng_states,loader_order_state,soup_member_states_for_full_trajectory); both arms restart as fresh matched 320-epoch runs, no pseudo-resume

Evidence (recorded in `continuation_mode.json`): the completed 160-epoch lite
segments persist only the best model state and the Top-5 soup *average*; no
optimizer state, no scheduler state, no RNG state, no loader/sampler state, no
individual soup-member states, and the frozen trainer always restarts at epoch 1
(`accepts_resume_argument=false`, `writes_optimizer_state=false`).  Exact
continuation therefore cannot be proven for either arm, so both arms restart as
fresh matched 320-epoch runs (no pseudo-resume, no mixing).

## Q2 — 320-epoch paired result

* REAL soup = 0.1369818167760386
* INDEP soup = 0.1425485412654816
* G_pair_320 = 0.005566724489443009  (material bar 0.003)
* reaches the bar: `True`

## Q3 — practical comparison with the completed TOPO arm

* TOPO soup (320, parent, never retrained) = 0.12681294702464949
* Delta_vs_TOPO = 0.010168869751389115 (positive = REAL worse)
* within the +0.003 practical tolerance: `False`

## Q4 — final label

`ATTRIBUTED_PAIRING_SUPPORTED_BUT_NOT_COMPETITIVE`

## Diagnostics

* late window [241, 320]: mean 0.006538768048032944, median 0.006299895591568197, positive fraction 0.7625, first 0.0018247949490323712, last 0.021618235073052328
* 160 -> 320 gap: 0.005561161399818965 -> 0.005566724489443009 (KEPT, sign_reversed=False)
* REAL soup change -0.01738934519328178; INDEP soup change -0.017383782103657736

## Deferred

* seed 1 replication — DEFERRED_PENDING_USER_AUTHORIZATION
* PCA32 dense control — DEFERRED_PENDING_USER_AUTHORIZATION
* continuity-v2 diagnostic — DEFERRED_PENDING_USER_AUTHORIZATION
* IHT-10/30/100/200 coder qualification — DEFERRED_PENDING_USER_AUTHORIZATION
* task-coupled E2E sparse dictionary (E0/E1/E2) — DEFERRED_PENDING_USER_AUTHORIZATION
* REAL->INDEP code-pairing-removal mechanism — DEFERRED_PENDING_USER_AUTHORIZATION
* node-only / edge-only pairing interventions — DEFERRED_PENDING_USER_AUTHORIZATION
* DenseTied specificity control — DEFERRED_PENDING_USER_AUTHORIZATION
* TOPO-OMP retraining — DEFERRED_PENDING_USER_AUTHORIZATION
* K64 dictionary — DEFERRED_PENDING_USER_AUTHORIZATION
* s12 dictionary — DEFERRED_PENDING_USER_AUTHORIZATION
* official test split — DEFERRED_PENDING_USER_AUTHORIZATION
* any architecture / hyper-parameter sweep — DEFERRED_PENDING_USER_AUTHORIZATION

Official test loaded: `false`.  Stage status: see `stage_status.json`.
