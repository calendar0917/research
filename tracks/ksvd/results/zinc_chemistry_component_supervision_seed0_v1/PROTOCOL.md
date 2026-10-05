# PROTOCOL — `zinc_chemistry_component_supervision_seed0_v1`

Frozen **before** either formal trajectory ran and before any new SUM/COMP dev score was read.
One question, two matched arms, one seed, the existing fresh fold, no search.

## 1. Question

Fix the current M input / skeleton / total objective `g`. Does adding **true chemical-component
supervision** (`ell = (logP − MU_LOGP)/sigma_logP`, `s = g − ell`) on the *same* fresh fold change
the calibrated internal-dev `g`-MAE relative to the matched total-only arm? And, conditional on
the model that accepts the supervision, is one component's fit-vs-dev behaviour distinct enough to
name a next research object?

This is a task-decomposition / supervision experiment, not a dictionary advantage experiment. It
does not prove that the original g-only model failed because of one chemical term. Even an internal
`g`-MAE < 0.09 is not an official-valid/test `y`-MAE and is not a deployable `y` result.

## 2. Fixed sources (read-only)

* Frozen fresh-fold artifacts:
  `tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/`
  (`fresh_fold.npz`, `fresh_targets.npz`, `fresh_tuple_payload.npz`, `fresh_prep.npz`,
  `fresh_kappa.json`, `fresh_manifest.json`), byte-checked by `load_fresh_objects`.
* Fresh-fold runner `zinc_local_tuple_fresh_fold_replication_seed0_v1.py` (`load_train_only_raw_rows`,
  `build_prepared_data`, recipe constants, schedule).
* `fresh_targets.npz` constants: `sigma_logP=1.434428173759835`,
  `sigma_SA=0.8327498022638992`, `mu_SA=-3.1924626828735625`,
  `sigma_cycle=0.2885551506645267`, `mu_cycle=-1.3447189184664423e-05`; `MU_LOGP=2.4570953396190123`.
* Fold: `new_fit` sha `2a21cb8771f6e24cfb4a5cc50602cf0b4390db9c4367f8bb910052f9f0aebbcb`,
  `new_dev` sha `270ab4126b0f0413f1f6ff7ada2e9914bbaca91ae50cc65bbd8ca2673d8f9357`,
  schedule sha `7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65`
  (position stream and the independent global graph-ID stream are both re-checked).
* Official valid/test are never loaded, instantiated, predicted or scored.
* The `zinc_direct_bond_relation_seed0_v1` result is read only to close that line; its `T` is never
  loaded, no beta coordinate is continued.

## 3. Component labels (identity, no redefinition of g)

```
ell_i   = (logP_i - MU_LOGP) / sigma_logP
s_i     = g_i - ell_i                     # "residual chemistry component"
s_SA_i  = (SA_i - mu_SA) / sigma_SA       # interpretation reference only
epsilon = s_i - s_SA_i
```

`g` bytes stay the frozen fresh-fold `g`; the identity `g = ell + s` is checked in float64.
Interpretation rule: `s` is **never** called pure SA when `epsilon` is non-negligible. The
pre-frozen SA-interpretation threshold (fit-G0 `epsilon`-MAE ≤ 1e-3 **and** fit-G0 max ≤ 1e-2) only
restricts wording; it never changes labels, the main target, or the training configuration.
Observed (frozen before scoring): fit-G0 `epsilon`-MAE `0.001512`, max `0.008953` → **threshold not
met**, so results are positioned as `ell + residual chemistry`, not as SA failure.

## 4. Arms (identical skeleton / init, only supervision differs)

Both arms start from the original fresh-fold untrained M init. The reader's last linear (39→1) is
replaced by 39→2 `[hat_ell, hat_s]`; both rows and both biases are half the original, so
`hat_g = hat_ell + hat_s` reproduces the original M output at construction (checked ≤ 1e-5).

| arm | optimised loss | params |
|---|---|---:|
| `SUM` | `L_g = MAE(hat_g, g)` | 297,539 |
| `COMP` | `L_g + 0.5·(L_ell + L_s)` | 297,539 |

`L_ell = MAE(hat_ell, ell)`, `L_s = MAE(hat_s, s)`. `0.5` is pre-fixed, not scanned / adapted /
dev-tuned. Both component targets are already in `g` contribution units. SUM logs the two component
losses but they are detached and never enter its backward graph. All state keys/values and
parameter counts are identical across arms; the standard forward returns `(n,)` total `g`; training
reads the two `(n,)` components cached from the *same* body computation (never a second forward).

## 5. Training recipe (both arms identical)

seed 0; Adam lr `1e-3`, coupled wd `1e-5`, global grad clip `5.0`; batch 128; 240 epochs
(15,120 steps each); FP32; no AMP / DDP; soup epochs 236–240 by parameter mean, no dev-selected
epoch, no extension. Same GPU regime / deployment commit / driver / torch / CUDA / Python for both
arms.

## 6. Evaluation and gate

Per arm: `p_raw = hat_ell + hat_s`, `b = median(g_fit − p_raw_fit)`, `p_cal = p_raw + b`; exactly
one fit-median bias, never a dev bias. Main comparison is COMP vs SUM:
`gain = MAE(SUM) − MAE(COMP)` (positive = improvement), paired row-bootstrap 1000 draws,
seed `20261008`, shared indices per endpoint per draw.

Candidate gate (all three):
1. dev-G0 calibrated `g` gain ≥ `0.003`;
2. dev-overall calibrated `g` gain ≥ `0.003`;
3. G0 paired 95% CI lower bound > 0.

* all three → `COMPONENT_SUPERVISION_CANDIDATE`;
* positive point but not all three → `DIRECTIONAL_NOT_CONFIRMED`;
* negative / near-zero → `NO_CANDIDATE` (with the actual interval; a CI crossing 0 is not
  "equivalence");
* practical equivalence only if both G0 and overall calibrated CIs lie inside ±0.003.

Raw results are reported alongside but are not a hard veto. If only the calibrated gate passes, the
result is labelled calibration-dependent and not attributed to a representational gain. A fixed
sensitivity drops the single dev row with the largest SUM calibrated error; it never replaces the
gate and never deletes training data.

## 7. COMP component diagnostics (delivered regardless of the gate)

Using only the trained COMP output (no new fitting): fit/dev, G0/overall raw-MAE, signed
mean/median, component std, fit→dev gap, the median-fit constant-predictor MAE on each split and the
model/constant ratio, per-k summaries (rare dev groups only described).

Pre-frozen fit-adequacy marker per component: `FIT_ADEQUATE` iff G0-fit model raw-MAE ≤
`0.20 ×` its G0-fit median-constant MAE; `DEGENERATE_TARGET` if that constant MAE is ≈ 0 (no ratio
constructed). The marker limits the diagnostic only and is not an extra performance threshold.

Error cancellation (raw only): `e_ell = hat_ell − ell`, `e_s = hat_s − s`, `e_g = e_ell + e_s`,
`triangle_gap = mean(|e_ell| + |e_s| − |e_g|) ≥ 0`; report mean `|e_ell|`, `|e_s|`, `|e_g|`,
opposite-sign fraction, `triangle_gap`, the sum of the two component MAEs and the total, and the
global `g` bias separately. The two component MAEs are never added as the `g` error budget.

Diagnostic conclusion buckets: fit-insufficient; single-component target; broad generalisation gap;
favourable cancellation. These are task-responsibility evidence in the new COMP model, **not** a
causal decomposition of the original g-only model.

## 8. Budget / stopping

Target wall clock ≤ 120 min, hard ≤ 150 min; no new training or diagnostic started after minute
100. GPU budget ≤ 0.8 allocation-GPU-hours, ≤ 2 independently allocated GPUs; both arms submitted
in parallel to `res-2 --pool res2-cu124` when legal. Per-task CPU threads ≤ 8 local, ≤ 4 remote.
Physical GPU identity (UUID/PCI) is captured *inside* the task before any cancellation decision;
`CUDA_VISIBLE_DEVICES=0` alone is never treated as a shared-card conflict.

Science status and execution status are recorded separately. No official-valid/test confirmation.
A performance candidate earns one frozen confirmation design (not executed here); otherwise a
component diagnostic can motivate one new design directed at a named property.
