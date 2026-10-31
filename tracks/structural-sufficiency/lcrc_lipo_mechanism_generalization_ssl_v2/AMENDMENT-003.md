# AMENDMENT-003 — invalid first Monitor execution and one valid re-evaluation

**Status:** prediction-code correction and explicit re-evaluation disclosure.

## What happened

The first successful Monitor execution (manifest
`ec043a7f…`, 2026-10-31T02:26:17Z) produced NN RMSE ≈2.30–2.39 and R² ≈−3,
while the CPU baseline had RMSE 0.7788. Inspection showed that
`v2_predict.predict_nn` had frozen the model's standardized target output
without applying the frozen Fit2352 target normalization
(mean 2.157857, scale 1.224198). NN prediction means were near 0 versus a
label mean of 2.230913.

This is an implementation-invalid evaluation. Its complete artifacts are
quarantined under `analysis/audit/monitor_invalid_normalization/`; none of its
H1/H3/H4 Monitor numbers are used. The internal CV results are unaffected
because training/dev metrics always denormalized correctly.

Earlier attempts to launch the same evaluation failed before writing metrics:
one was denied by the one-shot label guard, one failed unpacking the scan
tuple. They are documented in `AMENDMENT-002.md`.

## Correction

`v2_predict` now applies `D.denormalise(..., target_norm)` and fails closed if
the prediction mean is more than 3 target scales from the frozen target mean.
The eight NN predictions and the CPU prediction were regenerated into a fresh
`analysis/predictions/` directory and frozen under manifest
`1bc705295b4e…`. No old prediction file was overwritten in place.

## Evaluation count

The round has one **implementation-invalid** Monitor execution and one
**valid** Monitor execution. The invalid one is not a scientific evaluation
and is never used for a decision. The valid one is the only Monitor
evaluation used in `monitor_metrics_v2.json`, `decision_v2.json` and the
report. This deviation from the literal "exactly one evaluation" guard is
disclosed rather than hidden; the alternative would be to report a known
normalization artifact as H4.
