# AMENDMENT-002 — post-run analysis key resolution

**Status:** analysis-only correction after all GPU runs completed; no model,
data, fold, seed, recipe, SSL or decision-threshold change.

## Defect

`cv_analysis` looked up every recipe by the literal key
`F{fold}_{arm}_{recipe}_s{seed}`. Fold 1/2 only execute the selected recipe
under the registered `SEL` key (`F1_A_sel_s0`, `F2_A_sel_s0`, ...). As a
result the first pass of `cv_results_v2.json` contained:

- H2 for arm A on Fold 0 only, even though `F1/F2_A_R0_*` and
  `F1/F2_A_sel_*` both existed;
- H3 on Fold 0 only, even though `F1/F2_A_sel_*` and
  `F1/F2_A_SSL_sel_*` both existed.

No GPU result was missing or re-run. The bug only hid existing runs from the
analysis table.

## Fix

`_run_key_for_recipe` now uses the registered `SEL` alias for folds 1/2 when
(and only when) the frozen recipe lock selects that recipe. Fold 0 still uses
the explicit `F0_*_R1/R2/R3_*` keys. The same helper resolves the H3 scratch
arm. H1 and H4 are unaffected.

## Consequences

- H2 A now has three folds: gains 0.0629 / 0.0766 / 0.0879 RMSE, mean
  0.0758, all positive. H2 A stays promising. H2 C still has Fold 0 only
  because C-R0 was not registered on folds 1/2.
- H3 now has three folds: -0.0360 / +0.0111 / -0.0695 (scratch minus SSL),
  mean -0.0315, not consistent. H3 remains NOT_PROMISING despite the
  pretext gate passing.
- The corrected `cv_results_v2.json` is the primary CV artifact. The report
  discloses both the first-pass omission and the corrected values.

## Freeze / authorization

The protocol is re-frozen so its code hash covers this analysis helper.
The GPU authorization was already consumed; it remains pinned to the
pre-fix freeze and is not used to launch any further GPU job. The 3.5 h cap
and the ledger continue to apply.

## Addendum — prediction sidecar recipe lookup

The first `predict-nn` attempt treated `train.json["recipe"]` (a string such
as `"R3"`) as a recipe-config dict and failed before writing any prediction.
`v2_predict.predict_nn` now looks up `C.RECIPES[recipe_name]` for the dropout
value and writes the string recipe name into the sidecar. No prediction file
existed at the moment of the failure, so no frozen prediction was overwritten.
The protocol is re-frozen once more for this prediction-code fix; the frozen
manifest is created only after the fix.

## Addendum 2 — diagnostics recipe lookup

`v2_diagnostics` had the same two defects: it indexed the string `recipe` as
a dict, and it looked up `F1/F2_A_R3_s*` / `F1/F2_C_R3_s*` instead of the
registered `F1/F2_*_sel_s*` aliases. It now reads dropout via `C.RECIPES` and
resolves the `SEL` alias on folds 1/2. Diagnostics are inference-only on Fit
dev folds; no Monitor label is involved. The final freeze includes this fix.

## Addendum 3 — one-shot Monitor label read

The first `evaluate` attempt called `sha256_file(labels_fit_monitor.json)`
before `scan_label_artifact`; the v1 access guard correctly denied the
ungranted open. No Monitor value was constructed and no
`monitor_metrics_v2.json` was written. `v2_monitor_labels` now uses
`read_bytes_with_grant` exactly once, verifies the frozen SHA-256 from those
bytes, and scans the same in-memory text. The final freeze includes this fix;
the Monitor evaluation remains a single evaluation.

## Addendum 4 — unpack the single-pass scan

`v1_fit_labels._scan` returns `(values, meta, audit)`, not a dict. The first
corrected read constructed the 504 Monitor values and then failed on
`scanned["values"]`; no Monitor metric or `monitor_metrics_v2.json` was
written. The tuple is now unpacked. The final evaluation is the only
evaluation that writes metrics. Failed attempts are disclosed in the label
access audit trail; they did not produce or inspect Monitor errors.
