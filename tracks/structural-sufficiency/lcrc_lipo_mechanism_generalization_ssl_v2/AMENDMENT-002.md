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
