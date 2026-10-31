# Decision — LCRC Lipophilicity v3 chemistry-feature pilot

**Date:** 2026-10-31  
**Protocol:** `lcrc-lipo-chem-feature-pilot-v3`  
**Status:** **CPU PHASE COMPLETE — GPU NOT AUTHORIZED / NOT STARTED**  
**Freeze SHA-256:** `72f2e2142be7eedfe056bcbb948118d0d32ce73f3167fbb356847cf9acbfa14c`

## Current decision

No scientific verdict on the chemistry-feature hypothesis is possible yet. The CPU phase froze the protocol, features, folds, normalisation, model builders, audit suite, training registry and 0.30 A100 GPU-hour cap. The two unconditional GPU runs have **not** been executed, and the conditional third run is blocked by its pre-registered gate.

The descriptive Fold-1 error groups are **not** a treatment effect and were not used to change any run, threshold or feature.

## What is frozen

- Exactly `Chem12` (12 fixed RDKit descriptors) and `Chem13` (`Chem12` + `MolLogP`), order fixed.
- Only the `rho` global input is widened; local encoders, S2, `f`, atom51, bond13 and scalar9 are unchanged.
- Fold 1, seed 0, R3 only; A-R3 historical dev RMSE `0.74034`, C-R3 `0.72649`.
- Zero-extra forward equivalence and non-zero new-column gradients are audited.
- Added parameters: `A_Chem12` +768, `A_Chem13` +832, `C_Chem12` +768.
- Two unconditional runs and one conditional run, no other seed/fold/recipe/SSL/descriptor/full-Fit/Monitor work.
- Hard GPU cap 0.30 h; estimated 0.1217 h for all three runs.

## Pre-registered decision rules (unchanged)

Let `delta = RMSE(historical same-architecture) − RMSE(v3 chemistry model)`; positive favours the chemistry model. Tail stability uses the pre-registered top-2% historical-residual exclusion.

1. **`GENERAL_CHEM_PRIOR_SUPPORTED`**: Chem12 delta ≥ 0.03 and tail-robust, and Chem13 delta ≥ 0.03.
2. **`DIRECT_LOGP_PRIOR_ONLY`**: Chem12 delta < 0.03, Chem13 delta ≥ 0.03 and tail-robust. Interpret as the direct `MolLogP` prior, not a structural-representation breakthrough.
3. **`NO_GLOBAL_DESCRIPTOR_GAIN`**: both deltas < 0.03. Stop the global-descriptor supplementation route; any next step on atom-level chemical representations is a separate researcher decision.
4. **`INCONCLUSIVE`**: tail-dependent gain, `MolLogP` removing a Chem12 gain, or RMSE/MAE direction conflict.

The conditional `C_Chem12` run is allowed only if `A_Chem12` has RMSE delta ≥ 0.03, MAE delta > 0, and the gain is not tail-dependent. Its result is compared with C-R3 under the same rule.

## Next action

The researcher must authorize GPU work with the exact freeze SHA above. Without that authorization the runner refuses `train`. After the two A runs, the runner/analysis fills `analysis/table_v3.csv`, `analysis/results_v3.json`, `analysis/paired_v3.json` and `analysis/decision_v3.json`, and rebuilds `analysis/gpu_ledger_v3.json`.

No further CPU model search, feature combination, seed, fold, SSL or Monitor evaluation is authorized by this decision.
