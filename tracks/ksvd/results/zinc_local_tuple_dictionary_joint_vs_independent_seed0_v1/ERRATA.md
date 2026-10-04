# ERRATA — label/access lineage questions

Two lineage questions raised before this round. Both are resolved from logs and hashes; the
distinction used throughout is **file read** vs **label/constant fitted** vs **new prediction/scoring
touching a held-out row**. Written before any new dev/target score was produced.

## E1. `stage_refine` mixed-12000 constant usage

**Finding (confirmed):** `zinc_long_cycle_audit.py` `stage_verify` iterates **train, valid and
test** and writes `audit/formula_verification_per_molecule.csv` with **12,000 rows including
held-out molecules**. `stage_refine` then reads that mixed table and fits the cycle/label
constants on the bulk (train+valid+test), writing `label_effective_cycle.csv` and
`stage_refine.json`; `train_cycle_audit_label.csv` carries the resulting mixed-fit label columns.
The old `zftd` decomposition reads `REFINE_JSON` constants, so those constants are **mixed-fitted**.

**Consequence:** any historical analysis that used `label_effective_cycle.csv`,
`train_cycle_audit_label.csv` label columns, or the `stage_refine.json` constants inherits a
train+held-out fit. This round does **not** use those label columns or constants as targets or
analysis inputs; `g = y − c` comes from `zcdm.TARGETS_NPZ` (fit-only refit; see E2).
The errata therefore downgrades the *provenance* of those old artifacts, not their historical
tables; nothing was recomputed or overwritten.

## E2. Was `build_fit_only_targets` sourced from train-only or mixed data?

**Finding (confirmed, with a correction of the earlier claim):**
`zcdm.build_fit_only_targets` **did read** `formula_verification_per_molecule.csv`
(the mixed 12,000-row file), then filtered `split == "train"` and refit the constants on fit rows
only (`fit_rows_cycle_free = 7711`). A verification refit from the same file, keeping only train
rows, reproduces the stored `fit_only_targets.npz` constants **bit-exactly** (max diff `0.0`);
new-vs-old constant deltas are small (e.g. `sigma_logP +0.000812`). Recorded provenance
(`target_provenance.json`): `k` diff rows 0, `c` max abs diff `0.0336543` on non-fit rows,
`official_valid_loaded = false`, `fit_rows_only = true`; zcdm `input_manifest.json` hashes match
the current files (`formula_verification_per_molecule.csv 7add70d3…`,
`train_cycle_audit_label.csv 3be64826…`, `stage_refine.json 96156fa1…`).

**Correction to the earlier statement**: the old claim "本轮完全未读取 held-out" (held-out was never
read this round) is too strong. The correct statement is:

* **file read**: yes — the mixed file containing held-out rows was opened/parsed;
* **label/constant fitted on held-out**: no — constants were fit on the 8000 fit rows only
  (verified bit-exact refit);
* **new prediction/scoring on held-out**: no — `official_valid_loaded=false` /
  `official_test_loaded=false` are asserted in every artifact of this round.

**Classification:** `ACCESS_LINEAGE_RESOLVED` (not `ACCESS_LINEAGE_UNRESOLVED`); the earlier
phrasing is corrected here, and this round's own pipeline re-verifies the fit-only property at
`--phase-a` (see `historical_anchor_checks.json`, `tuple_environment_checks.json`).
