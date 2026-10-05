# EXECUTION — zinc_cycle_level_transfer_terminal_test_seed0_v1

Commit chain (isolated branch `task/zinc-cycle-level-transfer-terminal-test-seed0-v1`, not
pushed/merged): a83aae1 (protocol/contract/runner freeze) → 181519d (identity/folds/OOF
artifacts + init-witness fix via deterministic reload) → da36222 (D_full + manifest freeze +
heldout-access disclosure) → (statistics-direction fix) → final docs commit. Source bodies/
heads/prep/targets/prototype table: commit 947d2837c4936ef1276fcbbf0e9189e7d44ace23
(zinc_component_supervision_fulltrain_confirmation_seed0_v1) + 5a510d9
(zinc_cycle_prototype_transfer_cpu_v1), all read-only, SHA-verified.

## Phase log (wall clock, CPU-only, ≤8 threads, FP32)

| phase | seconds | notes |
|---|---|---|
| identity | ~8 | source replays exact (train/valid COMP y_cal, q_raw max diff 0.0) |
| folds | <1 | 355 classes → 3 class-grouped folds [3334, 3333, 3333]; held∩fit=∅ |
| train_folds | 49.3 | 6 trajectories (R/D × 3 folds), gate computed after all |
| fold_metrics | <1 | deterministic reload recovery (see deviation 1) |
| dfull | 12.0 | D_full 3975 params, gen seed 20261003, 23700 steps, train kAcc 0.9995 |
| freeze | <1 | terminal_eval_manifest committed before any test prediction |
| terminal | 13.4 | valid replay first, then test (post-freeze) |
| analyze + replay checks | ~3 | consolidated analysis + replay_checks.json (all pass) |

Total ≈ 90 minutes including exploration, inside the 120-minute hard cap; no new head
training was started after minute 60.

## Deviations and honest disclosures (none change the frozen recipe)

1. **Init-witness bookkeeping bug, fixed by deterministic reload (before any test read).**
   `train_folds` captured the `hidden_init` witness from the post-training soup state instead
   of the saved init state, so the in-run witness printed `hidden_init_equal=False`. The saved
   `R_fold{j}_init_state.pt` / `D_fold{j}_init_state.pt` are the true init states; the added
   `fold_metrics` phase reloaded them and verified R/D hidden init equality and equality to Q's
   untrained seed-0 hidden init for all 3 folds (all True), and recomputed the per-fold
   summaries from the saved soup states with the OOF arrays bit-identical to the saved
   `OOF_predictions.npz` (max abs 0.0). No retraining, no recipe change; the gate numbers were
   computed from the (always-correct) OOF arrays and were never affected.

2. **Pre-freeze test engineering touch (disclosed, superseded).** Before the roster freeze,
   the official test loader path was exercised for engineering feasibility only
   (loader/T25-build/SHA/prep-application/body-forward shapes; early artifact
   `test_body_inference.npz`). No roster system's test y-MAE was computed before the manifest
   commit; the frozen `terminal` phase recomputed everything from the frozen path. Recorded in
   `heldout_access.json` (test: 2 accesses — engineering pre-freeze + terminal post-freeze).

3. **Prior-round test values read during exploration.** Published test values of the OLD
   round (a different, larger system, zinc_full_decomposition_valid_test_confirmation_v1)
   were read while studying the test engineering path. They informed nothing: the roster and
   the purchase gate were decided by train-only OOF before the terminal read.

4. **Statistics-direction bug in the terminal analyzer, fixed after the test read
   (deterministic correction).** The first `terminal` run computed the paired-bootstrap
   comparisons from raw predictions instead of |pred − y| errors (crashed in the VALID eval
   before any test read; key reversal in the comparison tuple). Fixed; the `terminal` phase was
   re-run as a deterministic replay of the same frozen predictors (valid/test y-MAEs
   reproduced to the last digit: 0.122653/0.117406/0.096694/0.096150 and
   0.092619/0.085107/0.084719/0.081174). All point statistics in the report come from the
   fixed code; no training, calibration, epoch/rule/routing selection occurred after the test
   read (explicitly forbidden and not done).

5. **Test label provenance.** Test y comes from the official split; g/ell/s from
   gvae_full_properties.npz via smi_line join; c recomputed with the frozen full-train
   constants and k = round(snapped c). The audit CSV's own `label_cycle_component` column uses
   the old 8k constants and was NOT used. y = g + c exact (1.7e-18).

## Access ledger (official splits)

- valid (exposed, reused from the prototype round): identity replay + motivating-row
  explanation + C valid markers — all post-freeze reporting.
- test: engineering pre-freeze touch (deviation 2) + the single terminal read under the frozen
  manifest (roster SUM_Q/B/H/C; no test-driven selection; unconditional report as authorized).

## Artifacts (tracks/ksvd/results/zinc_cycle_level_transfer_terminal_test_seed0_v1/)

`source_identity.json`, `T25_group_folds.npz` (+`folds.json`), `R/D_fold{j}_init/soup_state.pt`,
`R/D_fold{j}_curve.json`, `OOF_predictions.npz`, `folds_summary.json`, `gate.json`,
`D_full_{init,soup}_state.pt` + `D_full_{curve,meta}.json` + `D_full_train_predictions.npz`,
`terminal_eval_manifest.json`, `heldout_access.json`, `valid/test_row_predictions.npz`,
`valid/test_evaluation.json`, `valid_test_gaps.json`, `analysis.json`, `replay_checks.json`,
`test_body_inference.npz` (superseded engineering artifact, kept for the audit trail),
`runner.py`, PROTOCOL.md, METHOD_CONTRACT.md, REPORT.md, DECISION.md, EVIDENCE_SCOPE.md.
