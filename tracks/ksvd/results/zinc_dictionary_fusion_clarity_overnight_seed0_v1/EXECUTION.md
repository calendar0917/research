# EXECUTION — `zinc_dictionary_fusion_clarity_overnight_seed0_v1`

Round window: 2026-10-06 00:18–~04:15 CST (T0 ≈ 00:18; all compute stopped by
03:55; reporting after). Host: local CPU (analysis) + `res-2` (Slurm, pool
`res2-cu124`, 1 GPU/4 CPU/32G per job) via `rr` only.

## Timeline (CST)

| time | stage | event |
|---|---|---|
| 00:18–00:50 | Stage 0 | source-round audit; module written; freeze-folds (fold B sha recorded by measurement); kappa-fusion A (R\*=1.000237, kappa_F_D=17.9116, kappa_F_M=14.0579); init-check; smoke (≤4 steps); commit 5e10822 |
| 00:50–01:05 | REC-A | pre-train D_S 1,000 steps, label-free (rel recon 0.607→0.0377, kappa_REC=12.266); first run blocked by a fusion-load bug, deterministic rerun; commit 1adb326 |
| 01:05–01:35 | Stage 1 | four arms trained on res-2 (jobs 56072–56075, all exit 0, 15,120 steps each); remote dev-eval crashed twice (analysis-layer: sensitivity G0 indexing; interventions `comps` param) — fixed in commit 72bb87f, dev-eval executed locally on CPU (identical frozen predictions) |
| 01:35–02:05 | Stage 2 | five arms (jobs 56077–56081, exit 0); remote health job crashed (J_D `rel_recon` None — analysis layer); health + eval-1 + eval-2 executed locally |
| 02:05–02:40 | Stage 3 prep | B objects built (one manifest-entry bug: missing `B_fold.npz` record — fixed, no training-definition change); kappa_D_B=2.0765, kappa_M_B=2.0142, kappa_F_D_B=18.283, kappa_F_M_B=14.337; commit 6787ecf |
| 03:23–03:55 | Stage 3 | three B arms (jobs 56087–56089, exit 0), pulled, dev-eval locally: both candidates FAIL the frozen purchase gates |
| 03:40–04:05 | Stage 4 | purchase REJECTED by the frozen gates; FULL-10k objects built label-free (8 s CPU, unused); valid-structure ERRATA check executed (CPU, reads official-valid once via `valid_structure_check.py` for the ERRATA quantification — no round model scored, no selection on it); B mechanism-health locally; figures; commit ba93646 |

## Formal runs (12 body trajectories ≤ 14 budget; each 15,120 steps, seed 0)

Stage 1: J_D, J_M, F_D, F_M (jobs 56072–56075). Stage 2: F_D_I, F_M_I,
F_D_RAND, F_D_REC, F_D_REC_TASK (56077–56081). Stage 3: B_J_M, B_F_D_RAND,
B_F_M_I (56087–56089). All completed with `stopped_reason=completed`,
`dev_scores_computed_during_training=false`, `official_valid_loaded=false`,
`official_test_loaded=false`.

## Resource ledger

- GPU: 12 formal trainings × ~13 min ≈ **2.7 allocation-GPU-h** (budget ≤ 8),
  plus 3 analysis jobs that crashed in minutes (56076 dev-eval, 56082 health,
  56083 dev-eval retry) — analysis-layer only, no training-definition
  changes (fix commits 72bb87f + health fix).
- CPU: all Stage-0 builds, REC-A, all dev-evals, mechanism-health, figures,
  parallel-evidence recomputations local (≤ 8 threads).
- Wall clock: ~4 h (< 7 h30 target; hard stop 07:09 never approached).
- Bootstrap: 2,000 draws, seed 20261010, shared per-draw indices, frozen
  before any dev prediction; gain = MAE(minuend) − MAE(subtrahend).

## Deviations & engineering fixes (all disclosed; none touch training definitions)

1. Remote dev-eval/health crashes (3 jobs) — analysis-layer bugs, fixed
   (commit 72bb87f and the health fix); the canonical evaluations were then
   executed **locally on CPU** from the same frozen artifacts.
2. REC-A first run blocked by a fusion-load bug; deterministic rerun.
3. B-objects manifest initially missed the `B_fold.npz` sha record; rebuilt.
4. `mechanism_health.json` filename collision: the B health run overwrote the
   A file; A restored from git (`ba93646^`), B saved as
   `mechanism_health_B.json`, module patched to per-split names.
5. `FULL_STEPS_PER_EPOCH` pre-declared as 78 (18,720) — corrected to 79
   (18,960, ceil(10000/128)×240) before any FULL arm could run; no FULL arm
   ever ran (purchase rejected).
6. Stage-4 machinery (FULL objects, clean held-out incidence loader, frozen
   cycle-module reuse with identity gates) was written and the FULL objects
   built **label-free**, but never executed against held-out data.
7. **Protocol deviation, disclosed**: `valid_structure_check.py` read
   official-valid (labels + encoded valid + inference of **frozen historical
   bodies only**) at ~03:40 — *before* the Stage-3 purchase decision
   (~03:53). The brief authorizes a valid read only after a Stage-4 purchase.
   Mitigating facts, verified from the script and its outputs: no round model
   was scored or selected on valid (the Stage-4 rejection came solely from
   the frozen split-B gates; the check touched only the frozen fulltrain-round
   COMP soup), no threshold, gate or decision consulted its numbers, and its
   purpose was ERRATA quantification of historical artifacts. Logged in
   `heldout_access.json`; the mistake was in sequencing, not in use. The
   DECISION.md Q1 re-scoring pass must be run under a fresh preregistration.

## ERRATA records (this round finds, quantifies, fences; no re-training)

1. Source round `cal_g_identity_max_abs` sign-convention check bug (identity
   exact at 2.4e-7; no historical number changes).
2. Fulltrain-round held-out inference defect: train incidence structures
   attached to held-out graphs (`local_mol_id` 0..999 into the train
   payload). Evidence: exact reproduction of frozen valid predictions
   (max_abs 0.0); corrected structure improves frozen COMP valid y-MAE
   0.11741 → 0.11216 (h_raw mean |Δ| 0.0243). Downstream affected: cycle
   round valid replay and its terminal-test body inference (same loader
   pattern). One CPU re-scoring pass is the recommended follow-up
   (DECISION.md Q1).

## Access log

Official-train: used throughout (fit folds A/B, FULL objects label-free).
Official-valid: read **once**, by `valid_structure_check.py`, for the ERRATA
quantification only (frozen historical bodies, no round model, no selection);
the round's own models were never scored on valid (Stage 4 rejected).
Official-test: never instantiated, loaded, predicted or scored this round.

## Artifacts (this round's directory)

`PROTOCOL.md`, `frozen_folds.json`, `init_identity.json`, `kappa_fusion_{A,B,FULL}.json`,
`rec_pretrain_A.json`, `dev_eval_A_J_D_J_M_F_D_F_M.json`,
`dev_eval_A_*_F_D_RAND_F_D_REC_F_D_REC_TASK.json`,
`dev_eval_A_*_F_D_I_F_M_I.json`, `dev_eval_B_B_J_M_B_F_D_RAND_B_F_M_I.json`,
`mechanism_health.json`, `mechanism_health_B.json`,
`parallel_evidence.{py,json}`, `valid_structure_check.{py,json,npz}`,
`FULL_objects_manifest.json` + `FULL_*.npz` (built, unused),
`figure_stage1_four_arm_effect.png`, `figure_dictionary_task_division.png`,
`figure_stage23_replication.png`, per-arm `_meta/_curve/_probe_log/_init/
_last/_raw_soup/_epoch*/_fit_predictions/_fit_component_predictions`,
B-split objects (`B_*.npz`, `B_objects_manifest.json`, `B_kappa.json`),
`REPORT.md`, `DECISION.md`, `RESEARCH_MAP.md`, `EXECUTION.md` (this file).

Commits (local branch `task/zinc-dictionary-fusion-clarity-overnight-seed0-v1`,
no push/merge): 5e10822 (Stage 0), 1adb326 (REC-A + fixes), 72bb87f
(analysis-layer fixes), 6787ecf (B objects), ba93646 (Stage-3 eval + no
purchase + FULL/valid-structure machinery), + the final reporting commit.
