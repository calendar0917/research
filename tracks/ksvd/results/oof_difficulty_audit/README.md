# OOF Difficulty / Heteroscedasticity Audit — output manifest

Diagnostic gate: K-fold (K=5) **out-of-fold** predictions on the official TRAIN split with
the frozen compact-v4-hinge protocol (best-epoch checkpoint on nested inner-valid per outer
fold; inner 7200/800). Official test and validation splits never loaded; architecture/loss/
features untouched. Full scientific note:
`tracks/ksvd/notes/oof_difficulty_heteroscedasticity_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_oof_difficulty_audit.py`
(stages `splits → fold → states → features → table → analysis → predictor → figures → decision`).

CSV/JSON/PNG artifacts below are generated outputs (git-ignored by policy; note + decision/
claim YAML records are the tracked scientific record).

## Files

| File | Contents |
|---|---|
| `fold_assignments.csv` | deterministic 5-fold assignment of the 10 000 train molecules (fold_seed=0) |
| `inner_splits.csv` | per-outer-fold inner 7200-train / 800-inner-valid indexes with SHA-256 fingerprints |
| `oof_per_molecule.csv` | 10 000 rows × ~196 cols: molecule_id, outer_fold, target, per-seed predictions + signed/absolute errors, mean prediction/error, disagreement, rarity (per-fold inner-train counters), structure, topology (hinge 25-D), seed-0 internal states |
| `table_A_fold_results.csv` | per-fold OOF MAE / mean signed residual per seed + over seeds + ensemble |
| `table_B_difficulty_correlations.csv` | Spearman/Pearson of 56 features (rarity/structure/topology/states/disagreement) vs difficulty, per fold + consistency |
| `table_C_difficulty_predictor.json` | nested-OOF StandardScaler+Ridge variants on log(error): rarity / structure / states / disagreement / model-visible / all |
| `table_D_difficulty_quintiles.csv` | actual MAE by predicted-difficulty quintile per variant |
| `table_E_epistemic_signal.json` | disagreement–error Spearman (overall/per-fold), ensemble MAE & gain |
| `rarity_bin_table.csv` | fixed half-open bins of rare≤5 ratio vs OOF MAE / median / signed mean |
| `quintile_bins_*.csv` | quintile MAE tables for rare≤5 / OOV / disagreement |
| `signed_sanity_leading_features.csv` | signed vs abs Spearman of leading features (context only) |
| `target_component_attribution.csv` | abs/signed correlation of difficulty with frozen y-components (context) |
| `difficulty_predictions.npz` | cross-fitted difficulty predictions per variant + absolute_error |
| `decision_record.json` | gated verdicts (difficulty GO / epistemic GO) + evidence dump |
| `stage_*.json` | idempotency markers + per-stage summaries |
| `cache/folds/fold{f}_seed{s}_{meta,npz,state.pt}` | per-fold OOF predictions + frozen checkpoints + meta (trace, params, vocab, clamps, SHAs) |
| `cache/rarity_rows.csv`, `cache/static_rows.csv`, `cache/states_seed0.csv` | intermediate per-fold feature tables |
| `figures/` | fig1–fig6 PNGs (error↔rarity, error↔disagreement, pred-vs-actual, quintile MAE, rarity↔disagreement, ensemble gain) |

## Key numbers (official train, per-molecule OOF, 10 000 molecules; seeds 0+1)

- Pooled OOF MAE seed 0 / seed 1 / mean-of-seeds: 0.1758 / 0.1743 / 0.1750; 2-seed
  ensemble MAE 0.1604 (gain −0.0146, −8.3%, concentrated in the hardest error quintile).
- Cross-fitted difficulty predictor: model-visible Spearman **0.374** (5/5 folds), all
  features (incl. disagreement) **0.490**; predicted-hardest quintile MAE 3.67× easiest.
- Rarity: rare≤5 Spearman 0.309, OOV 0.277, mean-inverse-1+freq 0.339 (all 5/5); MAE
  ladder 0.133 (no rare token) → 0.417 (> 20% rare tokens).
- Disagreement (seed-spread) is the single best difficulty feature: Spearman 0.393
  (per-fold 0.359–0.420, 5/5).
- Determinism: fold 0/seed 0 rerun bit-identical; states forward-gate max diff 0.0/5 folds.
- Decision: **difficulty GO + epistemic GO** — remaining error is predictive difficulty /
  heteroscedastic conditional variance (rarity + disagreement + low-y axes); no signed
  second-layer structural signal; no structural v5 channel
  (`records/decisions/decision-oof-difficulty-heteroscedasticity-go-20260910.yaml`).
