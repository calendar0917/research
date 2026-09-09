# Post-v4 Residual Audit — output manifest

Diagnosis-only audit of frozen compact-v4-hinge residuals (seeds 0–3), validation split
only; official test never loaded. Full scientific note:
`tracks/ksvd/notes/post_v4_residual_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_post_v4_residual_audit.py`.

CSV/JSON artifacts below are generated outputs (git-ignored by policy; note + decision/
claim YAML records are the tracked scientific record).

## Files

| File | Contents |
|---|---|
| `validation_per_molecule_residuals.csv` | per-molecule (valid, 1000 rows): target, per-seed prediction/residual/abs error, mean_prediction, mean_residual, mean_abs_error, residual_std_across_seeds |
| `validation_master_table.csv` | residuals + frozen y-components (z_logP/z_SA/z_cycle), subgroup excess, all probe features (valid) |
| `component_attribution_correlations.csv` | per-seed + ensemble Pearson/Spearman of residual & abs error vs z_logP/z_SA/z_cycle (all-valid and Group A) with sign-consistency flags |
| `component_quantile_bins.csv` | quintile bins (signed mean residual, MAE, per seed) per component |
| `probe_correlations_groupA.csv` | probe block A–C correlations (rarity/OOV, graph complexity, atom/bond composition), per seed + ensemble, consistency flags |
| `probe_feature_bins_groupA.csv` | bias-vs-variance quintile bins for leading rarity/complexity features |
| `probe_1d_cv_groupA.csv` | 19 valid-internal-CV 1D probes (oracles + structural + composition): ΔMAE, residual R² |
| `pair_probe_cv_groupA.csv` | exact unary/pair/relation/OOV 2048-D hash probes, valid-internal CV |
| `continuous_pair_seed{s}.npz` | frozen-state continuous pair h_i⊙h_j matrices (240-D, Group A rows), per seed |
| `continuous_pair_cv_groupA.csv` | Probe F ridge CV per seed + ensemble |
| `state_summary_seed{s}.csv` | per-molecule frozen state statistics (patch/pair/global/topology/unified norms, deltas, centroids) + residual columns |
| `state_stats_correlations_groupA.csv` / `state_stats_crossseed_groupA.csv` | Probe G correlations and cross-seed table |
| `cross_partial_correlations_groupA.csv` | partial correlations (Pearson/Spearman), residual & abs error vs components/proxies under component control |
| `cross_feature_intercorrelations_groupA.csv` | Group-A feature/component inter-correlations |
| `cross_zSA_zlogP_difficulty_grid.csv` | two-way median grid MAE (z_SA × z_logP) |
| `cross_target_quintile_difficulty.csv` | target quintile difficulty table |
| `groupA_error_distributions.csv` | signed/abs error distribution stats per seed + ensemble (A and all-valid) |
| `top30_groupA_molecules.csv` | 30 worst Group-A molecules by mean abs error with feature context |
| `probe_summary_table.csv` | assembled probe summary (all families) |
| `decision_record.json` | Q1–Q14 answers + gated decision + evidence dump |
| `stage_*.json` | idempotency markers + per-stage summaries |
| `cache/` | frozen records cache (`v4_records_{train,valid}.pkl.gz`) |
| `figures/` | fig1–fig6 PNGs |

## Key numbers (Group A, valid, 4-seed ensemble unless noted)

- MAE ensemble 0.1186 (per-seed 0.1378/0.1323/0.1389/0.1408); Group A n=965.
- z_SA: abs Spearman −0.359 (4/4; partial given z_logP −0.314); signed nil (≤2/4).
- z_logP: signed Pearson +0.058 ens (4/4, weak); abs −0.186 (partial given z_SA −0.032).
- rarity rare≤5: abs 0.28–0.32/seed (4/4), ens 0.406; partial given components +0.317.
- composition N H1+ (type 8): signed +0.109 ens (4/4); partial given components +0.175.
- All fitted probes (pairs, continuous pairs, 1D structural/rarity/composition):
  ΔMAE within −0.0019…+0.0001 — below the Weak band (0.003); best residual R² +0.017.
- Decision: NO CLEAR SECONDARY SIGNAL (records/decisions/decision-post-v4-residual-audit-*.yaml).
