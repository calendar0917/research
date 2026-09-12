# Raw-Graph → Patch-System Sufficiency Audit — results

**Verdict: Decision Case D — `CURRENT_PRE_NEURAL_PATCH_SYSTEM_SUFFICIENCY_NOT_REFUTED`.**

Zero-full-training / representation-family audit of the `G_raw -> F_pre(G)`
boundary of compact-v4.  Official valid and official test were never loaded;
all neighbour manifests were SHA-256 locked before any target was read.

## Headline numbers

| quantity | value |
|---|---:|
| `LB(P0)` | 7.95e-5 (2 raw-non-isomorphic token classes) |
| `LB(P1)` / `LB(P2)` / `LB(P3)` | 0.0 / 0.0 / 0.0 |
| P3 raw-non-isomorphic collisions | 0 |
| `eta(RAW)` (probe) | 0.6337 |
| `eta(WL)` | 0.7455 |
| `eta(SP)` | 0.6024 |
| `eta(PATCH_LOCAL)` | 0.5865 |
| `eta(PATCH_PAIR)` | 0.5888 |
| `eta(PATCH_FULL)` | 0.5459 |
| `Delta_pre = eta(PATCH_FULL) - eta(RAW)` | **-0.0878** (95% CI [-0.0999, -0.0748]) |
| 800 replication `Delta_pre` | -0.0909 |

The historical tokenizer alias is real at P0 but fully resolved by the
continuous patch descriptor at P1.  The current patch-system geometry is more
target-local than both generic raw-graph references.

## Files

* `audit_protocol_lock.json`, `raw_graph_inventory.json`,
  `pre_neural_input_inventory.json`, `split_inventory.json`
* `hard_signature_lock.json`, `p[0-3]_signature_summary.json`,
  `raw_graph_isomorphism_checks.json`, `functional_collision_checks.json`,
  `hard_aliasing_target_summary.json`, `hard_aliasing_lower_bounds.json`
* `raw_wl_lock.json`, `raw_sp_lock.json`, `patch_soft_signature_lock.json`,
  `normalization_stats.json`, `distance_scale_stats.json`
* `neighbors_<representation>_<split>.jsonl.gz` (12 manifests),
  `neighbor_manifest_hashes.json`
* `phaseU_integrity.json`, `target_metric_lock.json`,
  `random_neighbor_baseline.json`, `soft_geometry_metrics.csv`,
  `soft_geometry_bootstrap.json`, `collision_consistency.json`,
  `knn_predictor_diagnostics.json`, `integrity_tests.json`
* `final_decision.json`, `top1_pre_neural_hypothesis.json`,
  `answers_q1_q20.json`, `figure_data.json`, `figures/`

Reproducer: `uv run python -m tracks.ksvd.experiments.luyin16.zinc_raw_graph_patch_sufficiency_audit all`
(after `lock`, `hard_signatures`, `hard_targets`, `soft_u`, `soft_y`,
`integrity`, `decision`, `figures`).

Note: this whole directory is git-ignored by repo policy (`results/**/*.json`,
`*.csv`, `*.png`, `*.jsonl.gz`); the durable scientific record is the note and
the claim/decision YAMLs.
