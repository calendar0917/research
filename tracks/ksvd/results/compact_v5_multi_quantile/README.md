# compact-v5 multi-quantile regression — result index

Stage: **Compact-v5: Uncertainty-Aware Multi-Quantile Regression** (objective-only;
compact-v4-hinge representation frozen). Final note:
[`../../notes/compact_v5_multi_quantile_regression.md`](../../notes/compact_v5_multi_quantile_regression.md).

**Verdict: NO-GO** for a confirmed multi-quantile benchmark improvement.
Stage-1 seed-0 reached Case A (valid q50 Δ +0.0114, width–|error| 0.2543), but the
frozen 4-seed validation confirmation is only **Mild/Mild** (point +0.003686, 3/4
seeds; width 0.2111, 4/4 seeds) and the one-time benchmark **test does not
transfer** (v5 0.139403 ± 0.002510 vs v4 0.136885 ± 0.005552; paired −0.002518,
1/4 seeds).

## Tracked / untracked layout

Git tracks only this `README.md` and `run_map.json`-style summaries are ignored
(`tracks/*/results/**/*.json|csv|png` are git-ignored). The durable facts live in
`records/runs/` (promoted) and the note; the raw tables here are local evidence.

## Contents (local, git-ignored)

| path | what |
|---|---|
| `run_map.json` | run-id map: seed-0 variants, multi-seed λ=0.25, terminal test runs |
| `stage1_seed0_summary.json` | full analysis summary (primary table, diagnostics, multi-seed table, benchmark test table, decision) |
| `decision_record.json` | machine-readable decision record (mirror of the note §13) |
| `seed0/primary_table.csv` | λ sweep + controls |
| `seed0/width_vs_error_lambda0{10,25,50}.csv` | per-molecule q10/q50/q90, width, |error|, rarity, subgroup |
| `seed0/width_quintile_table_lambda0*.csv` | width-quintile MAE ladder |
| `seed0/rarity_bin_table_lambda0*.csv` | rarity-bin MAE/width ladder |
| `seed0/subgroup_table_lambda0*.csv`, `seed0/rarity_group_table_lambda0*.csv` | A/B/C and easy/medium/rare |
| `seed0/top_widest_lambda0*.csv` | widest-interval molecules |
| `seed0/multiseed_table.csv` | v4-vs-v5 paired validation deltas |
| `seed0/benchmark_test_table.csv` | v4-vs-v5 frozen benchmark test |
| `figures/fig1..fig6_*_lambda0{10,25,50}.png` | the six required figures |

## Reproduce

```
uv run research run zinc_patch_path_pooling --study zinc-context-gap \
  --config tracks/ksvd/configs/luyin16/zinc_compact_v5_quantile_lambda025.yaml \
  --seed N --mode terminal \
  --set evaluation.selection_checkpoint_test=true \
  --set evaluation.skip_train_valid_refit=true
uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v5_multi_quantile_analysis \
  tracks/ksvd/results/compact_v5_multi_quantile/run_map.json
```

## Promoted runs

Stage-1 controls/λ sweep, multi-seed λ=0.25, and the frozen benchmark test are
promoted under `records/runs/` (12 runs; see `STATE.yaml` `promoted_records`).
