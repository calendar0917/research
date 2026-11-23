# BBPA-v0 results

This directory contains the frozen outputs of the three seed-0 arms and the
post-run analysis. All training runs executed commit
`f607d9b534dedd9302ce700f3c27fa9214147f36`; per-file SHA-256 values are in
`../analysis/run_index.json`.

## Per-run directories

`BLOCK_BAG_S0/`, `LOCAL_PORT_S0/`, `TREE_PORT_S0/` each contain:

* `manifest.json` — arm, seed, commit, hashes, parameter ledger, recipe, best
  metrics, wall/GPU time, peak VRAM, zeroed categories;
* `curves.csv` — per-epoch train/Dev metrics;
* `state_best.pt` — best-Dev checkpoint (used for all reported predictions);
* `state_final.pt` — final-epoch checkpoint;
* `predictions_best_fit.csv`, `predictions_best_dev.csv` — per-molecule raw and
  standardised predictions in original label units;
* `predictions_best_fit.npz`, `predictions_best_dev.npz` — the same arrays in
  machine-readable form;
* `train.log` — training log (not part of the numerical analysis).

## Analysis outputs

* `summary.json` — primary table, contrasts, route label, stratified errors,
  worst errors, historical coordinates, state diagnostics, provenance and
  budget;
* `main_table.csv`, `stratified.csv`, `worst10_tree.csv`, `historical_d0.csv` —
  flat tables;
* `../analysis/checks.json` — 17/17 engineering checks;
* `../analysis/tables.md` — formatted tables generated from `summary.json`;
* `../analysis/paired_bootstrap.json`, `state_diagnostics.json`,
  `run_index.json`, `budget.json` — supporting analyses.

Sign conventions: `Δ_comp = RMSE(LOCAL_PORT) − RMSE(TREE_PORT)`, so positive
means TREE better. Historical `delta_vs_tree = historical − TREE`, so negative
means TREE worse. No official Valid/Test labels were read at any point.