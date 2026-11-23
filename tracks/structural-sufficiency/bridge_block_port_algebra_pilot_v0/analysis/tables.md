# BBPA-v0 tables (auto-generated from `results/summary.json`)

Protocol: `bbpa-v0-bridge-block-port-algebra-pilot`  
Training commit: `f607d9b534dedd9302ce700f3c27fa9214147f36`  
Generated: 2026-11-23T02:34:39Z

## Primary table

| model | seed | params | train RMSE@best | Dev RMSE | Dev MAE | best epoch | epochs | wall s | GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLOCK_BAG | 0 | 417857 | 0.6863 | 0.7831 | 0.6148 | 54 | 89 | 154.6 | 0.0410 |
| LOCAL_PORT | 0 | 417857 | 0.1982 | 0.6727 | 0.5019 | 63 | 98 | 709.0 | 0.1948 |
| TREE_PORT | 0 | 417857 | 0.0658 | 0.6565 | 0.4933 | 145 | 180 | 1277.6 | 0.3528 |

## Primary and subgroup contrasts

Positive RMSE delta = second model better. All are paired molecule bootstraps (5000 draws).

| contrast | n | RMSE delta | 95% CI | MAE delta | 95% CI |
|---|---:|---:|---|---:|---|
| delta_comp | 504 | +0.0162 | [-0.0314, +0.0641] | +0.0087 | [-0.0283, +0.0461] |
| delta_comp_bridges_ge_11 | 241 | +0.0637 | [+0.0028, +0.1285] | +0.0595 | [+0.0097, +0.1103] |
| delta_comp_bridges_ge_6 | 427 | +0.0611 | [+0.0139, +0.1087] | +0.0427 | [+0.0060, +0.0807] |
| delta_comp_bridges_le_5 | 77 | -0.1740 | [-0.3148, -0.0307] | -0.1804 | [-0.2892, -0.0666] |
| delta_comp_diameter_ge_3 | 478 | +0.0243 | [-0.0225, +0.0735] | +0.0148 | [-0.0217, +0.0527] |
| delta_comp_diameter_ge_6 | 354 | +0.0519 | [+0.0012, +0.1014] | +0.0381 | [-0.0038, +0.0779] |
| delta_comp_diameter_le_2 | 26 | -0.1169 | [-0.3554, +0.1032] | -0.1038 | [-0.2858, +0.0694] |
| local_minus_block | 504 | +0.1103 | [+0.0507, +0.1687] | +0.1128 | [+0.0655, +0.1611] |
| tree_minus_block | 504 | +0.1265 | [+0.0642, +0.1882] | +0.1215 | [+0.0726, +0.1718] |

## Dev error by bridge count and block-tree diameter

| stratifier | bin | n | BLOCK_BAG RMSE | LOCAL_PORT RMSE | TREE_PORT RMSE |
|---|---|---:|---:|---:|---:|
| n_bridges | 1 | 3 | 1.0214 | 1.2077 | 1.3447 |
| n_bridges | 2 | 10 | 0.6295 | 0.4411 | 0.6959 |
| n_bridges | 3-5 | 64 | 0.5501 | 0.7423 | 0.9129 |
| n_bridges | 6-10 | 186 | 0.7700 | 0.6684 | 0.6106 |
| n_bridges | 11+ | 241 | 0.8456 | 0.6551 | 0.5914 |
| block_tree_diameter | 0-2 | 26 | 0.6660 | 0.6796 | 0.7965 |
| block_tree_diameter | 3-5 | 124 | 0.6613 | 0.7372 | 0.7771 |
| block_tree_diameter | 6-8 | 193 | 0.8694 | 0.6567 | 0.6089 |
| block_tree_diameter | 9-11 | 108 | 0.7282 | 0.6528 | 0.6179 |
| block_tree_diameter | 12+ | 53 | 0.8700 | 0.6055 | 0.4965 |

## State diagnostics (TREE checkpoint, Dev)

- Dev molecules: 504; TREE directed states: 10940; bridges: 5470.
- TREE state per-dim std: mean 0.8907, min 0.4901, max 1.6524; near-constant fraction 0.000.
- TREE state effective rank: participation ratio 12.98, entropy rank 21.52, top-1 share 0.175.
- LOCAL state effective rank: participation ratio 9.40.
- q effective rank: participation ratio 14.53; u_global: 1.72.
- Same TREE checkpoint, TREE vs LOCAL mode: state L2 diff 5.374 mean / 16.004 max; q diff 4.613 mean / 22.381 max.
- Diameter ≤2: 26 molecules, max state diff 0.000000 (exact coincidence). Diameter ≥3: 478 molecules, mean max state diff 12.080; all deep molecules differ.

### Same-checkpoint mode difference by diameter

| diameter | molecules | fraction state >1e-6 | mean max state diff | fraction q >1e-6 |
|---|---:|---:|---:|---:|
| 0-2 | 26 | 0.00 | 0.000 | 0.00 |
| 12+ | 53 | 1.00 | 13.765 | 1.00 |
| 3-5 | 124 | 1.00 | 9.802 | 1.00 |
| 6-8 | 193 | 1.00 | 12.236 | 1.00 |
| 9-11 | 108 | 1.00 | 13.589 | 1.00 |

## Historical D0 coordinates (different recipes)

| model | D0 RMSE | delta vs TREE (positive = TREE better) | delta vs LOCAL |
|---|---:|---:|---:|
| GINE_REF | 0.5266 | -0.1299 | -0.1461 |
| ACRA_REF | 0.5802 | -0.0763 | -0.0925 |
| PATH_COUNT_RIDGE | 0.6167 | -0.0398 | -0.0560 |
| PSC_POOL_BAG_PAIR | 0.6871 | +0.0306 | +0.0144 |

## Worst 10 TREE errors (raw units)

| row_id | bridges | diameter | label | TREE pred | TREE abs err | LOCAL abs err | BLOCK abs err |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3144 | 13 | 7 | 4.300 | 1.722 | 2.578 | 2.201 | 1.721 |
| 2241 | 4 | 4 | 4.090 | 1.771 | 2.319 | 1.237 | 0.622 |
| 3739 | 13 | 9 | 3.880 | 1.617 | 2.263 | 1.644 | 1.599 |
| 712 | 5 | 4 | -1.400 | 0.699 | 2.099 | 1.202 | 0.453 |
| 3171 | 1 | 1 | 0.750 | 2.739 | 1.989 | 0.721 | 1.287 |
| 2804 | 22 | 13 | 0.750 | 2.621 | 1.871 | 1.972 | 0.882 |
| 3788 | 5 | 3 | 1.100 | 2.934 | 1.834 | 0.753 | 0.324 |
| 1667 | 12 | 6 | -1.410 | 0.386 | 1.796 | 1.713 | 1.858 |
| 2099 | 7 | 4 | 1.500 | 3.170 | 1.670 | 1.799 | 1.304 |
| 2855 | 11 | 10 | 4.420 | 2.757 | 1.663 | 1.258 | 0.871 |

## Route

**CAPACITY_WITHOUT_COMPETITIVENESS** (pre-registered rule; point estimate, single seed).
