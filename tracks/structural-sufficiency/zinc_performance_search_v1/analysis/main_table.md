# valid table (official valid-1000, raw-y MAE)

| model | policy | seed | params | epochs | best-valid | ep | W20 | final | train@best | d(row172) | d(top10) | gpu-h | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A-R0 | online | 0 | 413505 | 600/47400 | 0.076224 | 377 | 0.078081 | 0.078179 | 0.014680 | 0.074735 | 0.065096 | 0.301 | zinc_recipe_trunk_v2/runs/xBssl_P3N_s0 |
| B-R0 | online | 0 | 494193 | 600/47400 | 0.078537 | 489 | 0.079082 | 0.079076 | 0.005510 | 0.072510 | 0.061802 | 0.345 | zinc_dual_channel_composer_v1/runs/L_s0 |
