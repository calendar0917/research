# Clean local-distribution frozen terminal test

Historical disclosure: official test was viewed by older routes; this is a controlled frozen evaluation.

## train_only

| candidate | five-seed mean AUC | seed/model ensemble AUC |
|---|---:|---:|
| `s_ta_mean` | 0.770149 | 0.773375 |
| `s_ta_mean_std` | 0.771340 | 0.777981 |
| `s_ta_all12` | 0.778625 | 0.785072 |
| `fixed 0.5*mean_std+0.5*all12` | 0.782153 | 0.785923 |

## train_valid_refit

| candidate | five-seed mean AUC | seed/model ensemble AUC |
|---|---:|---:|
| `s_ta_mean` | 0.769712 | 0.772519 |
| `s_ta_mean_std` | 0.772260 | 0.779005 |
| `s_ta_all12` | 0.777926 | 0.784901 |
| `fixed 0.5*mean_std+0.5*all12` | 0.781445 | 0.784961 |
