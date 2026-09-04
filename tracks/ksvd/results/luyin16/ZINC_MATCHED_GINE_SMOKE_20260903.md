# ZINC sufficiently trained matched GINE baseline

Protocol: `luyin16-zinc-matched-gine-smoke-v1`

The architecture matches the GINE used in the Step-A report; this run changes only the ZINC training budget to 2000 epochs and batch size 32, with validation-based epoch selection.

| phase | MAE |
|---|---:|
| official train → valid (mean ± std) | 0.566864 ± 0.000000 |
| official train+valid → test (mean ± std) | 0.604222 ± 0.000000 |

Selected epochs: `[2]`.

Per-seed results:

| seed | selected epoch | valid MAE | test MAE |
|---:|---:|---:|---:|
| 0 | 2 | 0.566864 | 0.604222 |

Runtime: `4.9s`.
