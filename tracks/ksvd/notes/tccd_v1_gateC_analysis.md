# TCCD-v1 Gate C — Dictionary Uniqueness

Pre-registration: `notes/tccd_v1_preregistration.md`.
Result: `results/tccd_v1/gateC.json`.
Commit: `5e3a4cf`.
Device: **local CPU**, using the validated vectorized path. Official test: never loaded.
K-SVD refit: **NO**; reused TCCD-v0 `D0`.

## Results

| arm | best internal-dev MAE | Top-5 soup MAE |
|---|---:|---:|
| TASK-D | 0.938099 | 0.898649 |
| DENSE | 0.403933 | 0.387409 |

`MAE_TASK-D - MAE_DENSE = 0.534167`.

Threshold: PASS only if `MAE_TASK-D <= MAE_DENSE + 0.005`.

**Gate C: FAIL / STOP.**

Conclusion: the matched generic dense local representation decisively outperforms the sparse task-coupled dictionary. The TCCD-v1 dictionary-specific inductive bias is not supported as a predictive advantage. Gate D is forbidden by the preregistered decision tree.
