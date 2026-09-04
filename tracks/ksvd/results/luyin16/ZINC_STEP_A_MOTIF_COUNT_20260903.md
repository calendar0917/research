# ZINC Step A：collision-free typed motif count

Protocol: `luyin16-zinc-step-a-motif-count-v1`

## Representation

- radius: `3`; centres: `every atom`
- typed rooted-WL rounds: `0/1/2/3`
- vocabulary: `train-fold-only top-K exact nested tuple tokens; no proposed-feature hash bins`; K=`2048` + OOV
- readout: `raw count + count / number of centres`
- Morgan reference: `RDKit-free count-ECFP/Morgan analogue`, bits=`2048`

## XGBoost results

| view | dim | train CV MAE | valid MAE | valid ensemble | test MAE after train+valid | test ensemble |
|---|---:|---:|---:|---:|---:|---:|
| `s` | 62 | 0.564518 | 0.560657 ± 0.002477 | 0.555028 | 0.592425 ± 0.002899 | 0.586197 |
| `wl_count` | 16392 | 0.449792 | 0.410354 ± 0.004164 | 0.385745 | 0.429296 ± 0.005399 | 0.406441 |
| `s_wl_count` | 16454 | 0.421302 | 0.370918 ± 0.002473 | 0.348409 | 0.376450 ± 0.002650 | 0.356410 |
| `morgan_count` | 4096 | 0.568823 | 0.534471 ± 0.006046 | 0.519231 | 0.525467 ± 0.003958 | 0.513992 |
| `s_morgan_count` | 4158 | 0.455885 | 0.419934 ± 0.002568 | 0.407740 | 0.417120 ± 0.006291 | 0.405486 |

## Matched GINE

- valid MAE: `0.466455 ± 0.093768`; ensemble `0.350679`
- test after train+valid refit MAE: `0.453499 ± 0.068011`; ensemble `0.361413`

Selected XGBoost view by train-only CV: `s_wl_count`.

Runtime: `2208.0s`; token cache hit: `False`.
